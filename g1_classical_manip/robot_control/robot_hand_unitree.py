# Threaded Dex3 / Dex1 controllers for g1_classical_manip.
#
# Re-wrapped from unitreerobotics/xr_teleoperate (Apache-2.0) via the user's
# G1_teleop_dex fork: teleop/robot_control/robot_hand_unitree.py. See NOTICE.
#
# The upstream Dex3_1_Controller welds its control loop to XR hand-retargeting
# and a multiprocessing.Process, and its state subscriber reads only
# motor_state[].q. Here we keep the DDS plumbing, the _RIS_Mode bitfield, the
# command-message construction, and the joint-index enums VERBATIM, but:
#   * replace the process model with a plain threaded controller mirroring
#     G1_29_ArmController (subscribe thread + publish thread + ctrl_lock);
#   * drop all XR / hand_retargeting code; the public API is command(side, q7);
#   * the subscribe loop additionally reads motor_state[].dq, .tau_est and
#     press_sensor_state[].pressure -- the grasp-verification signals consumed
#     by ee/dex3.py.
#
# Higher-level grasp presets / verification live in g1_classical_manip/ee/.

import threading
import time
from enum import IntEnum

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

try:
    import logging_mp

    logger_mp = logging_mp.getLogger(__name__)
except Exception:
    import logging

    logger_mp = logging.getLogger(__name__)

Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"

LEFT = "left"
RIGHT = "right"


class _Buffer:
    def __init__(self):
        self._d = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._d

    def set(self, d):
        with self._lock:
            self._d = d


class _RIS_Mode:
    # Verbatim from upstream: packs (id:4, status:3, timeout:1) into one byte.
    def __init__(self, id=0, status=0x01, timeout=0):
        self.motor_mode = 0
        self.id = id & 0x0F
        self.status = status & 0x07
        self.timeout = timeout & 0x01

    def _mode_to_uint8(self):
        self.motor_mode |= (self.id & 0x0F)
        self.motor_mode |= (self.status & 0x07) << 4
        self.motor_mode |= (self.timeout & 0x01) << 7
        return self.motor_mode


class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6


class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


class Dex3Controller:
    """Threaded dual-Dex3 controller. DDS must already be initialised
    (ChannelFactoryInitialize) before construction.

    Public API:
        command(side, q7)            set the target for one hand (7-vector)
        command_dual(left7, right7)  set both targets
        get_q(side) -> (7,)          latest measured joint positions
        get_state(side) -> dict      {'q','dq','tau','press'} each (7,)
        stop()                       stop the control threads
    """

    def __init__(self, fps=100.0, kp=1.5, kd=0.2, simulation_mode=False):
        self.fps = fps
        self.kp = kp
        self.kd = kd
        self.simulation_mode = simulation_mode
        self._running = True
        self._ctrl_lock = threading.Lock()

        self._left_idx = list(Dex3_1_Left_JointIndex)
        self._right_idx = list(Dex3_1_Right_JointIndex)

        self._left_pub = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self._left_pub.Init()
        self._right_pub = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self._right_pub.Init()
        self._left_sub = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self._left_sub.Init()
        self._right_sub = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self._right_sub.Init()

        self._left_state = _Buffer()
        self._right_state = _Buffer()

        self._left_msg = self._init_cmd_msg(self._left_idx)
        self._right_msg = self._init_cmd_msg(self._right_idx)

        self._sub_thread = threading.Thread(target=self._subscribe, daemon=True)
        self._sub_thread.start()
        self._wait_for_state()

        # seed targets with current measured q so the publish loop holds posture
        self._left_target = self.get_q(LEFT)
        self._right_target = self.get_q(RIGHT)

        self._pub_thread = threading.Thread(target=self._publish, daemon=True)
        self._pub_thread.start()
        logger_mp.info("Dex3Controller initialised.")

    def _init_cmd_msg(self, joint_idx):
        msg = unitree_hg_msg_dds__HandCmd_()
        for jid in joint_idx:
            msg.motor_cmd[jid].mode = _RIS_Mode(id=jid, status=0x01)._mode_to_uint8()
            msg.motor_cmd[jid].q = 0.0
            msg.motor_cmd[jid].dq = 0.0
            msg.motor_cmd[jid].tau = 0.0
            msg.motor_cmd[jid].kp = self.kp
            msg.motor_cmd[jid].kd = self.kd
        return msg

    def _wait_for_state(self, timeout=5.0):
        t0 = time.time()
        while self._left_state.get() is None or self._right_state.get() is None:
            if time.time() - t0 > timeout:
                raise TimeoutError("Dex3Controller: no /state within timeout")
            time.sleep(0.01)
            logger_mp.warning("[Dex3Controller] waiting for dds state...")
        logger_mp.info("[Dex3Controller] dds state ok.")

    def _read_one(self, msg, joint_idx):
        n = len(joint_idx)
        q = np.zeros(n)
        dq = np.zeros(n)
        tau = np.zeros(n)
        press = np.zeros(n)
        for i, jid in enumerate(joint_idx):
            ms = msg.motor_state[jid]
            q[i] = ms.q
            dq[i] = ms.dq
            tau[i] = ms.tau_est
            # press_sensor_state[j].pressure is a per-finger tactile ARRAY (not a
            # scalar); reduce to a scalar (max contact) for grasp verification.
            try:
                p = msg.press_sensor_state[jid].pressure
                press[i] = float(np.max(p)) if np.ndim(p) else float(p)
            except Exception:
                press[i] = 0.0
        return {"q": q, "dq": dq, "tau": tau, "press": press}

    def _subscribe(self):
        while self._running:
            lm = self._left_sub.Read()
            rm = self._right_sub.Read()
            if lm is not None:
                self._left_state.set(self._read_one(lm, self._left_idx))
            if rm is not None:
                self._right_state.set(self._read_one(rm, self._right_idx))
            time.sleep(0.002)

    def _publish(self):
        dt = 1.0 / self.fps
        while self._running:
            t0 = time.time()
            with self._ctrl_lock:
                lt = np.array(self._left_target, copy=True)
                rt = np.array(self._right_target, copy=True)
            for i, jid in enumerate(self._left_idx):
                self._left_msg.motor_cmd[jid].q = float(lt[i])
            for i, jid in enumerate(self._right_idx):
                self._right_msg.motor_cmd[jid].q = float(rt[i])
            self._left_pub.Write(self._left_msg)
            self._right_pub.Write(self._right_msg)
            time.sleep(max(0.0, dt - (time.time() - t0)))

    # ------------------------------------------------------------------- API
    def command(self, side, q7):
        q7 = np.asarray(q7, dtype=float).reshape(Dex3_Num_Motors)
        with self._ctrl_lock:
            if side == LEFT:
                self._left_target = q7
            elif side == RIGHT:
                self._right_target = q7
            else:
                raise ValueError(side)

    def command_dual(self, left7, right7):
        self.command(LEFT, left7)
        self.command(RIGHT, right7)

    def get_state(self, side):
        st = (self._left_state if side == LEFT else self._right_state).get()
        return {k: v.copy() for k, v in st.items()}

    def get_q(self, side):
        return self.get_state(side)["q"]

    def stop(self):
        self._running = False


# --------------------------------------------------------------------- Dex1
kTopicGripperLeftCommand = "rt/dex1/left/cmd"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightCommand = "rt/dex1/right/cmd"
kTopicGripperRightState = "rt/dex1/right/state"


class Dex1Controller:
    """Threaded dual-Dex1 (1-DoF gripper) controller. Kept alongside Dex3 to
    preserve the end-effector swap seam (factory: hand=dex1)."""

    def __init__(self, fps=200.0, kp=5.0, kd=0.05, simulation_mode=False):
        self.fps = fps
        self.kp = kp
        self.kd = kd
        self.simulation_mode = simulation_mode
        self._running = True
        self._ctrl_lock = threading.Lock()

        self._left_pub = ChannelPublisher(kTopicGripperLeftCommand, MotorCmds_)
        self._left_pub.Init()
        self._right_pub = ChannelPublisher(kTopicGripperRightCommand, MotorCmds_)
        self._right_pub.Init()
        self._left_sub = ChannelSubscriber(kTopicGripperLeftState, MotorStates_)
        self._left_sub.Init()
        self._right_sub = ChannelSubscriber(kTopicGripperRightState, MotorStates_)
        self._right_sub.Init()

        self._left_state = _Buffer()
        self._right_state = _Buffer()
        self._left_msg = self._init_cmd_msg()
        self._right_msg = self._init_cmd_msg()

        self._sub_thread = threading.Thread(target=self._subscribe, daemon=True)
        self._sub_thread.start()
        self._wait_for_state()
        self._left_target = self.get_q(LEFT)
        self._right_target = self.get_q(RIGHT)
        self._pub_thread = threading.Thread(target=self._publish, daemon=True)
        self._pub_thread.start()
        logger_mp.info("Dex1Controller initialised.")

    def _init_cmd_msg(self):
        msg = MotorCmds_()
        msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        msg.cmds[0].dq = 0.0
        msg.cmds[0].tau = 0.0
        msg.cmds[0].kp = self.kp
        msg.cmds[0].kd = self.kd
        return msg

    def _wait_for_state(self, timeout=5.0):
        t0 = time.time()
        while self._left_state.get() is None or self._right_state.get() is None:
            if time.time() - t0 > timeout:
                raise TimeoutError("Dex1Controller: no /state within timeout")
            time.sleep(0.01)

    def _read_one(self, msg):
        s = msg.states[0]
        return {"q": np.array([s.q]), "dq": np.array([s.dq]),
                "tau": np.array([s.tau_est]), "press": np.array([0.0])}

    def _subscribe(self):
        while self._running:
            lm = self._left_sub.Read()
            rm = self._right_sub.Read()
            if lm is not None:
                self._left_state.set(self._read_one(lm))
            if rm is not None:
                self._right_state.set(self._read_one(rm))
            time.sleep(0.002)

    def _publish(self):
        dt = 1.0 / self.fps
        while self._running:
            t0 = time.time()
            with self._ctrl_lock:
                lt, rt = float(self._left_target[0]), float(self._right_target[0])
            self._left_msg.cmds[0].q = lt
            self._right_msg.cmds[0].q = rt
            self._left_pub.Write(self._left_msg)
            self._right_pub.Write(self._right_msg)
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def command(self, side, value):
        v = np.asarray(value, dtype=float).reshape(1)
        with self._ctrl_lock:
            if side == LEFT:
                self._left_target = v
            elif side == RIGHT:
                self._right_target = v
            else:
                raise ValueError(side)

    def command_dual(self, left, right):
        self.command(LEFT, left)
        self.command(RIGHT, right)

    def get_state(self, side):
        st = (self._left_state if side == LEFT else self._right_state).get()
        return {k: v.copy() for k, v in st.items()}

    def get_q(self, side):
        return self.get_state(side)["q"]

    def stop(self):
        self._running = False
