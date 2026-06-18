"""make_robot() — assemble the lean cuRobo-native robot: cuRobo planner + DDS
arm/hand controllers + executor + head-camera perception. No pinocchio (kinematics
live in cuRobo).

  * connect_dds=False    -> planner only (build/inspect plans, no controllers).
  * connect_dds=True     -> + ChannelFactoryInitialize, arm controller, hand, executor.
                            Debug mode (low-level rt/lowcmd control) is assumed to be
                            set by the OPERATOR via the physical remote (the proven
                            unitree_lerobot flow does the same, never calling
                            MotionSwitcher). Opt in to an SDK release with
                            enter_debug_mode=True; never in sim.
  * connect_camera=True  -> open the head-camera ZMQ stream (for the apriltag detector).

The frame-math owner (transforms.Frames) + the configured detector are always built
(cheap, no I/O) so robot.detector is available; the ground_truth detector needs no
camera, so detect() works planner-only/offline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import yaml

from g1_classical_manip.motion.curobo_planner import CuroboArmPlanner
from g1_classical_manip.motion.executor import Executor

_CONFIG_FILES = {"robot": "robot.yaml", "planner": "planner.yaml", "hands": "hands.yaml",
                 "perception": "perception.yaml"}
# Camera config is chosen separately (sim mono vs real ZED stereo); the --target
# ladder passes camera_config=camera_real.yaml.
DEFAULT_CAMERA_CONFIG = "camera_sim.yaml"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")


def load_configs(config_dir: str = DEFAULT_CONFIG_DIR,
                 camera_file: str = DEFAULT_CAMERA_CONFIG) -> Dict[str, Any]:
    cfg = {}
    for key, fname in _CONFIG_FILES.items():
        with open(os.path.join(config_dir, fname)) as f:
            cfg[key] = yaml.safe_load(f)
    with open(os.path.join(config_dir, camera_file)) as f:
        cfg["camera"] = yaml.safe_load(f)   # camera_sim.yaml | camera_real.yaml (ZED)
    return cfg


@dataclass
class Robot:
    cfg: Dict[str, Any]
    planner: CuroboArmPlanner
    arm: Any = None
    hand: Any = None
    executor: Optional[Executor] = None
    frames: Any = None          # transforms.Frames (frame-math owner)
    detector: Any = None        # perception Detector (apriltag | ground_truth)
    camera: Any = None          # image_server.HeadCamera (None unless connect_camera)
    connected: bool = False

    @property
    def planner_cfg(self):
        return self.cfg["planner"]


def _build_perception(planner, cfg: Dict[str, Any]):
    """Build the frame-math owner + the configured detector (no I/O). The seam for
    new detectors is the `detector:` selector in perception.yaml."""
    from g1_classical_manip.perception.transforms import Frames
    from g1_classical_manip.perception.apriltag_block import AprilTagDetector
    from g1_classical_manip.perception.ground_truth import GroundTruthDetector
    from g1_classical_manip.perception.sim_state import SimStateDetector

    sim_base = (cfg["robot"].get("sim", {}) or {}).get("base_world_pose")
    frames = Frames(planner, camera_cfg=cfg["camera"], sim_base_world_pose=sim_base)
    perc = cfg["perception"] or {}
    kind = perc.get("detector", "apriltag")
    if kind == "apriltag":
        detector = AprilTagDetector(frames, perc, cfg["camera"])
    elif kind == "ground_truth":
        block = (perc.get("ground_truth", {}) or {}).get("block", {})
        detector = GroundTruthDetector.from_config(frames, block)
    elif kind == "sim_state":
        # live sim ground-truth via rt/sim_state; needs DDS (connect_dds=True)
        detector = SimStateDetector.from_config(frames, perc.get("sim_state", {}))
    else:
        raise ValueError(f"unknown detector: {kind}")
    return frames, detector


def _build_camera(cfg: Dict[str, Any]):
    from g1_classical_manip.image_server.image_client import HeadCamera
    st = (cfg["camera"] or {}).get("stream", {})
    backend = st.get("backend", "zmq")
    # forward the rest of the stream block (port / request_port / stereo / stereo_side
    # / recv_timeout_ms) straight to HeadCamera as kwargs.
    extra = {k: v for k, v in st.items() if k not in ("backend", "host")}
    return HeadCamera(host=st.get("host", "127.0.0.1"), backend=backend, **extra)


def make_robot(config_dir: str = DEFAULT_CONFIG_DIR, connect_dds: bool = False,
               dds_domain: int = None, dds_interface: str = None, mode: str = None,
               connect_hand: bool = True, hand: str = None,
               connect_camera: bool = False, enter_debug_mode: bool = None,
               home_on_connect: bool = True, gravity_comp: bool = None,
               gravity_scale: float = None,
               camera_config: str = DEFAULT_CAMERA_CONFIG) -> Robot:
    cfg = load_configs(config_dir, camera_file=camera_config)
    robot_cfg = cfg["robot"]
    # optional overrides (unitree_sim_isaaclab loopback: domain 1, iface "lo", mode "sim")
    if dds_domain is not None:
        robot_cfg["dds"]["domain_id"] = dds_domain
    if dds_interface is not None:
        robot_cfg["dds"]["interface"] = dds_interface
    if mode is not None:
        robot_cfg["mode"] = mode
    if hand is not None:
        robot_cfg["hand"] = hand

    hand_key = robot_cfg.get("hand", "dex3")
    curobo_cfg = os.path.join(config_dir, "curobo", f"g1_{hand_key}_curobo.yml")
    planner = CuroboArmPlanner(curobo_cfg, planner_cfg=cfg["planner"])
    robot = Robot(cfg=cfg, planner=planner)

    # perception (frame math + detector) is always available; camera stream optional.
    robot.frames, robot.detector = _build_perception(planner, cfg)
    if connect_camera:
        robot.camera = _build_camera(cfg)

    if not connect_dds:
        return robot

    # --- DDS-connected controllers ---
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from g1_classical_manip.robot_control.robot_arm import G1_29_ArmController
    from g1_classical_manip.robot_control.robot_hand_unitree import (
        Dex3Controller, Dex1Controller)
    from g1_classical_manip.ee.dex3 import Dex3Hand
    from g1_classical_manip.ee.dex1 import Dex1Hand

    dds = robot_cfg["dds"]
    iface = dds.get("interface") or None
    if iface:
        ChannelFactoryInitialize(dds["domain_id"], iface)
    else:
        ChannelFactoryInitialize(dds["domain_id"])

    sim = robot_cfg.get("mode") == "sim"
    motion_mode = robot_cfg.get("mode") == "motion"

    # Low-level arm control (rt/lowcmd) needs the robot in DEBUG mode (high-level
    # loco/AI service released). On this rig the OPERATOR sets debug mode via the
    # physical remote before launch -- the proven unitree_lerobot flow does the same and
    # NEVER calls MotionSwitcher. So we do NOT auto-enter by default: calling
    # MotionSwitcher.ReleaseMode() on a robot already in physically-set debug mode can
    # drop it back OUT of low-level-control mode, after which rt/lowcmd is published but
    # ignored and the arms won't move (observed: launch home left residual ~87 deg = no
    # motion). Opt in with enter_debug_mode=True (e.g. no remote available); never in
    # sim. MotionSwitcher needs DDS already initialised (above).
    do_enter = bool(enter_debug_mode) and not sim
    if do_enter:
        from g1_classical_manip.robot_control.motion_switcher import MotionSwitcher
        status, active = MotionSwitcher().Enter_Debug_Mode()
        print(f"[make_robot] MotionSwitcher.Enter_Debug_Mode -> status={status}, "
              f"remaining active mode={active}")
    elif not sim:
        print("[make_robot] not calling MotionSwitcher -- assuming the robot is already "
              "in DEBUG mode (set via the physical remote). Pass enter_debug_mode=True to "
              "release high-level modes over the SDK instead.")

    arm = G1_29_ArmController(motion_mode=motion_mode, simulation_mode=sim,
                              velocity_limit=robot_cfg.get("arm_velocity_limit", 20.0))

    hand_obj = None
    if connect_hand:
        if hand_key == "dex3":
            hand_obj = Dex3Hand(Dex3Controller(simulation_mode=sim), cfg["hands"]["dex3"])
        elif hand_key == "dex1":
            hand_obj = Dex1Hand(Dex1Controller(simulation_mode=sim), cfg["hands"]["dex1"])
        else:
            raise ValueError(f"unknown hand: {hand_key}")

    ex = cfg["planner"].get("executor", {})
    # gravity_comp/gravity_scale: planner.yaml default (ON for real), overridable
    # per-call (the hardware bring-up ramps gravity_scale from the CLI to confirm the
    # sign before committing it). A non-None override wins. In sim the tau feed-forward
    # is neither needed nor validated, so force it OFF unless explicitly requested.
    gc = ex.get("gravity_comp", False) if gravity_comp is None else gravity_comp
    gs = ex.get("gravity_scale", 1.0) if gravity_scale is None else gravity_scale
    if sim and gravity_comp is None:
        gc = False
    # Trajectory playback speed: planner.yaml default (< 1.0 slows real moves so the
    # torque-throttled arm can track). Sim bypasses the velocity clip, so it tracks at
    # full speed -- force 1.0 there. Per-run override: 04_move --speed (post-connect).
    td = 1.0 if sim else float(ex.get("time_dilation", 1.0))
    executor = Executor(arm, planner=planner,
                        control_hz=ex.get("control_hz", 250.0),
                        tracking_error_abort_rad=ex.get("tracking_error_abort_rad", 0.20),
                        gravity_comp=gc, gravity_scale=gs, time_dilation=td)
    if gc:
        print(f"[make_robot] gravity-comp ON, scale={gs} (cuRobo RNEA feed-forward; "
              f"watch that the elbow residual DROPS -- if it GROWS the sign is flipped, "
              f"retry with a negative scale)")

    robot.arm = arm
    robot.hand = hand_obj
    robot.executor = executor
    robot.connected = True

    # LAUNCH HOME (default): drive the arms to the configured home with DIRECT
    # position control before any action primitive runs. The arm controller already
    # eases toward home on construction (q_target starts at zeros); this waits for it
    # to CONVERGE. Crucially it is UN-PLANNED -- it does NOT use cuRobo's collision-
    # aware planner, so it can move out of a pose cuRobo flags as a self-collision
    # START (e.g. the arms folded), where the planned home() primitive would refuse
    # to plan. NOT collision-avoided en route, so the path to home must be clear.
    # Disable with home_on_connect=False. After this, home()/move() plan from a
    # known-good, collision-free home.
    if home_on_connect:
        q_home = np.deg2rad(robot_cfg["home_q14_deg"])
        print("[make_robot] homing arms to launch pose (direct PD, un-planned, "
              "velocity-capped) -- ensure the path is clear ...")
        err = executor.go_home_direct(q_home)
        resid = np.rad2deg(q_home - arm.get_current_dual_arm_q())  # per-joint residual (deg)
        np.set_printoptions(precision=1, suppress=True, sign=" ")
        print(f"[make_robot] launch home: max residual {np.rad2deg(err):.2f} deg")
        print(f"[make_robot]   per-joint resid deg  L[sp sr sy el wr wp wy]: {resid[:7]}")
        print(f"[make_robot]                        R[sp sr sy el wr wp wy]: {resid[7:]}")
        if np.rad2deg(err) > 5.0:
            print("[make_robot]   WARNING: arms did not reach home. Large residual on the "
                  "shoulder/elbow joints => gravity-limited (PD torque is capped by the "
                  "velocity clip: max torque ~= kp * arm_velocity_limit * control_dt). Fix "
                  "with gravity_comp (planner.yaml) and/or a higher arm_velocity_limit; large "
                  "residual on arbitrary joints instead => something is fighting rt/lowcmd.")
    return robot
