"""The Robot facade -- the ONE object an agent host constructs and holds.

    import g1_primitives as g1
    robot = g1.connect("sim")            # or "real"
    robot.home()
    result = robot.grasp("right", "block")

``connect(target)`` folds in everything the old make_robot + scripts/_rig pair did:
target-keyed DDS settings (sim = loopback domain 1/"lo", real = robot.yaml), the
matching camera config (Isaac stream vs real ZED), debug-mode entry on real
(check-first ``hardware.ensure_debug_mode``; the operator's remote-set debug mode is
never disturbed), and the launch home (DIRECT, un-planned PD -- see ``connect``).
``offline()`` builds the planner + perception stack with no DDS/robot at all
(plan inspection, offline tools).

Reconfiguration goes through ``set_*`` methods -- never mutate ``robot.cfg`` and
rebuild components by hand. Action verbs delegate to ``api.primitives``; results
come back as ``Result`` / ``GraspResult`` (api.results).
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from g1_primitives.config import load_configs, DEFAULT_CONFIG_DIR
from g1_primitives.api import _builders
from g1_primitives.api import primitives as P
from g1_primitives.api.options import GraspOptions
from g1_primitives.api.results import Result, GraspResult

# unitree_sim_isaaclab loopback rig (SIM_NOTES.md); real DDS comes from robot.yaml.
SIM_DDS = (1, "lo")
_CAMERA_FILE = {"sim": "camera_sim.yaml", "real": "camera_real.yaml"}


def _warn_sim_uri():
    if not os.environ.get("CYCLONEDDS_URI"):
        print("WARNING: CYCLONEDDS_URI is unset -- sim loopback DDS discovery will likely "
              "fail. Prefix the command with\n  CYCLONEDDS_URI=file://$PWD/configs/"
              "cyclonedds_loopback.xml")


class Robot:
    """Facade over planner + controllers + executor + perception + grasp source.

    Construct via ``Robot.connect`` / ``Robot.offline`` (module-level ``connect`` is
    an alias). Attributes are readable (``robot.planner``, ``robot.camera`` ...) but
    reconfiguration must go through the ``set_*`` methods so the config dict and the
    rebuilt components never drift apart.
    """

    def __init__(self, cfg: Dict[str, Any], planner):
        self.cfg = cfg
        self.planner = planner
        self.arm = None             # hardware.robot_arm.G1_29_ArmController (connected only)
        self.hand = None            # ee.dex3.Dex3Hand | ee.dex1.Dex1Hand
        self.executor = None        # motion.executor.Executor
        self.frames = None          # perception.frames.Frames (frame-math owner)
        self.detector = None        # perception detector (sim_state)
        self.grasp_source = None    # grasp.GraspSource (graspgenx | sim_cloud)
        self.camera = None          # hardware.camera_client.HeadCamera
        self.connected = False
        self.target: Optional[str] = None   # "sim" | "real"
        self._config_dir = DEFAULT_CONFIG_DIR

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def connect(cls, target: str = "sim", *, config_dir: str = DEFAULT_CONFIG_DIR,
                hand: Optional[str] = None, camera: bool = True,
                home_on_connect: bool = True, enter_debug_mode: Optional[bool] = None,
                gravity_comp: Optional[bool] = None, gravity_scale: Optional[float] = None,
                speed: Optional[float] = None) -> "Robot":
        """Build + connect the full robot for ``target`` ("sim" | "real").

        sim  -> loopback DDS (domain 1, iface "lo"), mode=sim, camera_sim.yaml.
        real -> robot.yaml DDS, mode=debug, camera_real.yaml, and debug-mode entry:
                ``enter_debug_mode`` None (default) or True runs the check-first
                ``ensure_debug_mode`` (an operator-set debug mode is detected and left
                untouched; an active ai/loco mode is released); False skips the check
                entirely. Never runs in sim.

        NOTE: connecting HOMES THE ARMS with DIRECT (un-planned, velocity-capped) PD
        and waits for convergence, so the planners start from a known collision-free
        pose -- the path to home must be clear (``home_on_connect=False`` to skip).
        ``speed`` overrides executor playback speed (time_dilation) post-build.
        """
        if target not in _CAMERA_FILE:
            raise ValueError(f"unknown target: {target!r} (expected 'sim' or 'real')")
        cfg = load_configs(config_dir, camera_file=_CAMERA_FILE[target])
        if hand is not None:
            cfg["robot"]["hand"] = hand
        if target == "sim":
            _warn_sim_uri()
            cfg["robot"]["dds"]["domain_id"], cfg["robot"]["dds"]["interface"] = SIM_DDS
            cfg["robot"]["mode"] = "sim"
        else:
            cfg["robot"]["mode"] = "debug"
        robot = cls._assemble(cfg, config_dir, target, connect_camera=camera)
        robot._connect_dds(enter_debug_mode=enter_debug_mode,
                           home_on_connect=home_on_connect,
                           gravity_comp=gravity_comp, gravity_scale=gravity_scale)
        if speed is not None:
            robot.executor.time_dilation = float(speed)
        return robot

    @classmethod
    def offline(cls, target: str = "sim", *, config_dir: str = DEFAULT_CONFIG_DIR,
                hand: Optional[str] = None, camera: bool = False) -> "Robot":
        """Planner + perception + grasp source with NO DDS/controllers (plan
        inspection, offline tools). ``camera=True`` still opens the head-camera
        stream (offline frame capture)."""
        if target not in _CAMERA_FILE:
            raise ValueError(f"unknown target: {target!r} (expected 'sim' or 'real')")
        cfg = load_configs(config_dir, camera_file=_CAMERA_FILE[target])
        if hand is not None:
            cfg["robot"]["hand"] = hand
        return cls._assemble(cfg, config_dir, target, connect_camera=camera)

    @classmethod
    def _assemble(cls, cfg: Dict[str, Any], config_dir: str, target: str,
                  connect_camera: bool) -> "Robot":
        """Shared DDS-free assembly: cuRobo planner + frame math + detector + grasp
        source (+ optional camera stream)."""
        hand_key = cfg["robot"].get("hand", "dex3")
        curobo_cfg = os.path.join(config_dir, "curobo", f"g1_{hand_key}_curobo.yml")
        from g1_primitives.motion.planner import CuroboArmPlanner  # CUDA on instantiation
        planner = CuroboArmPlanner(curobo_cfg, planner_cfg=cfg["planner"])
        robot = cls(cfg, planner)
        robot.target = target
        robot._config_dir = config_dir
        robot.frames, robot.detector = _builders.build_perception(planner, cfg)
        robot.grasp_source = _builders.build_grasp_source(robot.frames, cfg)
        if connect_camera:
            robot.camera = _builders.build_camera(cfg)
        return robot

    def _connect_dds(self, *, connect_hand: bool = True,
                     enter_debug_mode: Optional[bool] = None,
                     home_on_connect: bool = True,
                     gravity_comp: Optional[bool] = None,
                     gravity_scale: Optional[float] = None) -> None:
        """DDS-connected half: ChannelFactory, debug-mode entry (real), arm/hand
        controllers, executor, launch home."""
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from g1_primitives.hardware.robot_arm import G1_29_ArmController
        from g1_primitives.hardware.robot_hand_unitree import (
            Dex3Controller, Dex1Controller)
        from g1_primitives.ee.dex3 import Dex3Hand
        from g1_primitives.ee.dex1 import Dex1Hand
        from g1_primitives.motion.executor import Executor

        robot_cfg = self.cfg["robot"]
        dds = robot_cfg["dds"]
        iface = dds.get("interface") or None
        if iface:
            ChannelFactoryInitialize(dds["domain_id"], iface)
        else:
            ChannelFactoryInitialize(dds["domain_id"])

        sim = robot_cfg.get("mode") == "sim"
        motion_mode = robot_cfg.get("mode") == "motion"

        # Low-level arm control (rt/lowcmd) needs the robot in DEBUG mode (high-level
        # loco/AI service released). ensure_debug_mode is CHECK-FIRST: an operator-set
        # debug mode (the usual flow on this rig -- physical remote before launch) is
        # detected and left untouched; only an active named mode is released. Blindly
        # calling ReleaseMode on an already-debug robot drops it OUT of low-level
        # control (observed: launch home residual ~87 deg = zero motion) -- hence the
        # check-first contract. enter_debug_mode=False skips even the check.
        if not sim:
            if enter_debug_mode is None or enter_debug_mode:
                from g1_primitives.hardware.motion_switcher import ensure_debug_mode
                ok, msg = ensure_debug_mode()
                print(f"[connect] debug mode: {msg}")
                if not ok:
                    print("[connect]   WARNING: could not confirm debug mode -- rt/lowcmd "
                          "may be ignored (arms won't move). Set debug mode via the "
                          "physical remote, or pass enter_debug_mode=False to skip this "
                          "check if the RPC service is unavailable.")
            else:
                print("[connect] debug-mode check skipped (enter_debug_mode=False) -- "
                      "assuming the OPERATOR set debug mode via the physical remote.")

        arm = G1_29_ArmController(motion_mode=motion_mode, simulation_mode=sim,
                                  velocity_limit=robot_cfg.get("arm_velocity_limit", 20.0))

        hand_obj = None
        if connect_hand:
            hand_key = robot_cfg.get("hand", "dex3")
            if hand_key == "dex3":
                hand_obj = Dex3Hand(Dex3Controller(simulation_mode=sim),
                                    self.cfg["hands"]["dex3"])
            elif hand_key == "dex1":
                hand_obj = Dex1Hand(Dex1Controller(simulation_mode=sim),
                                    self.cfg["hands"]["dex1"])
            else:
                raise ValueError(f"unknown hand: {hand_key}")

        ex = self.cfg["planner"].get("executor", {})
        # gravity_comp/gravity_scale: planner.yaml default (ON for real), overridable
        # per-call (the hardware bring-up ramps gravity_scale from the CLI to confirm
        # the sign before committing it). A non-None override wins. In sim the tau
        # feed-forward is neither needed nor validated, so force it OFF unless
        # explicitly requested.
        gc = ex.get("gravity_comp", False) if gravity_comp is None else gravity_comp
        gs = ex.get("gravity_scale", 1.0) if gravity_scale is None else gravity_scale
        if sim and gravity_comp is None:
            gc = False
        # Trajectory playback speed: planner.yaml default (< 1.0 slows real moves so
        # the torque-throttled arm can track). Sim bypasses the velocity clip, so it
        # tracks at full speed -- force 1.0 there. Per-run override: set_executor(speed=).
        td = 1.0 if sim else float(ex.get("time_dilation", 1.0))
        executor = Executor(arm, planner=self.planner,
                            control_hz=ex.get("control_hz", 250.0),
                            tracking_error_abort_rad=ex.get("tracking_error_abort_rad", 0.20),
                            gravity_comp=gc, gravity_scale=gs, time_dilation=td)
        if gc:
            print(f"[connect] gravity-comp ON, scale={gs} (cuRobo RNEA feed-forward; "
                  f"watch that the elbow residual DROPS -- if it GROWS the sign is "
                  f"flipped, retry with a negative scale)")

        self.arm = arm
        self.hand = hand_obj
        self.executor = executor
        self.connected = True

        # LAUNCH HOME (default): drive the arms to the configured home with DIRECT
        # position control before any action primitive runs. The arm controller already
        # eases toward home on construction (q_target starts at zeros); this waits for
        # it to CONVERGE. Crucially it is UN-PLANNED -- it does NOT use cuRobo's
        # collision-aware planner, so it can move out of a pose cuRobo flags as a
        # self-collision START (e.g. the arms folded), where the planned home()
        # primitive would refuse to plan. NOT collision-avoided en route, so the path
        # to home must be clear. Disable with home_on_connect=False. After this,
        # home()/move() plan from a known-good, collision-free home.
        if home_on_connect:
            q_home = np.deg2rad(robot_cfg["home_q14_deg"])
            print("[connect] homing arms to launch pose (direct PD, un-planned, "
                  "velocity-capped) -- ensure the path is clear ...")
            err = executor.go_home_direct(q_home)
            resid = np.rad2deg(q_home - arm.get_current_dual_arm_q())  # per-joint (deg)
            np.set_printoptions(precision=1, suppress=True, sign=" ")
            print(f"[connect] launch home: max residual {np.rad2deg(err):.2f} deg")
            print(f"[connect]   per-joint resid deg  L[sp sr sy el wr wp wy]: {resid[:7]}")
            print(f"[connect]                        R[sp sr sy el wr wp wy]: {resid[7:]}")
            if np.rad2deg(err) > 5.0:
                print("[connect]   WARNING: arms did not reach home. Large residual on "
                      "the shoulder/elbow joints => gravity-limited (PD torque is capped "
                      "by the velocity clip: max torque ~= kp * arm_velocity_limit * "
                      "control_dt). Fix with gravity_comp (planner.yaml) and/or a higher "
                      "arm_velocity_limit; large residual on arbitrary joints instead => "
                      "something is fighting rt/lowcmd.")

    def close(self) -> None:
        """Best-effort release of external resources (camera stream, viz server).
        Controllers keep publishing their hold pose -- there is no motor 'off' here;
        that stays with the operator/e-stop."""
        for obj in (self.camera, getattr(self.grasp_source, "viz", None)):
            if obj is not None and hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:   # noqa: BLE001 - shutdown must never raise
                    pass

    # --------------------------------------------------------- introspection
    @property
    def planner_cfg(self) -> Dict[str, Any]:
        return self.cfg["planner"]

    @property
    def grasp_source_kind(self) -> str:
        """The configured grasp-source selector ('graspgenx' | 'sim_cloud')."""
        return (self.cfg.get("grasp") or {}).get("grasp_source", "graspgenx")

    # -------------------------------------------------------- reconfiguration
    def set_grasp_source(self, name: str, **overrides) -> None:
        """Switch the grasp source ('graspgenx' | 'sim_cloud') and/or override its
        config block, then rebuild it. Overrides land in the source's own block
        (falling back to 'graspgenx' for shared keys like the tool transform)."""
        g = self.cfg.setdefault("grasp", {})
        g["grasp_source"] = name
        if overrides:
            block = name if name in g else "graspgenx"
            blk = g.get(block) or {}
            blk.update(overrides)
            g[block] = blk
        self.grasp_source = _builders.build_grasp_source(self.frames, self.cfg)

    def set_segmenter(self, mode: Optional[str], **overrides) -> None:
        """Set the SAM3 segmentation mode ('none' | 'auto' | 'interactive') + block
        overrides (prompt, host/port ...), then rebuild the grasp source that owns it."""
        g = self.cfg.setdefault("grasp", {})
        seg = g.get("segment") or {}
        seg["mode"] = mode
        seg.update(overrides)
        g["segment"] = seg
        self.grasp_source = _builders.build_grasp_source(self.frames, self.cfg)

    def set_visualize(self, enabled: bool = True, **overrides) -> None:
        """Toggle the viser grasp visualization (graspgenx `visualize` block) and
        rebuild the grasp source (the viz client lives inside it)."""
        g = self.cfg.setdefault("grasp", {})
        gx = g.get("graspgenx") or {}
        v = gx.get("visualize") or {}
        v["enabled"] = bool(enabled)
        v.update(overrides)
        gx["visualize"] = v
        g["graspgenx"] = gx
        self.grasp_source = _builders.build_grasp_source(self.frames, self.cfg)

    def set_executor(self, speed: Optional[float] = None,
                     abort_thresh_rad: Optional[float] = None,
                     gravity_comp: Optional[bool] = None,
                     gravity_scale: Optional[float] = None) -> None:
        """Tune trajectory execution: ``speed`` = playback time_dilation (<1 slows the
        same plan), abort threshold, gravity-comp feed-forward. Only given args change."""
        ex = self.executor
        if ex is None:
            raise RuntimeError("no executor -- this robot was built offline()")
        if speed is not None:
            ex.time_dilation = float(speed)
        if abort_thresh_rad is not None:
            ex.abort_thresh = float(abort_thresh_rad)
        if gravity_comp is not None:
            # same guard as Executor.__init__: RNEA needs the planner
            ex.gravity_comp = bool(gravity_comp) and ex.planner is not None
        if gravity_scale is not None:
            ex.gravity_scale = float(gravity_scale)

    def set_collision_world(self, enabled: bool = True, **overrides) -> None:
        """Enable/disable the depth-ESDF collision world for grasp approach planning
        (planner.yaml `grasp.collision_world`), with block overrides (voxel_size,
        exclude_object, robot_mask_margin ...). Applies to the planner immediately."""
        gp = self.cfg["planner"].setdefault("grasp", {})
        if gp is None:      # yaml empty block
            gp = self.cfg["planner"]["grasp"] = {}
        cw = gp.get("collision_world") or {}
        cw.update(overrides)
        cw["enabled"] = bool(enabled)
        gp["collision_world"] = cw
        self.planner.set_collision_world(bool(enabled), cw)

    def set_grasp_strategies(self, strategies: List[Dict[str, Any]]) -> None:
        """Replace the plan_grasp sweep strategies (planner.yaml `grasp.strategies`):
        an ordered list of {approach_offset, lift_offset, ...} dicts tried in order."""
        gp = self.cfg["planner"].setdefault("grasp", {})
        if gp is None:
            gp = self.cfg["planner"]["grasp"] = {}
        gp["strategies"] = [dict(s) for s in strategies]

    # -------------------------------------------------------------- perception
    def wait_for_frames(self, rgb: bool = True, depth: bool = False,
                        timeout_s: float = 6.0) -> bool:
        """Warm up the head-camera stream: poll until the requested frame kinds have
        arrived (the CONFLATE sockets deliver nothing until the publisher's first
        frame lands). THE one cold-stream warm-up -- call before the first perception
        read after connect. Returns False on timeout or no camera."""
        cam = self.camera
        if cam is None:
            return False
        got_rgb, got_depth = not rgb, not depth
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if not got_rgb and cam.get_rgb_frame() is not None:
                got_rgb = True
            # has_depth can flip true once the stream's first depth frame lands,
            # so keep polling it rather than early-exiting.
            if not got_depth and cam.has_depth and cam.get_depth_frame() is not None:
                got_depth = True
            if got_rgb and got_depth:
                return True
            time.sleep(0.05)
        return False

    def rgb(self):
        """Latest head-camera RGB frame (H,W,3 uint8) or None."""
        return self.camera.get_rgb_frame() if self.camera is not None else None

    def depth(self):
        """Latest head-camera depth frame (H,W float32 mm) or None."""
        if self.camera is None or not self.camera.has_depth:
            return None
        return self.camera.get_depth_frame()

    def detect(self, target: str = "block", frames: int = 5):
        """Detect ``target`` -> Detection (``.pose`` in the pelvis frame) or None."""
        return P.detect(self, target, frames=frames)

    def grasp_candidates(self, side: str, target: str = "block"):
        """Run the configured grasp source once -> ranked GraspCandidates (empty list
        = nothing found). Warms up the camera first when the source needs it; what the
        source SAW is retained on ``grasp_source.last_snapshot``."""
        if self.camera is not None:
            self.wait_for_frames(rgb=True, depth=(self.grasp_source_kind == "graspgenx"))
        return self.grasp_source.grasps(self, side, target)

    # ------------------------------------------------------------ action verbs
    def home(self) -> Result:
        """Planned, collision-aware move of both arms to the configured home pose."""
        return P.home(self)

    def move(self, side: str, goal_pose) -> Result:
        """Move ``side`` wrist to ``goal_pose`` (pelvis frame); the other arm holds."""
        return P.move(self, side, goal_pose)

    def open_hand(self, side: str, verify: bool = False) -> Result:
        return P.open_hand(self, side, verify=verify)

    def close_hand(self, side: str, verify: bool = False,
                   fraction: float = 1.0) -> Result:
        return P.close_hand(self, side, verify=verify, fraction=fraction)

    def grasp(self, side: str, target: str = "block",
              options: Optional[GraspOptions] = None) -> GraspResult:
        """The one agent pick verb: grasp candidates for ``target`` -> native cuRobo
        plan_grasp (goalset + approach/grasp/lift) -> execute + close per ``options``
        (None = full autonomous defaults). Returns a GraspResult with a per-phase
        ``report``."""
        candidates = self.grasp_candidates(side, target)
        if not candidates:
            return GraspResult(
                False, f"no grasp candidates from source '{self.grasp_source_kind}' "
                       f"for target '{target}'")
        return P.grasp_motion(self, side, candidates, options)


def connect(target: str = "sim", **kwargs) -> Robot:
    """Top-level alias of :meth:`Robot.connect` -- ``g1_primitives.connect("sim")``."""
    return Robot.connect(target, **kwargs)
