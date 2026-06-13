"""make_robot()-style registry: assemble arm / hand / planner / perception from
configs (mirrors unitree_lerobot/eval_robot/make_robot.py's ARM_CONFIG/EE_CONFIG
pattern -- mirrored, not imported).

Two modes:
  * connect_dds=False  -> offline planning robot: ik, frames, planner, perception
    detector, retimer/executor configs. No DDS, no controllers. Used by tests and
    scripts/offline_plan_check.py.
  * connect_dds=True   -> full hardware robot: ChannelFactoryInitialize + arm
    controller + hand controller + executor, on top of the offline components.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np
import yaml

from g1_classical_manip.robot_control.robot_arm_ik import G1_29_ArmIK
from g1_classical_manip.perception.transforms import Frames
from g1_classical_manip.perception.apriltag_block import AprilTagBlockDetector
from g1_classical_manip.motion.cartesian_planner import CartesianPlanner
from g1_classical_manip.motion.curobo_planner import CuRoboPlanner
from g1_classical_manip.motion.executor import Executor

ARM_IK = {"G1_29": G1_29_ArmIK}
PLANNERS = {"cartesian": CartesianPlanner, "curobo": CuRoboPlanner}

_CONFIG_FILES = {
    "robot": "robot.yaml", "camera": "camera.yaml", "perception": "perception.yaml",
    "planner": "planner.yaml", "hands": "hands.yaml",
    "pick_place": "task_pick_place.yaml", "handover": "task_handover.yaml",
}

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")


def load_configs(config_dir: str = DEFAULT_CONFIG_DIR) -> Dict[str, Any]:
    cfg = {}
    for key, fname in _CONFIG_FILES.items():
        path = os.path.join(config_dir, fname)
        with open(path) as f:
            cfg[key] = yaml.safe_load(f)
    return cfg


@dataclass
class Robot:
    cfg: Dict[str, Any]
    ik: G1_29_ArmIK
    frames: Frames
    planner: Any
    perception: AprilTagBlockDetector
    arm: Any = None
    hand: Any = None
    executor: Optional[Executor] = None
    connected: bool = False

    @property
    def planner_cfg(self):
        return self.cfg["planner"]


def _locked_reference_full_nq(ik_urdf_reference_deg):
    # legs(12)+waist(3) provided in deg; the reduced builder expects a FULL-model
    # reference. We pass only the locked-joint values via the named lock list, so
    # a full zero vector with these set is unnecessary -- load_g1_reduced freezes
    # locked joints at the *reference_configuration* it is given. For simplicity
    # the locked reference is applied as zeros unless a non-trivial posture is set.
    arr = np.asarray(ik_urdf_reference_deg, float)
    if np.allclose(arr, 0.0):
        return None
    return np.deg2rad(arr)


def make_robot(config_dir: str = DEFAULT_CONFIG_DIR, connect_dds: bool = False,
               build_perception: bool = True) -> Robot:
    cfg = load_configs(config_dir)
    robot_cfg = cfg["robot"]

    # --- kinematics / IK (offline-capable) ---
    urdf = os.path.join(_REPO_ROOT, robot_cfg["model"]["urdf"])
    locked = _locked_reference_full_nq(robot_cfg["model"].get("locked_reference_deg", []))
    ik = G1_29_ArmIK(urdf_path=urdf, locked_reference=locked)
    frames = Frames(reduced_robot=ik.reduced_robot, camera_cfg=cfg["camera"])

    planner_name = cfg["planner"].get("planner", "cartesian")
    planner = PLANNERS[planner_name].from_config(ik, cfg["planner"]) \
        if hasattr(PLANNERS[planner_name], "from_config") \
        else PLANNERS[planner_name](ik, cfg["planner"])

    perception = None
    if build_perception:
        perception = AprilTagBlockDetector(frames, cfg["perception"], cfg["camera"])

    robot = Robot(cfg=cfg, ik=ik, frames=frames, planner=planner,
                  perception=perception)

    if not connect_dds:
        return robot

    # --- DDS-connected controllers (hardware) ---
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
    arm = G1_29_ArmController(motion_mode=motion_mode, simulation_mode=sim)

    hand_key = robot_cfg.get("hand", "dex3")
    if hand_key == "dex3":
        hctrl = Dex3Controller(simulation_mode=sim)
        hand = Dex3Hand(hctrl, cfg["hands"]["dex3"])
    elif hand_key == "dex1":
        hctrl = Dex1Controller(simulation_mode=sim)
        hand = Dex1Hand(hctrl, cfg["hands"]["dex1"])
    else:
        raise ValueError(f"unknown hand: {hand_key}")

    ex = cfg["planner"].get("executor", {})
    executor = Executor(arm, ik=ik,
                        control_hz=cfg["planner"]["retimer"].get("control_hz", 250.0),
                        tracking_error_abort_rad=ex.get("tracking_error_abort_rad", 0.20))

    robot.arm = arm
    robot.hand = hand
    robot.executor = executor
    robot.connected = True
    return robot
