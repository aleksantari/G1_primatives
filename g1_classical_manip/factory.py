"""make_robot() — assemble the lean cuRobo-native robot: cuRobo planner + DDS
arm/hand controllers + executor + head-camera perception. No pinocchio (kinematics
live in cuRobo).

  * connect_dds=False    -> planner only (build/inspect plans, no controllers).
  * connect_dds=True     -> + ChannelFactoryInitialize, arm controller, hand, executor.
  * connect_camera=True  -> open the head-camera ZMQ stream (for the apriltag detector).

The frame-math owner (transforms.Frames) + the configured detector are always built
(cheap, no I/O) so robot.detector is available; the ground_truth detector needs no
camera, so detect() works planner-only/offline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import yaml

from g1_classical_manip.motion.curobo_planner import CuroboArmPlanner
from g1_classical_manip.motion.executor import Executor

_CONFIG_FILES = {"robot": "robot.yaml", "planner": "planner.yaml", "hands": "hands.yaml",
                 "camera": "camera.yaml", "perception": "perception.yaml"}

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")


def load_configs(config_dir: str = DEFAULT_CONFIG_DIR) -> Dict[str, Any]:
    cfg = {}
    for key, fname in _CONFIG_FILES.items():
        with open(os.path.join(config_dir, fname)) as f:
            cfg[key] = yaml.safe_load(f)
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

    sim_base = (cfg["robot"].get("sim", {}) or {}).get("base_world_pose")
    frames = Frames(planner, camera_cfg=cfg["camera"], sim_base_world_pose=sim_base)
    perc = cfg["perception"] or {}
    kind = perc.get("detector", "apriltag")
    if kind == "apriltag":
        detector = AprilTagDetector(frames, perc, cfg["camera"])
    elif kind == "ground_truth":
        block = (perc.get("ground_truth", {}) or {}).get("block", {})
        detector = GroundTruthDetector.from_config(frames, block)
    else:
        raise ValueError(f"unknown detector: {kind}")
    return frames, detector


def _build_camera(cfg: Dict[str, Any]):
    from g1_classical_manip.image_server.image_client import HeadCamera
    st = (cfg["camera"] or {}).get("stream", {})
    backend = st.get("backend", "unitree_lerobot")
    extra = {k: v for k, v in st.items() if k not in ("backend", "host")}
    return HeadCamera(host=st.get("host", "127.0.0.1"), backend=backend, **extra)


def make_robot(config_dir: str = DEFAULT_CONFIG_DIR, connect_dds: bool = False,
               dds_domain: int = None, dds_interface: str = None, mode: str = None,
               connect_hand: bool = True, hand: str = None,
               connect_camera: bool = False) -> Robot:
    cfg = load_configs(config_dir)
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
    executor = Executor(arm, planner=planner,
                        control_hz=ex.get("control_hz", 250.0),
                        tracking_error_abort_rad=ex.get("tracking_error_abort_rad", 0.20),
                        gravity_comp=ex.get("gravity_comp", False),
                        gravity_scale=ex.get("gravity_scale", 1.0))

    robot.arm = arm
    robot.hand = hand_obj
    robot.executor = executor
    robot.connected = True
    return robot
