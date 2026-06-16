"""make_robot() — assemble the lean cuRobo-native robot: cuRobo planner + DDS
arm/hand controllers + executor. No pinocchio (kinematics live in cuRobo).

Two modes:
  * connect_dds=False -> planner only (build/inspect plans, no controllers).
  * connect_dds=True  -> + ChannelFactoryInitialize, arm controller, hand, executor.

Perception/tasks are out of scope for the MVP (perception kept on disk, dormant).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import yaml

from g1_classical_manip.motion.curobo_planner import CuroboArmPlanner
from g1_classical_manip.motion.executor import Executor

_CONFIG_FILES = {"robot": "robot.yaml", "planner": "planner.yaml", "hands": "hands.yaml"}

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
    connected: bool = False

    @property
    def planner_cfg(self):
        return self.cfg["planner"]


def make_robot(config_dir: str = DEFAULT_CONFIG_DIR, connect_dds: bool = False,
               dds_domain: int = None, dds_interface: str = None, mode: str = None,
               connect_hand: bool = True, hand: str = None) -> Robot:
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
    arm = G1_29_ArmController(motion_mode=motion_mode, simulation_mode=sim)

    hand_obj = None
    if connect_hand:
        if hand_key == "dex3":
            hand_obj = Dex3Hand(Dex3Controller(simulation_mode=sim), cfg["hands"]["dex3"])
        elif hand_key == "dex1":
            hand_obj = Dex1Hand(Dex1Controller(simulation_mode=sim), cfg["hands"]["dex1"])
        else:
            raise ValueError(f"unknown hand: {hand_key}")

    ex = cfg["planner"].get("executor", {})
    executor = Executor(arm, ik=None, gravity_comp=False,
                        control_hz=ex.get("control_hz", 250.0),
                        tracking_error_abort_rad=ex.get("tracking_error_abort_rad", 0.20))

    robot.arm = arm
    robot.hand = hand_obj
    robot.executor = executor
    robot.connected = True
    return robot
