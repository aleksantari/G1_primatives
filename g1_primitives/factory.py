"""DEPRECATED shim -- assembly moved to the Robot facade.

The canonical entry points are now:

    g1_primitives.connect(target)        # api.robot.Robot.connect
    g1_primitives.Robot.offline(...)     # planner-only, no DDS
    g1_primitives.config.load_configs    # the only yaml reader
    g1_primitives.api._builders          # per-component builders

This module exists ONLY so the not-yet-migrated scripts/ keep running; it is
deleted in the scripts reorg. Do not add imports of it in new code.
"""
from __future__ import annotations

from typing import Optional

from g1_primitives.config import (load_configs, DEFAULT_CONFIG_DIR,        # noqa: F401
                                  DEFAULT_CAMERA_CONFIG, _REPO_ROOT)
from g1_primitives.api.robot import Robot                                  # noqa: F401
from g1_primitives.api._builders import (                                  # noqa: F401
    build_grasp_source as _build_grasp_source,
    build_perception as _build_perception,
    build_segmenter as _build_segmenter,
    build_grasp_viz as _build_grasp_viz,
    build_camera as _build_camera)


def make_robot(config_dir: str = DEFAULT_CONFIG_DIR, connect_dds: bool = False,
               dds_domain: int = None, dds_interface: str = None, mode: str = None,
               connect_hand: bool = True, hand: str = None,
               connect_camera: bool = False, enter_debug_mode: Optional[bool] = None,
               home_on_connect: bool = True, gravity_comp: bool = None,
               gravity_scale: float = None,
               camera_config: str = DEFAULT_CAMERA_CONFIG) -> Robot:
    """DEPRECATED: use g1_primitives.connect(target) / Robot.offline() instead.
    Old-signature assembly delegating to the Robot facade internals. NOTE one
    behavior change vs the pre-facade make_robot: on a real (non-sim) DDS connect,
    enter_debug_mode=None now runs the check-first ensure_debug_mode (an
    operator-set debug mode is left untouched); pass False to skip the check."""
    cfg = load_configs(config_dir, camera_file=camera_config)
    rc = cfg["robot"]
    if dds_domain is not None:
        rc["dds"]["domain_id"] = dds_domain
    if dds_interface is not None:
        rc["dds"]["interface"] = dds_interface
    if mode is not None:
        rc["mode"] = mode
    if hand is not None:
        rc["hand"] = hand
    target = "sim" if rc.get("mode") == "sim" else "real"
    robot = Robot._assemble(cfg, config_dir, target, connect_camera=connect_camera)
    if connect_dds:
        robot._connect_dds(connect_hand=connect_hand, enter_debug_mode=enter_debug_mode,
                           home_on_connect=home_on_connect, gravity_comp=gravity_comp,
                           gravity_scale=gravity_scale)
    return robot
