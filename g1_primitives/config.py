"""Config loading -- the ONLY yaml reader in the package.

``load_configs`` assembles the full config dict from ``configs/*.yaml``:
robot / planner / hands / perception / grasp, plus ONE camera file chosen by the
caller (``camera_sim.yaml`` for the Isaac sim stream, ``camera_real.yaml`` for the
real ZED -- the Robot facade picks it from the connect target). Key-migration
validation (actionable errors on renamed config keys) is added alongside the
`source:` selector rename.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml

_CONFIG_FILES = {"robot": "robot.yaml", "planner": "planner.yaml", "hands": "hands.yaml",
                 "perception": "perception.yaml", "grasp": "grasp.yaml"}
# Camera config is chosen separately (sim mono vs real ZED stereo); Robot.connect
# passes camera_real.yaml for target="real".
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
