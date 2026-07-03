"""Config loading -- the ONLY yaml reader in the package.

``load_configs`` assembles the full config dict from ``configs/*.yaml``:
robot / planner / hands / perception / grasp, plus ONE camera file chosen by the
caller (``camera_sim.yaml`` for the Isaac sim stream, ``camera_real.yaml`` for the
real ZED -- the Robot facade picks it from the connect target).

Every pluggable seam is selected by a ``source:`` key naming a sibling block
(grasp.yaml ``source: graspgenx|sim_cloud``; perception.yaml ``source: sim_state``).
``load_configs`` raises an actionable ConfigMigrationError when a config still uses
a pre-rename key (``grasp_source:`` / ``detector:``) or a removed block (AprilTag),
so a stale local config fails loudly at load instead of silently falling back.
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


class ConfigMigrationError(ValueError):
    """A config file still uses a key/block from before the api restructure."""


# (config, old_key) -> what to do instead. Checked on every load so a stale local
# config fails loudly with the fix, instead of silently using a default.
_MIGRATED_KEYS = {
    ("grasp", "grasp_source"): "grasp.yaml: rename `grasp_source:` to `source:`",
    ("grasp", "apriltag"): ("grasp.yaml: the AprilTag grasp source was removed -- delete the "
                            "`apriltag:` block (`source:` is graspgenx | sim_cloud)"),
    ("perception", "detector"): "perception.yaml: rename `detector:` to `source:`",
    ("perception", "tag"): ("perception.yaml: the AprilTag detector was removed -- delete the "
                            "`tag:` block (`source:` is sim_state)"),
    ("perception", "ground_truth"): ("perception.yaml: the ground_truth detector was removed -- "
                                     "delete the `ground_truth:` block (`source:` is sim_state)"),
}


def validate_configs(cfg: Dict[str, Any]) -> None:
    """Raise ConfigMigrationError listing every pre-restructure key still present."""
    problems = [fix for (section, key), fix in _MIGRATED_KEYS.items()
                if key in (cfg.get(section) or {})]
    if problems:
        raise ConfigMigrationError(
            "config uses pre-restructure keys:\n  - " + "\n  - ".join(problems))


def load_configs(config_dir: str = DEFAULT_CONFIG_DIR,
                 camera_file: str = DEFAULT_CAMERA_CONFIG) -> Dict[str, Any]:
    cfg = {}
    for key, fname in _CONFIG_FILES.items():
        with open(os.path.join(config_dir, fname)) as f:
            cfg[key] = yaml.safe_load(f)
    with open(os.path.join(config_dir, camera_file)) as f:
        cfg["camera"] = yaml.safe_load(f)   # camera_sim.yaml | camera_real.yaml (ZED)
    validate_configs(cfg)
    return cfg
