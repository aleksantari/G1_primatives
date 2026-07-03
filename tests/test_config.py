"""Config loading + the `source:` key-migration guard (GPU-free).

The shipped configs/ must load clean with the new unified `source:` selectors; a
stale local config still using a pre-restructure key must fail LOUDLY with the fix
in the message (not silently fall back to a default source)."""
import pytest

from g1_primitives.config import (load_configs, validate_configs,
                                  ConfigMigrationError, DEFAULT_CONFIG_DIR)


def test_shipped_configs_load_clean_with_source_selectors():
    cfg = load_configs()                       # the real configs/ dir
    assert cfg["grasp"]["source"] in ("graspgenx", "sim_cloud")
    assert cfg["perception"]["source"] == "sim_state"
    assert "grasp_source" not in cfg["grasp"]
    assert "detector" not in cfg["perception"]
    for key in ("robot", "planner", "hands", "camera"):
        assert cfg[key], key


def test_old_grasp_source_key_raises_actionable():
    cfg = {"grasp": {"grasp_source": "graspgenx"}, "perception": {}}
    with pytest.raises(ConfigMigrationError, match="`grasp_source:` to `source:`"):
        validate_configs(cfg)


def test_old_detector_key_raises_actionable():
    cfg = {"grasp": {}, "perception": {"detector": "sim_state"}}
    with pytest.raises(ConfigMigrationError, match="`detector:` to `source:`"):
        validate_configs(cfg)


def test_removed_apriltag_blocks_raise():
    with pytest.raises(ConfigMigrationError, match="AprilTag grasp source was removed"):
        validate_configs({"grasp": {"apriltag": {}}, "perception": {}})
    with pytest.raises(ConfigMigrationError, match="AprilTag detector was removed"):
        validate_configs({"grasp": {}, "perception": {"tag": {}}})
    with pytest.raises(ConfigMigrationError, match="ground_truth detector was removed"):
        validate_configs({"grasp": {}, "perception": {"ground_truth": {}}})


def test_multiple_problems_all_listed():
    cfg = {"grasp": {"grasp_source": "x"}, "perception": {"detector": "y"}}
    with pytest.raises(ConfigMigrationError) as ei:
        validate_configs(cfg)
    msg = str(ei.value)
    assert "grasp_source" in msg and "detector" in msg


def test_clean_and_empty_sections_pass():
    validate_configs({"grasp": {"source": "graspgenx"}, "perception": {"source": "sim_state"}})
    validate_configs({})                       # missing sections are fine here
    validate_configs({"grasp": None, "perception": None})   # yaml empty files
