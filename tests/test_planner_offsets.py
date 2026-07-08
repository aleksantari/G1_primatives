"""plan_grasp approach offset x LEFT mirror -- the numeric lock for
motion.curobo_planner.approach_offset_for_side (module-level, GPU-free: importing the
planner module pulls no torch/cuRobo). cuRobo forms the pre-grasp as
`wrist_goal.multiply(T(axis * offset))` in the TOOL frame (curobo/_src/motion/
motion_planner.py plan_grasp Step 2); replicated here with our Pose."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from g1_primitives.spatial.pose import Pose, rpy_to_matrix
from g1_primitives.ee.hand_base import LEFT, RIGHT
from g1_primitives.motion.planner import approach_offset_for_side
from g1_primitives.grasp.tool_transform import build_T_wristyaw_grasp, wrist_goal_from_grasp


def _graspgenx_constants():
    """The LIVE derived grasp->wrist constants from configs/grasp.yaml (not a copy), so these
    tests break if the transform convention and the offset policy ever drift apart."""
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "grasp.yaml").read_text())
    gx = cfg["graspgenx"]
    return list(gx["wristyaw_grasp_rpy"]), list(gx["palm_offset_xyz"])


def _pregrasp(wrist_goal: Pose, off_eff: float) -> Pose:
    """cuRobo's tool-frame approach pose for axis 'y': wrist_goal * T([0, off_eff, 0])."""
    return wrist_goal * Pose(np.eye(3), [0.0, float(off_eff), 0.0])


def _random_grasp_poses(n=25, seed=7):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        rpy = rng.uniform(-np.pi, np.pi, 3)
        t = rng.uniform([-0.2, -0.5, -0.3], [0.6, 0.5, 0.5])
        yield Pose(rpy_to_matrix(*rpy), t)


def test_pregrasp_backs_off_along_grasp_approach_both_sides():
    # THE left-arm bug lock: with the derived constants, the side-corrected tool-frame "y"
    # offset must displace the pre-grasp by -|offset| along the grasp approach axis (grasp +Z,
    # into the object) -- i.e. BACK OFF -- for BOTH hands. Without the LEFT sign flip the left
    # pre-grasp lands |offset| INTO the object (the failure seen under --collision-world).
    rpy, palm = _graspgenx_constants()
    R_wg = rpy_to_matrix(*rpy)
    cfg_offset = -0.10                                     # planner.yaml strategies convention
    for side in (LEFT, RIGHT):
        T_wg = build_T_wristyaw_grasp(palm, side, R_wg)
        off_eff = approach_offset_for_side(side, "y", True, cfg_offset)
        for T_pg in _random_grasp_poses():
            wrist = wrist_goal_from_grasp(T_pg, T_wg)
            pre = _pregrasp(wrist, off_eff)
            d = pre.translation - wrist.translation
            a = T_pg.rotation[:, 2]                        # grasp approach (into the object)
            # atol 1e-6: grasp.yaml stores rpy to 7 digits (pi = 3.1415927), so the clean-axis
            # map holds to ~1e-7 rad -- a WRONG sign would miss by 2*|offset| = 0.2, not 1e-7.
            np.testing.assert_allclose(d, -abs(cfg_offset) * a, atol=1e-6,
                                       err_msg=f"side={side}: pre-grasp not backing off")
            np.testing.assert_allclose(pre.rotation, wrist.rotation, atol=1e-12)


def test_approach_offset_for_side_scope():
    # the flip applies ONLY to (LEFT, tool-frame, axis 'y'); everything else passes through.
    assert approach_offset_for_side(LEFT, "y", True, -0.1) == pytest.approx(0.1)
    assert approach_offset_for_side(RIGHT, "y", True, -0.1) == pytest.approx(-0.1)
    assert approach_offset_for_side(LEFT, "x", True, -0.1) == pytest.approx(-0.1)
    assert approach_offset_for_side(LEFT, "z", True, -0.1) == pytest.approx(-0.1)
    assert approach_offset_for_side(LEFT, "y", False, -0.1) == pytest.approx(-0.1)


def test_tool_frame_offset_equals_world_backoff():
    # identity with the diagnostics pre-grasp reconstruction (world back-off dist along grasp
    # +Z, orientation kept) -- proves diagnose_candidates sweeps the SAME pre-grasps plan_grasp
    # actually plans to, on both sides.
    rpy, palm = _graspgenx_constants()
    R_wg = rpy_to_matrix(*rpy)
    dist = 0.15
    for side in (LEFT, RIGHT):
        T_wg = build_T_wristyaw_grasp(palm, side, R_wg)
        off_eff = approach_offset_for_side(side, "y", True, -dist)
        for T_pg in _random_grasp_poses(n=10, seed=3):
            wrist = wrist_goal_from_grasp(T_pg, T_wg)
            pre_tool = _pregrasp(wrist, off_eff)
            axis = T_pg.rotation[:, 2]
            pre_world = wrist.copy()
            pre_world.translation = wrist.translation - dist * axis
            # atol 1e-6: the 7-digit yaml pi (see test above); direction errors would be ~2*dist.
            np.testing.assert_allclose(pre_tool.translation, pre_world.translation, atol=1e-6)
            np.testing.assert_allclose(pre_tool.rotation, pre_world.rotation, atol=1e-12)


# --------------------------------------------------- portable robot-config paths
def test_portable_robot_cfg_reanchors_missing_paths(tmp_path):
    """The committed curobo yml bakes ABSOLUTE asset paths from the machine that generated
    it (build_g1_dex3.py). _portable_robot_cfg must re-anchor a missing path's assets/...
    tail onto THIS checkout (repo root = two dirs above the yml), keep existing paths
    untouched, and return a fresh dict per call (callers mutate tool_frames)."""
    import yaml
    from g1_primitives.motion.planner import _portable_robot_cfg

    cfg_dir = tmp_path / "configs" / "curobo"
    cfg_dir.mkdir(parents=True)
    exists = tmp_path / "somewhere.urdf"                    # a path that DOES exist
    exists.write_text("")
    yml = cfg_dir / "robot.yml"
    yml.write_text(yaml.safe_dump({"kinematics": {
        "urdf_path": "/home/otheruser/repos/G1/assets/g1/g1_29dof_mode_16_dex3.urdf",
        "asset_root_path": "/home/otheruser/repos/G1/assets/g1",
        "tool_frames": ["a", "b"]}}))

    kin = _portable_robot_cfg(str(yml))["kinematics"]
    assert kin["urdf_path"] == str(tmp_path / "assets" / "g1" /
                                   "g1_29dof_mode_16_dex3.urdf")   # re-anchored
    assert kin["asset_root_path"] == str(tmp_path / "assets" / "g1")

    yml.write_text(yaml.safe_dump({"kinematics": {"urdf_path": str(exists)}}))
    assert _portable_robot_cfg(str(yml))["kinematics"]["urdf_path"] == str(exists)

    a, b = _portable_robot_cfg(str(yml)), _portable_robot_cfg(str(yml))
    a["kinematics"]["tool_frames"] = ["x"]                  # fresh dicts: no cross-talk
    assert "tool_frames" not in b["kinematics"]


def test_portable_robot_cfg_real_config_resolves():
    """The COMMITTED config must load with both asset paths pointing at real files on
    this machine, whatever machine generated it."""
    import os
    from g1_primitives.motion.planner import _portable_robot_cfg
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kin = _portable_robot_cfg(
        os.path.join(here, "configs", "curobo", "g1_dex3_curobo.yml"))["kinematics"]
    assert os.path.isfile(kin["urdf_path"])
    assert os.path.isdir(kin["asset_root_path"])
