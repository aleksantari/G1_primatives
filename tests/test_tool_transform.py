from pathlib import Path

import numpy as np
import pytest
import yaml

from g1_primitives.spatial.pose import Pose, rpy_to_matrix
from g1_primitives.ee.hand_base import LEFT, RIGHT
from g1_primitives.grasp.tool_transform import (
    build_T_wristyaw_grasp, wrist_goal_from_grasp, approach_offset_for_side)

PALM = [0.1192, -0.0346, 0.0]


def _graspgenx_constants():
    """The LIVE derived grasp->wrist constants from configs/grasp.yaml (not a copy), so these
    tests break if the transform convention and the offset policy ever drift apart."""
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "grasp.yaml").read_text())
    gx = cfg["graspgenx"]
    return list(gx["wristyaw_grasp_rpy"]), list(gx["palm_offset_xyz"])


def test_build_T_wristyaw_grasp_side_mirror():
    # the stored rotation + offset describe the RIGHT hand; LEFT is the mirror across wrist y
    # (negate t.y; rotation -> S R S, still a proper rotation). GraspGenX derived constants.
    R = rpy_to_matrix(np.pi / 2, 0.0, np.pi / 2)
    off = [0.0442, 0.0414, 0.0]
    Tr, Tl = build_T_wristyaw_grasp(off, RIGHT, R), build_T_wristyaw_grasp(off, LEFT, R)
    np.testing.assert_allclose(Tr.translation, [0.0442, 0.0414, 0.0])
    np.testing.assert_allclose(Tl.translation, [0.0442, -0.0414, 0.0])    # y mirrored
    S = np.diag([1.0, -1.0, 1.0])
    np.testing.assert_allclose(Tl.rotation, S @ Tr.rotation @ S, atol=1e-12)
    assert np.isclose(np.linalg.det(Tl.rotation), 1.0)                    # proper rotation


def test_round_trip_recovers_commanded_wrist():
    # a known commanded wrist pose and a known fixed grasp->tool transform
    wrist = Pose(rpy_to_matrix(0.3, -0.5, 1.1), [0.4, -0.1, 0.9])
    T_wg = build_T_wristyaw_grasp(PALM, RIGHT, rpy_to_matrix(0.0, np.pi / 2, 0.0))
    T_pelvis_grasp = wrist * T_wg                   # where the gripper grasp frame ends up
    recovered = wrist_goal_from_grasp(T_pelvis_grasp, T_wg)
    np.testing.assert_allclose(recovered.homogeneous, wrist.homogeneous, atol=1e-9)


def test_identity_rotation_reduces_to_direct_backoff():
    # with grasp rotation == wrist rotation and T_wristyaw_grasp = (I, off), the GraspGenX
    # math reduces to the direct form wrist = target - R @ off (hardware-validated A-B algebra).
    R = rpy_to_matrix(np.pi, 0.0, 0.0)
    target = np.array([0.45, -0.2, 0.85])
    off = np.array([0.1192, 0.0346, 0.02])          # palm offset (RIGHT mirror) + grasp margin
    T_pelvis_grasp = Pose(R, target)
    T_wg = Pose(np.eye(3), off)
    wrist_goal = wrist_goal_from_grasp(T_pelvis_grasp, T_wg)
    np.testing.assert_allclose(wrist_goal.translation, target + R @ (-off), atol=1e-9)
    np.testing.assert_allclose(wrist_goal.rotation, R, atol=1e-12)


# ------------------------------------------------------------------------------------------
# plan_grasp approach offset x LEFT mirror -- the numeric lock for approach_offset_for_side.
# cuRobo forms the pre-grasp as `wrist_goal.multiply(T(axis * offset))` in the TOOL frame
# (curobo/_src/motion/motion_planner.py plan_grasp Step 2); replicated here with our Pose.
# ------------------------------------------------------------------------------------------

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


def test_tool_frame_offset_equals_09_world_backoff():
    # identity with 09_graspgen._approach_pose (world back-off dist along grasp +Z, orientation
    # kept) -- proves diagnose_candidates sweeps the SAME pre-grasps plan_grasp actually plans
    # to, on both sides. (_approach_pose replicated inline; scripts/ is not a package.)
    rpy, palm = _graspgenx_constants()
    R_wg = rpy_to_matrix(*rpy)
    dist = 0.15
    for side in (LEFT, RIGHT):
        T_wg = build_T_wristyaw_grasp(palm, side, R_wg)
        off_eff = approach_offset_for_side(side, "y", True, -dist)
        for T_pg in _random_grasp_poses(n=10, seed=3):
            wrist = wrist_goal_from_grasp(T_pg, T_wg)
            pre_tool = _pregrasp(wrist, off_eff)
            # scripts/09_graspgen.py::_approach_pose(grasp_wrist, grasp_pose, dist)
            axis = T_pg.rotation[:, 2]
            pre_09 = wrist.copy()
            pre_09.translation = wrist.translation - dist * axis
            # atol 1e-6: the 7-digit yaml pi (see test above); direction errors would be ~2*dist.
            np.testing.assert_allclose(pre_tool.translation, pre_09.translation, atol=1e-6)
            np.testing.assert_allclose(pre_tool.rotation, pre_09.rotation, atol=1e-12)
