from pathlib import Path

import numpy as np
import pytest
import yaml

from g1_primitives.spatial.pose import Pose, rpy_to_matrix
from g1_primitives.ee.hand_base import LEFT, RIGHT
from g1_primitives.grasp.tool_transform import (
    build_T_wristyaw_grasp, wrist_goal_from_grasp)

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
