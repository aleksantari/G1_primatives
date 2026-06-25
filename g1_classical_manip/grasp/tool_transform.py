"""Grasp-frame -> tool (wrist-yaw) transform. Isolated + offline-testable.

A grasp model returns ``T_pelvis_grasp`` in the GRASP frame (GraspGenX: +Z = approach
into the object, +X = closing direction). We command the **wrist_yaw** link. With a fixed
``T_wristyaw_grasp`` (the grasp frame expressed in the wrist_yaw frame), the wrist goal is::

    T_pelvis_wristyaw = T_pelvis_grasp * inverse(T_wristyaw_grasp)

``T_wristyaw_grasp`` is seeded from the URDF-measured palm offset (translation — rigorous,
shared with the AprilTag path) plus an axis-map rotation (grasp(+Z,+X) -> wrist_yaw axes)
that is **EMPIRICAL** and must be verified on hardware (the AprilTag path never committed a
grasp orientation, so there is nothing to copy from).
"""
from __future__ import annotations

import numpy as np

from g1_classical_manip.spatial.pose import Pose
from g1_classical_manip.ee.hand_base import LEFT


def palm_offset(palm_offset_xyz, side: str) -> np.ndarray:
    """wrist_yaw -> palm grasp center; the lateral ``y`` is kept for the LEFT hand and
    negated for the RIGHT, exactly as ``07_pick_place._palm_offset`` (with the configured
    ``y = -0.0346`` that means left -y, right +y -- the A-B-faithful behavior)."""
    x, y, z = np.asarray(palm_offset_xyz, float)
    return np.array([x, y if side == LEFT else -y, z])


def build_T_wristyaw_grasp(palm_offset_xyz, side: str, R_wristyaw_grasp) -> Pose:
    """Fixed grasp frame expressed in the wrist_yaw frame: translation = palm offset
    (URDF-measured, side-mirrored), rotation = the EMPIRICAL grasp->wrist axis map."""
    return Pose(rotation=R_wristyaw_grasp,
                translation=palm_offset(palm_offset_xyz, side))


def wrist_goal_from_grasp(T_pelvis_grasp: Pose, T_wristyaw_grasp: Pose) -> Pose:
    """Wrist-yaw goal so the gripper grasp frame lands at ``T_pelvis_grasp``."""
    return T_pelvis_grasp * T_wristyaw_grasp.inverse()
