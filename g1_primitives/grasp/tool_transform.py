"""Grasp-frame -> tool (wrist-yaw) transform. Isolated + offline-testable.

A grasp model returns ``T_pelvis_grasp`` in the GRASP frame (GraspGenX: +Z = approach
into the object, +X = the jaw-closing/opposition axis). We command the **wrist_yaw** link.
With a fixed ``T_wristyaw_grasp`` (the grasp frame expressed in the wrist_yaw frame), the
wrist goal is::

    T_pelvis_wristyaw = T_pelvis_grasp * inverse(T_wristyaw_grasp)

For the GraspGenX path ``T_wristyaw_grasp`` is **DERIVED from kinematics**:
``scripts/derive_graspgenx_tool_transform.py`` registers GraspGenX's grasp convention against
the Dex3 URDF -- rotation = the fixed grasp(+Z approach,+X closing)->wrist_yaw axis map, and
translation = our power_close contact midpoint (hand FK) minus the GraspGenX fingertip depth
along the approach axis. The config (``grasp.yaml: graspgenx``) stores the RIGHT-hand values;
the LEFT hand is the mirror image across the wrist Y-plane, applied here.
"""
from __future__ import annotations

from typing import List

import numpy as np

from g1_primitives.spatial.pose import Pose
from g1_primitives.ee.hand_base import LEFT
from g1_primitives.grasp.base import GraspCandidate
from g1_primitives.latency import LOG   # latency instrumentation (no-op unless enabled)

_MIRROR_Y = np.diag([1.0, -1.0, 1.0])      # reflect a wrist-frame transform R/t across Y (R<->L)


def build_T_wristyaw_grasp(palm_offset_xyz, side: str, R_wristyaw_grasp) -> Pose:
    """Fixed grasp frame in the wrist_yaw frame (GraspGenX path). The passed rotation + offset
    describe the RIGHT hand (derived constants); the LEFT hand is their mirror image across the
    wrist Y-plane -- reflect both ``R`` (S R S, still a proper rotation) and ``t`` (negate y)."""
    R = np.asarray(R_wristyaw_grasp, float).reshape(3, 3)
    t = np.asarray(palm_offset_xyz, float).reshape(3)
    if side == LEFT:
        R = _MIRROR_Y @ R @ _MIRROR_Y
        t = _MIRROR_Y @ t
    return Pose(rotation=R, translation=t)


def wrist_goal_from_grasp(T_pelvis_grasp: Pose, T_wristyaw_grasp: Pose) -> Pose:
    """Wrist-yaw goal so the gripper grasp frame lands at ``T_pelvis_grasp``."""
    return T_pelvis_grasp * T_wristyaw_grasp.inverse()


def candidates_from_grasps(grasps, conf, side: str, palm_offset_xyz,
                           R_wristyaw_grasp, branch_tags=None) -> List[GraspCandidate]:
    """``(K,4,4)`` pelvis-frame grasps + ``(K,)`` confidences -> ranked ``GraspCandidate``s
    (confidence-descending), each mapped to a wrist-yaw goal via the fixed grasp->tool
    transform. Shared by the GraspGenX and sim-cloud sources so the grasp->wrist mapping
    lives in one place. Returns ``[]`` for an empty / length-0 input.

    ``branch_tags`` (optional, GraspGenX protocol v2) is the per-grasp ``"obb"``/``"diff"``
    list aligned to ``grasps``; each tag is stashed on ``cand.extra["branch_tag"]`` indexed by
    the SAME sort, so the tag stays attached to its grasp regardless of confidence order."""
    grasps = np.asarray(grasps, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)
    k = min(grasps.shape[0], conf.shape[0])            # guard a grasps/conf length mismatch
    if k == 0:
        return []
    grasps, conf = grasps[:k], conf[:k]
    tags = list(branch_tags) if branch_tags is not None else None
    with LOG.span("tool_transform"):                   # grasp -> wrist-yaw goal (+ left mirror)
        T_wg = build_T_wristyaw_grasp(palm_offset_xyz, side, R_wristyaw_grasp)
        out: List[GraspCandidate] = []
        for i in np.argsort(-conf):                    # confidence descending
            T_pelvis_grasp = Pose.from_homogeneous(grasps[i])
            cand = GraspCandidate(wrist_goal=wrist_goal_from_grasp(T_pelvis_grasp, T_wg),
                                  confidence=float(conf[i]), grasp_pose=T_pelvis_grasp)
            if tags is not None and i < len(tags):
                cand.extra["branch_tag"] = tags[i]
            out.append(cand)
    return out
