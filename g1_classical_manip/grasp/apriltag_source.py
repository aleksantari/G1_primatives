"""AprilTag grasp source — the known-good A-B reference.

One candidate: the detected object pose + a fixed orientation, backed off by the
URDF-measured palm offset. Numerically identical to ``07_pick_place``'s inline grasp
geometry, so it stays the sanity check the GraspGenX source is validated against.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from g1_classical_manip import primitives as P
from g1_classical_manip.spatial.pose import Pose, rpy_to_matrix
from g1_classical_manip.grasp.base import GraspSource, GraspCandidate
from g1_classical_manip.grasp.tool_transform import palm_offset


class AprilTagGraspSource(GraspSource):
    def __init__(self, palm_offset_xyz, grasp_offset, grasp_rpy: Optional[list] = None):
        self.palm_offset_xyz = np.asarray(palm_offset_xyz, float)
        self.grasp_offset = np.asarray(grasp_offset, float)
        self.grasp_rpy = grasp_rpy

    def grasps(self, robot, side: str, target: str) -> List[GraspCandidate]:
        det = P.detect(robot, target)
        if det is None:
            return []
        q = robot.arm.get_current_dual_arm_q()
        R = (rpy_to_matrix(*self.grasp_rpy) if self.grasp_rpy is not None
             else robot.planner.fk(side, q).rotation.copy())     # FK orientation = reachable
        off = palm_offset(self.palm_offset_xyz, side) + self.grasp_offset
        wrist = det.pose.translation + R @ (-off)                # 07_pick_place._wrist_goal
        return [GraspCandidate(wrist_goal=Pose(rotation=R, translation=wrist),
                               confidence=1.0)]
