"""GraspSource interface + the GraspCandidate it emits."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from g1_classical_manip.spatial.pose import Pose


@dataclass
class GraspCandidate:
    """One candidate grasp, ready for the planner.

    ``wrist_goal`` is the wrist-yaw goal Pose in the pelvis frame — what
    ``primitives.move`` / ``plan_to_pose`` consume directly (the source has already
    applied the grasp->tool transform). ``grasp_pose`` is the raw model grasp frame
    (debug/telemetry)."""
    wrist_goal: Pose
    confidence: float = 1.0
    grasp_pose: Optional[Pose] = None
    extra: dict = field(default_factory=dict)


class GraspSource(ABC):
    """Produces ranked wrist-yaw goal Poses for a target object. The consumer (script /
    planner) picks the first reachable / collision-free candidate."""

    @abstractmethod
    def grasps(self, robot, side: str, target: str) -> List[GraspCandidate]:
        """Ranked candidates (highest confidence first); empty list = nothing found."""
