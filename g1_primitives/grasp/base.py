"""GraspSource interface + the GraspCandidate it emits + the SourceSnapshot it retains."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from g1_primitives.spatial.pose import Pose


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


@dataclass
class SourceSnapshot:
    """What the source SAW on its most recent ``grasps()`` call -- the TYPED cross-object
    contract replacing ad-hoc attribute peeking. ``mask`` is the target's 2D segmentation
    (H,W bool, depth-aligned; the collision world reads it to CUT the target out of the
    ESDF via ``collision_world.exclude_object``). ``cloud`` is the object PointCloud the
    grasp model consumed (the perception-validation tool compares it against sim ground
    truth). Sources reset it to None on entry, so a failed capture never leaves stale
    state behind."""
    target: str
    mask: Optional[Any] = None            # (H,W) bool np.ndarray
    cloud: Optional[Any] = None           # spatial.pointcloud.PointCloud
    n_candidates: int = 0


class GraspSource(ABC):
    """Produces ranked wrist-yaw goal Poses for a target object. The consumer (script /
    planner) picks the first reachable / collision-free candidate."""

    last_snapshot: Optional[SourceSnapshot] = None    # set by every grasps() call

    @abstractmethod
    def grasps(self, robot, side: str, target: str) -> List[GraspCandidate]:
        """Ranked candidates (highest confidence first); empty list = nothing found.
        Implementations MUST reset ``last_snapshot`` on entry and populate it with what
        they saw (mask/cloud) before returning."""
