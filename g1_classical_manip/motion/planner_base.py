"""Geometry/timing containers for the motion stack (pinocchio-free).

    CuroboArmPlanner.plan_to_pose(...) -> JointPath (geometry, no timing)
    Retimer.retime(JointPath)          -> JointTrajectory
    Executor.run(JointTrajectory)

Poses are `spatial.pose.Pose` in the pelvis frame (see spatial/pose.py); planning
goals are passed to the planner directly, so the old Cartesian Goal/Waypoint/World
containers are gone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

DOF = 14  # dual arm: left 7 + right 7, upstream joint order (G1_29_JointArmIndex)


@dataclass
class JointPath:
    """Geometric joint-space path, no timing. ``q`` is (N, 14)."""
    q: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=float).reshape(-1, DOF)

    @property
    def n(self) -> int:
        return self.q.shape[0]

    def max_consecutive_jump(self) -> float:
        if self.n < 2:
            return 0.0
        return float(np.max(np.linalg.norm(np.diff(self.q, axis=0), axis=1)))


@dataclass
class JointTrajectory:
    """Time-parameterized trajectory. t (M,), q/qd/qdd each (M, 14)."""
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float).reshape(-1)
        self.q = np.asarray(self.q, dtype=float).reshape(-1, DOF)
        self.qd = np.asarray(self.qd, dtype=float).reshape(-1, DOF)
        self.qdd = np.asarray(self.qdd, dtype=float).reshape(-1, DOF)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size else 0.0

    def sample(self, t: float) -> np.ndarray:
        """Linear-interpolate q at time t (clamped to the trajectory span)."""
        return np.array([np.interp(t, self.t, self.q[:, j]) for j in range(DOF)])
