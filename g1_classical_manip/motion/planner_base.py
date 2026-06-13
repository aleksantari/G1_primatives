"""Planner-agnostic interfaces and data containers for the motion stack.

Hard seam (see CLAUDE.md rule 1):
    Planner.plan(start_q14, goal, world) -> JointPath   (geometry, NO timing)
    Retimer.retime(JointPath)            -> JointTrajectory
    Executor.run(JointTrajectory)

The Cartesian planner (v1) and the cuRobo adapter (v2) MUST both return the same
``JointPath`` so that swapping ``planner: cartesian -> curobo`` is a one-line config
change. IK never streams to the robot; only the executor holds the controller.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import pinocchio as pin

DOF = 14  # dual arm: left 7 + right 7, upstream joint order (G1_29_JointArmIndex)


@dataclass
class CartesianWaypoint:
    """A dual-arm Cartesian waypoint. ``None`` for an arm means 'hold this arm's
    current EE pose' -- single-arm motion is produced this way, never by slicing
    the 14-vector."""
    left: Optional[pin.SE3] = None
    right: Optional[pin.SE3] = None
    label: str = ""


@dataclass
class Goal:
    """Ordered Cartesian waypoints to traverse after the start configuration."""
    waypoints: List[CartesianWaypoint] = field(default_factory=list)


@dataclass
class Box:
    """Axis-aligned box in the pelvis frame (meters)."""
    lo: np.ndarray
    hi: np.ndarray

    def contains(self, p) -> bool:
        p = np.asarray(p, dtype=float)
        return bool(np.all(p >= self.lo) and np.all(p <= self.hi))


@dataclass
class World:
    """Collision / workspace context. The Cartesian planner only uses
    ``workspace_box``; cuRobo consumes the obstacle cuboids too."""
    workspace_box: Optional[Box] = None
    obstacles: List[Box] = field(default_factory=list)  # (table, block, ...) cuboids


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


class Planner(ABC):
    """Geometry-only planner. Returns a JointPath; never a timed trajectory."""

    @abstractmethod
    def plan(self, start_q14: np.ndarray, goal: Goal,
             world: Optional[World] = None) -> JointPath:
        ...
