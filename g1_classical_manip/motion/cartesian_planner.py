"""Cartesian planner v1: linear SE(3) interpolation between waypoints, dual-arm
IK per sample (warm-started off the previous solution), continuity-checked.

Emits the standard ``JointPath`` (same dataclass cuRobo will later emit). Holds
the non-moving arm by re-targeting its *current* EE pose -- single-arm motion is
never produced by slicing the 14-vector.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pinocchio as pin

from g1_classical_manip.motion.planner_base import (
    Planner, Goal, World, JointPath, CartesianWaypoint, DOF)


class PlanningError(RuntimeError):
    pass


def se3_interp(A: pin.SE3, B: pin.SE3, u: float) -> pin.SE3:
    """Geodesic SE(3) interpolation: A at u=0, B at u=1."""
    nu = pin.log6(A.inverse() * B)          # Motion (twist) from A to B
    return A * pin.exp6(u * nu.vector)


def _trans_dist(A: pin.SE3, B: pin.SE3) -> float:
    return float(np.linalg.norm(A.translation - B.translation))


def _rot_angle(A: pin.SE3, B: pin.SE3) -> float:
    return float(np.linalg.norm(pin.log3(A.rotation.T @ B.rotation)))


class CartesianPlanner(Planner):
    def __init__(self, ik, samples_per_meter: float = 200.0, min_samples: int = 20,
                 rot_samples_per_rad: float = 30.0, continuity_jump_rad: float = 0.35):
        self.ik = ik
        self.samples_per_meter = samples_per_meter
        self.min_samples = int(min_samples)
        self.rot_samples_per_rad = rot_samples_per_rad
        self.continuity_jump_rad = continuity_jump_rad

    @classmethod
    def from_config(cls, ik, planner_cfg: dict):
        c = (planner_cfg or {}).get("cartesian", {})
        return cls(ik,
                   samples_per_meter=c.get("samples_per_meter", 200.0),
                   min_samples=c.get("min_samples", 20),
                   rot_samples_per_rad=c.get("rot_samples_per_rad", 30.0),
                   continuity_jump_rad=c.get("continuity_jump_rad", 0.35))

    def _n_samples(self, AL, BL, AR, BR) -> int:
        d = max(_trans_dist(AL, BL), _trans_dist(AR, BR))
        a = max(_rot_angle(AL, BL), _rot_angle(AR, BR))
        return max(self.min_samples,
                   int(np.ceil(d * self.samples_per_meter)),
                   int(np.ceil(a * self.rot_samples_per_rad)))

    def _check_workspace(self, world: Optional[World], wp: CartesianWaypoint):
        if world is None or world.workspace_box is None:
            return
        for T in (wp.left, wp.right):
            if T is not None and not world.workspace_box.contains(T.translation):
                raise PlanningError(
                    f"waypoint '{wp.label}' target {np.round(T.translation,3)} "
                    f"outside workspace box")

    def plan(self, start_q14: np.ndarray, goal: Goal,
             world: Optional[World] = None) -> JointPath:
        q_cur = np.asarray(start_q14, dtype=float).reshape(DOF)
        TL_cur, TR_cur = self.ik.fk(q_cur)
        qs = [q_cur.copy()]
        labels = []

        for wp in goal.waypoints:
            self._check_workspace(world, wp)
            tgt_L = wp.left if wp.left is not None else TL_cur
            tgt_R = wp.right if wp.right is not None else TR_cur
            n = self._n_samples(TL_cur, tgt_L, TR_cur, tgt_R)
            for k in range(1, n + 1):
                u = k / n
                Ti_L = se3_interp(TL_cur, tgt_L, u)
                Ti_R = se3_interp(TR_cur, tgt_R, u)
                q_sol, _ = self.ik.solve_ik(Ti_L.homogeneous, Ti_R.homogeneous, q_cur)
                jump = float(np.linalg.norm(q_sol - q_cur))
                if jump > self.continuity_jump_rad:
                    raise PlanningError(
                        f"IK discontinuity {jump:.3f} rad at waypoint "
                        f"'{wp.label}' sample {k}/{n} (branch flip / unreachable)")
                qs.append(q_sol.copy())
                q_cur = q_sol
            # advance using the achieved pose to avoid accumulating IK residual
            TL_cur, TR_cur = self.ik.fk(q_cur)
            labels.append((len(qs) - 1, wp.label))

        return JointPath(np.array(qs), meta={"waypoint_index": labels})
