"""Ruckig retiming: JointPath (geometry) -> JointTrajectory (timed).

Approach: parameterize the path by joint-space **arc length** s, where the unit
tangent dq/ds satisfies |dq_j/ds| <= 1 for every joint. A single-DoF jerk-limited
profile s(t) from Ruckig with bounds (min_j v_max, min_j a_max, min_j j_max) then
keeps every joint within its limits, and we resample q along the path at s(t).

The path is retimed **segment-by-segment between key waypoints, coming to rest at
each** (the planner records waypoint indices in JointPath.meta). This is the
natural pick-place behaviour (stop to grasp/release) and -- crucially -- it
removes the acceleration cusps that a single non-stop pass would create where the
Cartesian path reverses direction (e.g. descend -> lift). Within a segment the IK
solution varies smoothly, so curvature (hence acceleration) stays bounded.

This makes the retimer the sole speed authority: if the arm controller's velocity
clip fires during nominal execution, the limits here are wrong (treat as a bug).
"""
from __future__ import annotations

import logging
from typing import Optional, List

import numpy as np
from ruckig import Ruckig, InputParameter, Trajectory

from g1_classical_manip.motion.planner_base import JointPath, JointTrajectory, DOF

_log = logging.getLogger(__name__)


def _as14(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        a = np.full(DOF, float(a))
    return a.reshape(DOF)


def _segment_boundaries(path: JointPath) -> List[int]:
    wi = (path.meta or {}).get("waypoint_index")
    bounds = [0]
    if wi:
        for idx, _label in wi:
            if idx > bounds[-1]:
                bounds.append(int(idx))
    if bounds[-1] != path.n - 1:
        bounds.append(path.n - 1)
    return bounds


def _ruckig_s_of_t(L: float, vmax: float, amax: float, jmax: float, dt: float):
    """Return (times, s_values) for a rest-to-rest single-DoF move 0 -> L."""
    if L <= 1e-9:
        return np.array([0.0]), np.array([0.0])
    inp = InputParameter(1)
    inp.current_position = [0.0]
    inp.current_velocity = [0.0]
    inp.current_acceleration = [0.0]
    inp.target_position = [L]
    inp.target_velocity = [0.0]
    inp.target_acceleration = [0.0]
    inp.max_velocity = [vmax]
    inp.max_acceleration = [amax]
    inp.max_jerk = [jmax]
    otg = Ruckig(1)
    traj = Trajectory(1)
    res = otg.calculate(inp, traj)
    if int(res) < 0:
        raise RuntimeError(f"Ruckig failed to retime segment: {res}")
    dur = float(traj.duration)
    ts = np.arange(0.0, dur + 0.5 * dt, dt)
    s = np.array([traj.at_time(min(float(t), dur))[0] for t in ts])
    return ts, s


def retime(path: JointPath, *, control_hz: float = 250.0,
           max_velocity=2.0, max_acceleration=5.0, max_jerk=30.0) -> JointTrajectory:
    vmax = _as14(max_velocity)
    amax = _as14(max_acceleration)
    jmax = _as14(max_jerk)
    dt = 1.0 / float(control_hz)
    sv, sa, sj = float(np.min(vmax)), float(np.min(amax)), float(np.min(jmax))

    q = path.q
    if q.shape[0] < 2:
        z = np.zeros((1, DOF))
        return JointTrajectory(np.array([0.0]), q[:1].copy(), z, z,
                               meta={"degenerate": True})

    vlim, alim = float(np.min(vmax)), float(np.min(amax))
    bounds = _segment_boundaries(path)
    times: List[float] = [0.0]
    qcols: List[np.ndarray] = [q[0].copy()]
    qdcols: List[np.ndarray] = [np.zeros(DOF)]
    qddcols: List[np.ndarray] = [np.zeros(DOF)]
    t_off = 0.0
    max_time_scale = 1.0

    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        sub = q[b0:b1 + 1]
        seglen = np.linalg.norm(np.diff(sub, axis=0), axis=1)
        keep = np.concatenate([[True], seglen > 1e-9])
        subk = sub[keep]
        if subk.shape[0] < 2:
            continue
        s_knots = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(subk, axis=0), axis=1))])
        L = float(s_knots[-1])

        # The arc-length bound (sv, sa, sj) only guarantees joint limits for a
        # STRAIGHT joint path. The IK path curves, so the curvature term ~s_dot^2
        # adds acceleration. Iteratively slow this segment (scale time by k) until
        # the ACTUAL qd/qdd are within limits: qd ~ 1/k, qdd ~ 1/k^2.
        k = 1.0
        for _ in range(6):
            ts, s_t = _ruckig_s_of_t(L, sv / k, sa / k**2, sj / k**3, dt)
            q_seg = np.column_stack([np.interp(s_t, s_knots, subk[:, j]) for j in range(DOF)])
            if len(ts) < 2:
                qd_seg = np.zeros_like(q_seg)
                qdd_seg = np.zeros_like(q_seg)
                break
            qd_seg = np.gradient(q_seg, ts, axis=0)
            qdd_seg = np.gradient(qd_seg, ts, axis=0)
            rv = np.max(np.abs(qd_seg)) / vlim
            ra = np.max(np.abs(qdd_seg)) / alim
            f = max(rv, np.sqrt(ra))
            if f <= 1.02:
                break
            k *= f * 1.05
        max_time_scale = max(max_time_scale, k)

        for i in range(1, len(ts)):
            times.append(t_off + ts[i])
            qcols.append(q_seg[i])
            qdcols.append(qd_seg[i])
            qddcols.append(qdd_seg[i])
        # force exact rest at the segment join (continuity, no spurious join accel)
        if len(qdcols) > 1:
            qdcols[-1] = np.zeros(DOF)
            qddcols[-1] = np.zeros(DOF)
        t_off = times[-1]

    if len(times) < 2:  # path collapsed to a single point (start == goal)
        z = np.zeros((1, DOF))
        return JointTrajectory(np.array([0.0]), q[:1].copy(), z, z,
                               meta={"degenerate": True})

    t = np.array(times)
    q_t = np.vstack(qcols)
    qd = np.vstack(qdcols)
    qdd = np.vstack(qddcols)

    near_singular = max_time_scale > 3.0
    if near_singular:
        _log.warning("[retimer] path slowed %.1fx to respect accel limits -- "
                     "likely a near-singular / poorly-conditioned IK path; "
                     "consider re-orienting the grasp or using cuRobo.",
                     max_time_scale)
    return JointTrajectory(t, q_t, qd, qdd, meta={
        "n_segments": len(bounds) - 1,
        "duration": float(t[-1]),
        "limits": {"v": vmax.tolist(), "a": amax.tolist(), "j": jmax.tolist()},
        "max_qd": float(np.max(np.abs(qd))),
        "max_qdd": float(np.max(np.abs(qdd))),
        "max_time_scale": float(max_time_scale),
        "near_singular": bool(near_singular),
    })


def retime_from_config(path: JointPath, planner_cfg: dict) -> JointTrajectory:
    rt = (planner_cfg or {}).get("retimer", {})
    return retime(path,
                  control_hz=rt.get("control_hz", 250.0),
                  max_velocity=rt.get("max_velocity", 2.0),
                  max_acceleration=rt.get("max_acceleration", 5.0),
                  max_jerk=rt.get("max_jerk", 30.0))
