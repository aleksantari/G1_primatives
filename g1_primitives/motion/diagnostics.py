"""Planner diagnostics -- explain WHY cuRobo rejects a config or a grasp plan.

Stateless module functions taking a ``CuroboArmPlanner`` as their first argument (a
documented FRIEND module: they may read planner internals such as ``_grasp_planner``,
``_cw_loaded`` and ``_mp``). Kept out of motion/planner.py so the planning core stays
lean; nothing here mutates planner state.

  diagnose(planner, q, side, ...)        per-element breakdown of a rejected CONFIG
                                         (self-collision pairs, ESDF hits, joint
                                         limits, wrist singularity)
  diagnose_pose(planner, side, pose)     the END-state counterpart for a target pose
  diagnose_candidates(planner, ...)      sweep a goalset's grasp/pre-grasp reachability
  world_check(planner, side, q)          TRUTH-TEST vs the grasp planner's OWN ESDF gate
                                         (the exact 'Start or End state in collision'
                                         predicate, eta=0)
  explain_failure(planner, ...)          the full failure post-mortem (START diagnose +
                                         world_check + candidate sweep + viz overlays)
  set_curobo_log_level(level)            turn up cuRobo's own logger

GPU-free at import (torch/cuRobo load lazily inside the planner methods these call).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from g1_primitives.motion.trajectory import DOF
from g1_primitives.motion.planner import PlanningError, REPO_ARM
from g1_primitives.ee.hand_base import LEFT


def set_curobo_log_level(level: str = "debug") -> None:
    """Turn up cuRobo's OWN logger so plan_grasp's internal failure reasons get printed instead of
    swallowed: the graph planner's 'Start or End state in collision' (graph_planner_prm), the IK
    stage's 'No grasp in goal set was reachable' (motion_planner), and per-stage trajopt warnings.
    `level` in {debug, info, warning, error}; cuRobo defaults to 'warning'. Call once before planning
    (09/12 expose it as --debug-planner). Best-effort: never blocks if the logging API moves."""
    try:
        from curobo.logging import setup_logger
        setup_logger(level)
        print(f"[curobo] log level -> {level}")
    except Exception as e:                       # noqa: BLE001 - diagnostics only, never block a run
        print(f"set_curobo_log_level: could not set cuRobo log level ({e})")


def _quat_to_R(wxyz):
    w, x, y, z = [float(v) for v in wxyz]
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)]])

def _wrist_jacobian_fd(planner, q_repo14, side: str, eps: float = 1e-5):
    """The 6x7 geometric Jacobian (linear 3 + angular 3, pelvis frame) of `side`'s wrist w.r.t.
    that arm's 7 joints, by FINITE DIFFERENCE over cuRobo FK. cuRobo's KinematicsState.tool_jacobians
    is a zero placeholder unless the model is built with compute_jacobian=True, so we FD it off the
    reliable tool_poses (validated stable for eps 1e-3..1e-6). Used for the singularity measure."""
    q = np.asarray(q_repo14, float).reshape(DOF)
    cols = [REPO_ARM.index(n) for n in (REPO_ARM[0:7] if side == LEFT else REPO_ARM[7:DOF])]
    p0 = planner.fk(side, q)
    R0 = _quat_to_R(p0.quaternion_wxyz())
    J = np.zeros((6, 7))
    for n, k in enumerate(cols):
        qp = q.copy(); qp[k] += eps
        pp = planner.fk(side, qp)
        J[0:3, n] = (pp.translation - p0.translation) / eps
        Rrel = _quat_to_R(pp.quaternion_wxyz()) @ R0.T
        v = np.array([Rrel[2, 1]-Rrel[1, 2], Rrel[0, 2]-Rrel[2, 0], Rrel[1, 0]-Rrel[0, 1]])
        ang = np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))
        J[3:6, n] = (v / 2 if ang < 1e-9 else (ang / (2*np.sin(ang))) * v) / eps
    return J

def diagnose(planner, q_repo14, side: str, world_points=None, voxel_size: float = 0.01,
             near_limit_deg: float = 5.0, sing_eps: float = 0.01, top: int = 8, show: bool = True):
    """Explain WHY a config is rejected by cuRobo -- the per-element breakdown behind a generic
    'Start or End state in collision' / 'No grasp in goal set was reachable'. At `q_repo14` (repo
    14-vector) for arm `side`, reports:
      * SELF-collision: which link/sphere PAIRS overlap + penetration mm (ignore-matrix faithful).
      * WORLD/ESDF: which spheres sit inside the depth voxels + depth mm (uses the SAME occupied
        points 09/12 overlay in red; pass `world_points` or defaults to collision_world_points()).
      * JOINT LIMITS: the side arm's joints closest to a limit (deg), flags any within near_limit_deg.
      * SINGULARITY: condition number + Yoshikawa manipulability of the side's 6x7 wrist Jacobian
        (flags cond > cond_warn) -- the near-singular configs seen at the task-space edges.
    Returns a dict (incl. offending sphere centers/radii for a red overlay). `show` prints a report.
    Runs on the MAIN 14-DoF model (planner._mp) -- the same sphere set + limits the grasp solve uses."""
    q = np.asarray(q_repo14, float).reshape(DOF)
    ks = planner._mp.compute_kinematics(planner._joint_state(q))
    sph = ks.robot_spheres.detach().cpu().numpy().reshape(-1, 4)   # (443, 4): xyz + radius
    links = planner._sphere_link_names()
    c, r = sph[:, :3], sph[:, 3]

    # --- (a) self-collision: pairwise overlap over cuRobo's checked pairs (ignores pre-removed)
    pairs, pad = planner._self_collision_pairs()
    pi, pj = pairs[:, 0], pairs[:, 1]
    valid = (r[pi] > 0) & (r[pj] > 0)
    dist = np.linalg.norm(c[pi] - c[pj], axis=1)
    # padded overlap (>0 => cuRobo's self-collision would fire): radii + per-sphere padding - gap
    overlap = (r[pi] + r[pj] + pad[pi] + pad[pj]) - dist
    hit = valid & (overlap > 0) & (np.asarray(links)[pi] != np.asarray(links)[pj])
    self_rows = [{"i": int(pi[k]), "j": int(pj[k]), "link_i": links[pi[k]], "link_j": links[pj[k]],
                  "penetration_mm": float(overlap[k] * 1000.0)} for k in np.nonzero(hit)[0]]
    self_rows.sort(key=lambda d: -d["penetration_mm"])

    # --- (b) world/ESDF: robot spheres inside the occupied voxels (sphere-vs-nearest-voxel)
    wp = world_points if world_points is not None else planner.collision_world_points()
    world_rows = []
    if wp is not None and len(wp):
        wp = np.asarray(wp, float).reshape(-1, 3)
        act = np.nonzero(r > 0)[0]
        for s in act:
            d = np.linalg.norm(wp - c[s], axis=1).min()           # nearest occupied voxel centre
            depth = (r[s] + 0.5 * voxel_size) - d                 # >0 => sphere overlaps a voxel
            if depth > 0:
                world_rows.append({"i": int(s), "link": links[s], "depth_mm": float(depth * 1000.0)})
        world_rows.sort(key=lambda d: -d["depth_mm"])

    # --- (c) joint limits: this side's 7 joints, closest to a bound first
    jl = planner._mp.kinematics.get_joint_limits()
    lo = jl.position[0].detach().cpu().numpy(); hi = jl.position[1].detach().cpu().numpy()
    side_names = set(REPO_ARM[0:7]) if side == LEFT else set(REPO_ARM[7:DOF])
    cols = [k for k, n in enumerate(planner.active) if n in side_names]
    deg = 180.0 / np.pi
    qa = q[[REPO_ARM.index(planner.active[k]) for k in cols]]
    limit_rows = []
    for k, qk in zip(cols, qa):
        m = min(qk - lo[k], hi[k] - qk)                            # rad to nearest bound
        limit_rows.append({"joint": planner.active[k], "q_deg": float(qk * deg),
                           "margin_deg": float(m * deg), "lo_deg": float(lo[k] * deg),
                           "hi_deg": float(hi[k] * deg)})
    limit_rows.sort(key=lambda d: d["margin_deg"])

    # --- (d) singularity: side's 6x7 wrist Jacobian (FD -- cuRobo tool_jacobians is a zero
    # placeholder). sigma_min = distance-to-rank-loss (the flag; healthy configs run ~0.02-0.08),
    # + condition number + Yoshikawa manipulability sqrt(det(J J^T)).
    J = _wrist_jacobian_fd(planner, q, side)
    sv = np.linalg.svd(J, compute_uv=False)
    smin = float(sv.min()); smax = float(sv.max())
    cond = float(smax / smin) if smin > 1e-9 else float("inf")
    manip = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
    sing = {"cond": cond, "manip": manip, "sigma_min": smin}

    off_idx = sorted({d["i"] for d in self_rows} | {d["j"] for d in self_rows}
                     | {d["i"] for d in world_rows})
    out = {"self": self_rows, "world": world_rows, "limits": limit_rows, "singularity": sing,
           "offending_centers": c[off_idx].astype(np.float32) if off_idx else np.zeros((0, 3), np.float32),
           "offending_radii": r[off_idx].astype(np.float32) if off_idx else np.zeros((0,), np.float32)}
    if show:
        _print_diagnose(out, side, near_limit_deg, sing_eps, top)
    return out

def _print_diagnose(out, side, near_limit_deg, sing_eps, top):
    print(f"[diagnose] side={side}  (self-collision faithful to the ignore matrix; "
          f"world = sphere vs the red ESDF voxels)")
    sr = out["self"]
    print(f"  SELF-COLLISION: {len(sr)} overlapping pair(s)"
          + (":" if sr else "  -- none (start/end self-collision-free)"))
    for d in sr[:top]:
        print(f"    {d['link_i']}[{d['i']}] <-> {d['link_j']}[{d['j']}]  overlap {d['penetration_mm']:.1f} mm")
    if len(sr) > top:
        print(f"    ... +{len(sr) - top} more")
    wr = out["world"]
    if out["world"] is not None:
        print(f"  WORLD/ESDF: {len(wr)} sphere(s) inside the depth voxels"
              + (":" if wr else "  -- none (or no collision world loaded)"))
        for d in wr[:top]:
            print(f"    {d['link']}[{d['i']}]  {d['depth_mm']:.1f} mm inside")
        if len(wr) > top:
            print(f"    ... +{len(wr) - top} more")
    lr = out["limits"]
    flagged = [d for d in lr if d["margin_deg"] < near_limit_deg]
    print(f"  JOINT LIMITS ({side} arm): closest {min(3, len(lr))}"
          + (f"  [{len(flagged)} within {near_limit_deg:g} deg!]" if flagged else ""))
    for d in lr[:3]:
        bang = "  <-- near limit" if d["margin_deg"] < near_limit_deg else ""
        print(f"    {d['joint']}: {d['margin_deg']:.1f} deg to bound "
              f"(q={d['q_deg']:.1f}, [{d['lo_deg']:.0f}, {d['hi_deg']:.0f}]){bang}")
    s = out["singularity"]
    near = "  <-- NEAR-SINGULAR" if s["sigma_min"] < sing_eps else ""
    print(f"  SINGULARITY ({side} wrist 6x7): sigma_min={s['sigma_min']:.4f}  "
          f"cond={s['cond']:.1f}  manip={s['manip']:.4f}{near}")

def diagnose_pose(planner, side: str, wrist_pose, world_points=None, start_q_repo14=None,
                  label: str = "pose", **kw):
    """Diagnose the CONFIG that reaches a target wrist pose (e.g. a pre-grasp) -- the END-state
    counterpart to diagnose() for 'Planning to approach pose failed'. plan_grasp rejects the
    pre-grasp but does NOT return its config, so we re-solve one on the MAIN planner, which has
    NO world (only the grasp planners are voxel-capable) -> a WORLD-IGNORING config for the pose,
    then diagnose() it against `world_points` (the ESDF). Interpreting the result:
      * reachable + WORLD non-empty -> the pre-grasp genuinely puts arm spheres in the ESDF
        (real clutter collision; that grasp's approach is blocked -- not a sphere-model issue).
      * reachable + WORLD empty     -> a collision-free pre-grasp config EXISTS; plan_grasp's
        goalset/seeds just didn't find it -> more num_ik_seeds/num_trajopt_seeds may fix it.
      * unreachable                 -> the pose is infeasible even ignoring the world (self-
        collision / no IK) -> the grasp candidate itself is bad, not the world.
    NOTE plan_to_pose returns ONE IK branch (elbow up/down); a WORLD hit doesn't prove EVERY
    branch collides, but a WORLD miss proves a free one exists. Returns the diagnose() dict
    (empty sections if unreachable) + {'reachable', 'config'}."""
    start = np.zeros(DOF) if start_q_repo14 is None else start_q_repo14
    empty = {"self": [], "world": [], "limits": [], "singularity": {},
             "offending_centers": np.zeros((0, 3), np.float32),
             "offending_radii": np.zeros((0,), np.float32)}
    try:
        traj = planner.plan_to_pose(start, side, wrist_pose)      # main planner: self-only, NO world
        cfg = traj.q[-1]
    except PlanningError as e:
        print(f"[diagnose_pose] '{label}' ({side}) UNREACHABLE ignoring the world "
              f"(self-collision / no IK) -> the POSE itself is infeasible, not a world issue\n"
              f"    {e}")
        return {**empty, "reachable": False, "config": None}
    print(f"[diagnose_pose] '{label}' ({side}) reached at "
          f"{np.round(np.asarray(wrist_pose.translation), 3)} (world-free IK); "
          f"checking that config vs the ESDF:")
    out = diagnose(planner, cfg, side, world_points=world_points, **kw)
    out["reachable"] = True
    out["config"] = np.asarray(cfg, float)
    return out

def _reach(planner, side, pose, start_q):
    """World-free config reaching `pose` (main planner, no ESDF), or None if plan_to_pose can't.
    NOTE trajopt: None folds no-IK + no-collision-free-path; for a short home->pose move it is
    IK-bound in practice."""
    try:
        return planner.plan_to_pose(start_q, side, pose).q[-1]
    except PlanningError:
        return None

def diagnose_candidates(planner, side, grasp_goals, pregrasp_goals, world_points=None,
                        k: int = 0, start_q_repo14=None, show: bool = True):
    """Sweep the top-`k` grasp candidates (k<=0 == ALL) to localize a plan_grasp approach failure
    across the goalset (plan_grasp picks ONE of K -- the top-confidence one being bad doesn't mean
    all are). Per candidate: is the GRASP pose reachable, is the PRE-GRASP reachable, and if so
    does that config penetrate the world ESDF? All reachability on the world-free main planner.
    The grasp-vs-pre-grasp comparison is the confound remover: a grasp reachable from home but its
    pre-grasp (a short back-off) not is a genuine reach/geometry boundary, not a path artifact.
    Reading the summary: some PRE-GRASP reachable AND world-free -> a collision-free approach
    EXISTS and plan_grasp didn't converge to it (raise solver.num_ik_seeds / num_trajopt_seeds --
    THE 'more resources' case); reachable pre-grasps ALL world-blocked -> real clutter (adjust
    approach / crop-or-declutter the ESDF); grasp reachable but NO pre-grasp -> the back-off leaves
    reach or hits a wrist limit (shrink approach_dist / change axis); no grasp reachable ->
    candidates are out of reach for this arm (bad grasps / wrong side). ~2 world-free plan solves
    per candidate (~0.4s each) -- a full sweep of a large goalset takes a minute+."""
    start = np.zeros(DOF) if start_q_repo14 is None else start_q_repo14
    n = min(len(grasp_goals), len(pregrasp_goals))
    if int(k) > 0:
        n = min(n, int(k))
    # world predicate: prefer the TRUTH-TEST (the grasp planner's own ESDF checker -- the exact
    # 'Start or End state in collision' gate) over the legacy geometric voxel-centre test, which
    # under-counts (no eta semantics) and produced the unverified '74/200 free' readings.
    truth = planner._cw_enabled and side in planner._cw_loaded
    print(f"[diagnose_candidates] sweeping {n} candidate(s) "
          f"(~{n * 0.8:.0f}s; 2 world-free plan solves each; world predicate: "
          f"{'cuRobo ESDF gate (truth)' if truth else 'geometric (no live ESDF loaded)'}) ...")
    rows = []
    for i in range(n):
        gcfg = _reach(planner, side, grasp_goals[i], start)
        pcfg = _reach(planner, side, pregrasp_goals[i], start)
        wh = None
        if pcfg is not None:
            if truth:
                wh = len(world_check(planner, side, pcfg, show=False)["violations"])
            elif world_points is not None and len(world_points):
                wh = len(diagnose(planner, pcfg, side, world_points=world_points, show=False)["world"])
        rows.append({"i": i, "grasp": gcfg is not None, "pregrasp": pcfg is not None,
                     "world_hits": wh})
        if n > 20 and (i + 1) % 20 == 0:
            print(f"    ... {i + 1}/{n}")
    s = {"n": n, "rows": rows,
         "grasp_reach": sum(r["grasp"] for r in rows),
         "pregrasp_reach": sum(r["pregrasp"] for r in rows),
         "pregrasp_free": sum(1 for r in rows if r["pregrasp"] and r["world_hits"] == 0),
         "pregrasp_blocked": sum(1 for r in rows if r["pregrasp"] and (r["world_hits"] or 0) > 0)}
    if show:
        _print_candidates(s, side)
    return s

def _print_candidates(s, side):
    rows = s["rows"]
    free = [r for r in rows if r["pregrasp"] and r["world_hits"] == 0]
    print(f"[diagnose_candidates] {side}: swept {s['n']} candidate(s) "
          f"(reachability on the world-free planner)")
    print(f"  summary: grasp-reachable {s['grasp_reach']}/{s['n']}, "
          f"pre-grasp-reachable {s['pregrasp_reach']}/{s['n']} "
          f"(world-free {s['pregrasp_free']}, ESDF-blocked {s['pregrasp_blocked']})")
    if free:                                      # the actionable ones: plan_grasp should hit these
        idx = ", ".join(str(r["i"]) for r in free[:20])
        print(f"  world-FREE reachable pre-grasp candidate idx: [{idx}"
              + (f", +{len(free) - 20} more" if len(free) > 20 else "") + "]")
    show_rows = rows if s["n"] <= 20 else (       # full list when small; else free + a sample
        free[:10] + [r for r in rows if not (r["pregrasp"] and r["world_hits"] == 0)][:10])
    for r in show_rows:
        wh = "n/a" if r["world_hits"] is None else (
            f"{r['world_hits']} in ESDF" if r["world_hits"] else "world-free")
        print(f"  cand {r['i']:3d}: grasp {'OK ' if r['grasp'] else 'NO '} | "
              f"pre-grasp {'OK ' if r['pregrasp'] else 'NO '} | {wh}")
    if s["n"] > 20:
        print(f"  (showing {len(show_rows)} of {s['n']}; summary above is the full count)")
    if s["pregrasp_free"] > 0:
        print("  => a reachable, world-FREE pre-grasp EXISTS. plan_grasp commits to ONE goalset "
              "winner (no internal retry) -> the candidate-retry loop should reach these; if it "
              "still fails, world_check the START config (a start in the ESDF fails EVERY plan) "
              "and check collision_world.exclude_object.")
    elif s["pregrasp_reach"] > 0 and s["pregrasp_blocked"] == s["pregrasp_reach"]:
        print("  => every reachable pre-grasp is INSIDE the ESDF -> real clutter collision: "
              "adjust the approach (offset/axis), crop the ESDF, or de-clutter.")
    elif s["grasp_reach"] > 0 and s["pregrasp_reach"] == 0:
        print("  => grasps reachable but NO pre-grasp is -> the back-off leaves the arm's reach "
              "or hits a wrist limit: shrink approach_dist or change approach axis. "
              "(If EVERY pre-grasp fails while grasps pass, also sanity-check the approach "
              "reconstruction direction.)")
    elif s["grasp_reach"] == 0:
        print("  => no grasp pose is even reachable for this arm -> candidates are out of reach "
              "(bad grasps / wrong side / object out of the arm's workspace).")

def world_check(planner, side: str, q_repo14, activation_distance: float = 0.0,
                near_distance: Optional[float] = None, disable_links: Optional[List[str]] = None,
                top: int = 8, show: bool = True):
    """TRUTH-TEST a config against the GRASP PLANNER'S OWN loaded ESDF world -- the exact same
    checker + predicate that rejects plan_grasp with 'Start or End state in collision'. Unlike
    diagnose()'s geometric sphere-vs-occupied-voxel-centre approximation, this queries cuRobo's
    scene_collision_checker: per-sphere cost = (radius + eta) - esdf(center), > 0 == collision.
    The graph-planner/IK feasibility gate runs at eta = ``activation_distance`` = 0.0 (cuRobo
    curobo/content/configs/task/metrics_base.yml: scene_collision activation_distance 0.0) --
    planner.yaml's solver.collision_activation_distance shapes only the OPTIMIZER cost. A second
    query at ``near_distance`` (default: that optimizer eta) reports the spheres the optimizer is
    actively pushing on. ``disable_links`` mimics plan_grasp Step-1/3 (spheres zeroed for those
    links); Step-2 (the approach plan -- where the failures happen) runs with ALL links enabled,
    the default here. Returns {violations, near, in_collision, max_penetration_mm,
    offending_centers/radii, world_loaded, n_spheres}; empty + world_loaded=False when this
    side's grasp planner has no LIVE ESDF loaded (never trust the empty build grid)."""
    mp = planner._grasp_planner(side)
    out = {"violations": [], "near": [], "in_collision": False, "max_penetration_mm": 0.0,
           "offending_centers": np.zeros((0, 3), np.float32),
           "offending_radii": np.zeros((0,), np.float32),
           "world_loaded": side in planner._cw_loaded, "n_spheres": 0}
    if mp.scene_collision_checker is None or not out["world_loaded"]:
        if show:
            print(f"[world_check] {side}: NO live ESDF loaded in this side's grasp planner "
                  f"(collision world {'enabled but not built yet' if planner._cw_enabled else 'OFF'})"
                  f" -- nothing to check against.")
        return out
    assert set(mp.joint_names) == set(planner.active), "grasp planner joints != main planner set"
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer
    if disable_links:
        mp.disable_link_collision(list(disable_links))
    try:
        # NOTE: the grasp planner's joint ORDER differs from the main planner's, and low-level
        # compute_kinematics does NOT reorder by name -> build the state in ITS order.
        ks = mp.compute_kinematics(planner._joint_state(q_repo14, names=mp.joint_names))
    finally:
        if disable_links:
            mp.enable_link_collision(list(disable_links))
    sph = ks.robot_spheres
    out["n_spheres"] = int(sph.shape[-2])

    def _query(eta: float):
        buf = CollisionBuffer.from_shape(sph.shape, mp.device_cfg)
        d = mp.scene_collision_checker.get_sphere_distance(
            ks, buf, mp.device_cfg.to_device([1.0]), mp.device_cfg.to_device([float(eta)]))
        return d.detach().float().cpu().numpy().reshape(-1)      # (N,) cost; >0 == collision

    links = planner._sphere_link_names(mp)                          # THIS planner's sphere layout
    arr = sph.detach().float().cpu().numpy().reshape(-1, 4)
    radii = arr[:, 3]
    cost0 = _query(activation_distance)                          # the FAILURE predicate (eta 0)
    near_eta = (float((planner.planner_cfg.get("solver") or {}).get(
        "collision_activation_distance", 0.01)) if near_distance is None else float(near_distance))
    cost_n = _query(near_eta) if near_eta > activation_distance else cost0

    def _rows(cost):
        idx = np.nonzero((cost > 0.0) & (radii > 0.0))[0]        # r<=0 = disabled spheres
        rows = [{"i": int(i), "link": links[i], "penetration_mm": float(cost[i] * 1000.0)}
                for i in idx]
        rows.sort(key=lambda r: -r["penetration_mm"])
        return rows

    out["violations"] = _rows(cost0)
    hit = {r["i"] for r in out["violations"]}
    out["near"] = [r for r in _rows(cost_n) if r["i"] not in hit]
    out["in_collision"] = bool(out["violations"])
    out["max_penetration_mm"] = out["violations"][0]["penetration_mm"] if out["violations"] else 0.0
    off = sorted(hit)
    out["offending_centers"] = arr[off, :3].astype(np.float32) if off else out["offending_centers"]
    out["offending_radii"] = radii[off].astype(np.float32) if off else out["offending_radii"]
    if show:
        _print_world_check(out, side, activation_distance, near_eta, top)
    return out

def _print_world_check(out, side, eta, near_eta, top):
    v, n = out["violations"], out["near"]
    print(f"[world_check] {side}: cuRobo's OWN ESDF query ({out['n_spheres']} spheres, "
          f"gate eta={eta:g})")
    print(f"  IN COLLISION (the 'Start or End state in collision' predicate): {len(v)} sphere(s)"
          + (":" if v else "  -- config PASSES the real gate"))
    for r in v[:top]:
        print(f"    {r['link']}[{r['i']}]  {r['penetration_mm']:.1f} mm inside")
    if len(v) > top:
        print(f"    ... +{len(v) - top} more")
    if n:
        print(f"  NEAR (within optimizer eta={near_eta:g}, pushed but not gate-failing): "
              f"{len(n)} sphere(s)")
        for r in n[:min(top, 4)]:
            print(f"    {r['link']}[{r['i']}]  {r['penetration_mm']:.1f} mm into the margin")
        if len(n) > min(top, 4):
            print(f"    ... +{len(n) - min(top, 4)} more")

def explain_failure(planner, side: str, q_fail, candidates=None, grasp_cfg=None,
                    viz=None, k: int = 0, collision_world: bool = False):
    """The full grasp-failure post-mortem (the old 09_graspgen --diagnose except-block,
    reusable): diagnose the START config, TRUTH-TEST it against the live ESDF gate when the
    collision world is on, then sweep every candidate's grasp/pre-grasp reachability.

    ``candidates`` are GraspCandidate-likes (``.wrist_goal`` + optional ``.grasp_pose``);
    ``grasp_cfg`` is the planner.yaml ``grasp`` block (for the first strategy's approach
    offset); ``viz`` an optional GraspViz for the offending-sphere overlays (magenta =
    geometric START offenders, red = the real gate's failures). Never raises: diagnostics
    must not mask the original failure."""
    try:
        wpts = planner.collision_world_points()
        print("--- diagnose: START config (current) ---")
        out = diagnose(planner, q_fail, side, world_points=wpts)
        if viz is not None and len(out["offending_centers"]):      # magenta = start offenders
            viz.show_collision_spheres(out["offending_centers"], out["offending_radii"],
                                       color=[255, 0, 255], name="diag_offenders")
        if collision_world:              # TRUTH-TEST: the planner's OWN ESDF gate at start --
            wc = world_check(planner, side, q_fail)  # the real 'Start or End' predicate
            if viz is not None and len(wc["offending_centers"]):   # red = gate-failing spheres
                viz.show_collision_spheres(wc["offending_centers"], wc["offending_radii"],
                                           color=[255, 60, 60], name="diag_world_gate")
        cands = list(candidates or [])
        if cands:                        # END: sweep the candidates' grasp + pre-grasp
            gp = grasp_cfg or {}
            strat = (gp.get("strategies") or [{}])[0]
            dist = abs(float(strat.get("approach_offset", -0.10)))
            grasp_goals = [c.wrist_goal for c in cands]
            pregrasp_goals = [_approach_pose(c.wrist_goal, c.grasp_pose, dist) for c in cands]
            print(f"--- diagnose: END configs (top candidates, pre-grasp back-off {dist:.2f}m) ---")
            diagnose_candidates(planner, side, grasp_goals, pregrasp_goals,
                                world_points=wpts, k=k, start_q_repo14=q_fail)
    except Exception as de:              # noqa: BLE001 - diagnostics must never mask the abort
        print(f"diagnose failed: {de}")


def _approach_pose(grasp_wrist, grasp_pose, dist: float):
    """Pre-grasp reconstruction: back off ``dist`` along the grasp APPROACH axis (grasp +Z =
    into the object) when the 6-DoF grasp frame is known, else straight up (defensive).
    Identical to the tool-frame offset plan_grasp applies (tests/test_planner_offsets.py)."""
    axis = grasp_pose.rotation[:, 2] if grasp_pose is not None else np.array([0.0, 0.0, -1.0])
    p = grasp_wrist.copy()
    p.translation = grasp_wrist.translation - dist * np.asarray(axis, float)
    return p
