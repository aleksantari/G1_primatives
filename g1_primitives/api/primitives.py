"""Action + perception primitives — the LLM-agent tool surface (implementation).

Thin, composable verbs over the cuRobo planner + DDS hand control + head-camera
perception:
    home(robot)
    move(robot, side, goal_pose)   — wrist of `side` to goal_pose (pelvis frame)
    grasp_motion(robot, side, candidates, options)  — the full native cuRobo pick
    open_hand(robot, side) / close_hand(robot, side)
    detect(robot, target)          — object pose (pelvis frame)
Action verbs return a Result(ok, info); grasp_motion returns a GraspResult with a
serializable GraspReport; detect returns a Detection (its `.pose` is a pelvis-frame
Pose, so `move(robot, side, detect(robot).pose)` composes). Callers normally reach
these through the Robot facade methods (api.robot); tasks are built by composing them.
Simple by design — keep new behavior out of here unless it's genuinely a new primitive.
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Optional

import numpy as np

from g1_primitives.spatial.pose import Pose
from g1_primitives.ee.hand_base import LEFT, RIGHT  # noqa: F401 (re-export sides)
from g1_primitives.perception.base import Detection  # noqa: F401 (re-export)
from g1_primitives.api.results import Result, GraspResult, GraspReport, PHASES
from g1_primitives.api.options import GraspOptions, GraspObserver
from g1_primitives.latency import LOG   # latency instrumentation (no-op unless enabled)


def home(robot) -> Result:
    """Move both arms to the configured home pose (the sim launch pose: arm joints
    at 0 = forearms forward) and settle there. The canonical reset/ready start."""
    q_home = np.deg2rad(robot.cfg["robot"]["home_q14_deg"])
    res = robot.executor.run(robot.planner.plan_joint(
        robot.arm.get_current_dual_arm_q(), q_home))
    err = robot.executor.settle(q_home)
    return Result(bool(res.success) and err < 0.1,
                  f"home (settle residual {np.rad2deg(err):.1f} deg)")


def move(robot, side: str, goal_pose: Pose) -> Result:
    """Move `side` wrist to goal_pose (pelvis frame); the other arm holds."""
    q0 = robot.arm.get_current_dual_arm_q()
    traj = robot.planner.plan_to_pose(q0, side, goal_pose)
    res = robot.executor.run(traj)
    return Result(bool(res.success), res.reason)


def _update_collision_world(robot, side: str, q0=None) -> None:
    """Build the depth-ESDF collision world from the head camera before a grasp plan. Gated:
    a no-op unless the planner has the collision world enabled AND a head depth frame is
    available. SOURCE-INDEPENDENT -- it uses the head depth, so it works with any grasp source
    (graspgenx / sim_cloud). `q0` (current arm config) self-filters the robot out of
    the depth. Best-effort: a perception hiccup never blocks the grasp (the planner just falls
    back to self-collision-only)."""
    planner = robot.planner
    if not getattr(planner, "collision_world_enabled", False):
        return
    cam = getattr(robot, "camera", None)
    if cam is None or not cam.has_depth:
        return
    depth = None
    for _ in range(20):                          # CONFLATE socket: wait briefly for a fresh frame
        depth = cam.get_depth_frame()
        if depth is not None:
            break
        time.sleep(0.05)
    if depth is None:
        print("grasp_motion: collision world skipped (no head depth frame)")
        return
    try:
        K = robot.cfg["camera"]["intrinsics"]
        T_pc = robot.frames.T_pelvis_camera(None)          # head cam is q-independent (locked torso)
        hand = getattr(robot, "hand", None)                # live finger angles -> self-filter the hand
        hand_q = {s: hand.get_q(s) for s in (LEFT, RIGHT)} if hasattr(hand, "get_q") else None
        # The TARGET's SAM3 mask (graspgenx source; None elsewhere) -> cut the object out of the
        # ESDF so the approach can reach it (gated by collision_world.exclude_object). Read from
        # the source's typed SourceSnapshot (grasp.base), not an ad-hoc attribute.
        snap = getattr(getattr(robot, "grasp_source", None), "last_snapshot", None)
        object_mask = snap.mask if snap is not None else None
        if planner.update_grasp_world(side, depth, K, T_pc, q0, hand_q=hand_q,
                                      object_mask=object_mask):
            print("grasp_motion: collision world updated from head depth (ESDF, robot self-filtered)")
    except Exception as e:                        # noqa: BLE001 - best-effort; never block the grasp
        print(f"grasp_motion: collision world skipped: {e}")


def _shift_z(pose: Pose, dz: float) -> Pose:
    """Copy a wrist goal Pose shifted by `dz` along world +z (pelvis frame)."""
    p = pose.copy()
    p.translation = pose.translation + np.array([0.0, 0.0, dz])
    return p


def _report_from_outcome(out, n_candidates: int) -> GraspReport:
    """Map the planner's GraspPlanOutcome onto the serializable GraspReport. Motion-phase
    entries start from the PLANNING result ('pending' = a planned segment not yet executed,
    'skipped' = not part of this plan, 'failed' = the planner rejected it); execution then
    overwrites them with ok/failed."""
    phases = {}
    for label, seg, planned_ok in (("approach", out.approach, out.approach_success),
                                   ("grasp", out.grasp, out.grasp_success),
                                   ("lift", out.lift, out.lift_success)):
        phases[label] = ("pending" if seg is not None
                         else ("skipped" if planned_ok or out.success else "failed"))
    phases["close"] = "pending"
    failed = None
    if not out.success:
        failed = next((lbl for lbl, ok in (("approach", out.approach_success),
                                           ("grasp", out.grasp_success),
                                           ("lift", out.lift_success)) if not ok), None)
    return GraspReport(chosen_index=int(getattr(out, "chosen_index", -1)),
                       n_candidates=int(n_candidates),
                       phases=phases, failed_phase=failed,
                       planner_status=str(out.status or ""),
                       strategy=getattr(out, "strategy", None))


def grasp_motion(robot, side: str, candidates,
                 options: Optional[GraspOptions] = None) -> GraspResult:
    """Native cuRobo plan_grasp over ranked grasp candidates: solve a K-goalset (cuRobo picks
    the feasible grasp) sweeping the configured approach/lift offsets, then execute
    approach -> grasp -> close -> lift per ``options`` (api.options.GraspOptions; None = the
    full autonomous defaults). ``options.confirm(label)`` gates each phase for operator runs;
    ``options.observer`` receives viz/diagnostics hooks (world built / winner selected / phase
    done). Reads the ``planner.grasp`` config block. Returns a GraspResult whose ``report``
    is the serializable per-phase summary."""
    opts = options or GraspOptions()
    obs = opts.observer or GraspObserver()
    cands = list(candidates)
    if opts.max_candidates is not None:
        cands = cands[:max(0, int(opts.max_candidates))]
    if not cands:
        return GraspResult(False, "no candidates")
    if opts.grasp_z_offset:
        cands = [replace(c, wrist_goal=_shift_z(c.wrist_goal, opts.grasp_z_offset))
                 for c in cands]
    gp = (robot.cfg["planner"].get("grasp") or {})
    strategies = [dict(s) for s in gp.get("strategies",
                                          [{"approach_offset": -0.10, "lift_offset": 0.10}])]
    if not opts.approach:      # straight to the grasp pose (frame sanity / constrained scenes)
        strategies = [{**s, "approach_offset": 0.0, "plan_approach": False} for s in strategies]
    if not opts.lift:
        strategies = [{**s, "plan_lift": False} for s in strategies]

    q0 = robot.arm.get_current_dual_arm_q()
    with LOG.span("collision_world"):           # depth-ESDF world (gated), self-filtered at q0
        _update_collision_world(robot, side, q0)
    if opts.observer is not None:               # let callers overlay the ESDF (even on plan failure)
        obs.on_world_built(robot.planner.collision_world_points())
    with LOG.span("plan_grasp"):                # cuRobo native goalset + approach/grasp/lift solve
        out = robot.planner.plan_grasp_set_sweep(
            q0, side, [c.wrist_goal for c in cands],
            strategies=strategies,
            approach_axis=gp.get("approach_axis", "y"), lift_axis=gp.get("lift_axis", "z"),
            approach_in_tool_frame=gp.get("approach_in_tool_frame", True),
            lift_in_tool_frame=gp.get("lift_in_tool_frame", False),
            hold_idle=gp.get("hold_idle_arm", True),
            disable_collision_links=gp.get("disable_collision_links"),
            max_candidate_retries=gp.get("max_candidate_retries", 12))
    report = _report_from_outcome(out, len(cands))
    if not out.success:
        # Surface WHICH phase cuRobo rejected + its status string (the raw plan_grasp diagnostic).
        # For the per-config breakdown run motion.diagnostics.explain_failure.
        print(f"grasp_motion: plan_grasp FAILED -- approach={out.approach_success} "
              f"grasp={out.grasp_success} lift={out.lift_success} | status: {out.status or '(none)'}")
        return GraspResult(False, f"plan_grasp failed: {out.status}", report)

    chosen = cands[out.chosen_index] if 0 <= out.chosen_index < len(cands) else None
    if chosen is not None:
        obs.on_selected(chosen, report)

    def _phase(label: str, ok: bool, info: str):
        report.phases[label] = "ok" if ok else "failed"
        obs.on_phase(label, ok, info)
        if not ok:
            report.failed_phase = label

    for label, traj in (("approach", out.approach), ("grasp", out.grasp)):
        if traj is None:
            continue
        if opts.confirm is not None:
            opts.confirm(label)
        with LOG.span(f"exec:{label}", LOG.EXEC):
            r = robot.executor.run(traj)
        _phase(label, bool(r.success), r.reason)
        if not r.success:
            return GraspResult(False, f"{label}: {r.reason}", report)

    hand = getattr(robot, "hand", None)
    if opts.close_hand and hand is not None:
        # Settle at the grasp pose first: the fingers close on a converged pose, and the droop
        # accumulated during the (multi-second) close won't trip the lift prime's abort.
        grasp_traj = out.grasp if out.grasp is not None else out.approach
        if grasp_traj is not None:
            robot.executor.settle(grasp_traj.q[-1])
        if opts.confirm is not None:
            opts.confirm("close")
        with LOG.span("hand:close", LOG.EXEC):
            grasped = hand.close(side, verify=opts.verify_close, fraction=opts.close_fraction)
        closed_ok = bool(grasped) if opts.verify_close else True
        _phase("close", closed_ok, "grasped" if grasped else
               ("no grasp" if opts.verify_close else "close commanded (unverified)"))
        if not closed_ok:
            return GraspResult(False, "close: no grasp detected", report)
    else:
        report.phases["close"] = "skipped"

    if out.lift is not None:
        if opts.confirm is not None:
            opts.confirm("lift")
        with LOG.span("exec:lift", LOG.EXEC):
            r = robot.executor.run(out.lift)
        _phase("lift", bool(r.success), r.reason)
        if not r.success:
            return GraspResult(False, f"lift: {r.reason}", report)

    return GraspResult(True, "ok", report)


def detect(robot, target: str = "block", frames: int = 5) -> Optional[Detection]:
    """Detect `target` from the head camera; return a Detection whose `.pose` is the
    object pose in the pelvis frame (or None if not found). Pulls up to `frames` head
    images to fill the detector's filter; detectors with no camera (sim_state) ignore
    the frames. Compose with move: `move(robot, side, det.pose)`."""
    arm = getattr(robot, "arm", None)
    q14 = arm.get_current_dual_arm_q() if arm is not None else None
    cam = getattr(robot, "camera", None)
    got, attempts, last = 0, 0, None
    while got < max(1, frames) and attempts < max(1, frames) * 4:
        attempts += 1
        # RGB-first: detectors get RGB; future RGB perception primitives consume
        # get_rgb_frame() the same way.
        rgb = cam.get_rgb_frame() if cam is not None else None
        if cam is not None and rgb is None:
            continue                                   # frame not ready; retry
        last = robot.detector.detect(rgb, q14).get(target, last)
        got += 1
    pose = robot.detector.block_pose(target)
    if pose is None:
        return None
    return Detection(pose=pose, label=target,
                     score=last.score if last else 1.0,
                     tag_id=last.tag_id if last else None)


def open_hand(robot, side: str, verify: bool = False) -> Result:
    ok = robot.hand.open(side, verify=verify)
    return Result(bool(ok) if verify else True,
                  "opened" if ok else
                  ("open failed" if verify else "open commanded (unverified)"))


def close_hand(robot, side: str, verify: bool = False, fraction: float = 1.0) -> Result:
    grasped = robot.hand.close(side, verify=verify, fraction=fraction)
    return Result(bool(grasped) if verify else True,
                  "grasped" if grasped else
                  ("no grasp" if verify else "close commanded (unverified)"))
