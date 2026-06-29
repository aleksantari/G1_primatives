"""Action + perception primitives — the LLM-agent tool surface.

Thin, composable verbs over the cuRobo planner + DDS hand control + head-camera
perception:
    home(robot)
    move(robot, side, goal_pose)   — wrist of `side` to goal_pose (pelvis frame)
    open_hand(robot, side) / close_hand(robot, side)
    detect(robot, target)          — head-camera object pose (pelvis frame)
Action verbs return a Result(ok, info); detect returns a Detection (its `.pose` is
a pelvis-frame Pose, so `move(robot, side, detect(robot).pose)` composes). Tasks are
built by composing these. Simple by design — keep new behavior out of here unless
it's genuinely a new primitive.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from g1_classical_manip.spatial.pose import Pose
from g1_classical_manip.ee.hand_base import LEFT, RIGHT  # noqa: F401 (re-export sides)
from g1_classical_manip.perception.base import Detection  # noqa: F401 (re-export)


@dataclass
class Result:
    ok: bool
    info: str = ""


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


def move_to_candidates(robot, side: str, goal_poses) -> Result:
    """Move `side` wrist to the FIRST reachable goal in a ranked list (e.g. grasp
    candidates, best-first); the other arm holds. The planner tries each in order and
    picks the first that solves -- so callers don't assume a single grasp."""
    q0 = robot.arm.get_current_dual_arm_q()
    traj = robot.planner.plan_to_pose_set(q0, side, goal_poses)
    res = robot.executor.run(traj)
    return Result(bool(res.success), res.reason)


@dataclass
class GraspResult:
    ok: bool
    info: str = ""
    chosen_index: int = -1
    outcome: object = None        # the planner GraspPlanOutcome (segments, per-phase flags)


def _update_collision_world(robot, side: str) -> None:
    """Build the depth-ESDF collision world from the head camera before a grasp plan. Gated:
    a no-op unless the planner has the collision world enabled AND a head depth frame is
    available. SOURCE-INDEPENDENT -- it uses the head depth, so it works with any grasp source
    (apriltag / graspgenx / sim_cloud). Best-effort: a perception hiccup never blocks the grasp
    (the planner just falls back to self-collision-only)."""
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
        if planner.update_grasp_world(side, depth, K, T_pc):
            print("grasp_motion: collision world updated from head depth (ESDF)")
    except Exception as e:                        # noqa: BLE001 - best-effort; never block the grasp
        print(f"grasp_motion: collision world skipped: {e}")


def grasp_motion(robot, side: str, candidates, close_cb=None, confirm_cb=None,
                 on_selected=None) -> GraspResult:
    """Native cuRobo plan_grasp over ranked grasp candidates: solve a K-goalset (cuRobo picks
    the feasible grasp) sweeping the configured approach/lift offsets, then execute
    approach -> grasp -> [close_cb] -> lift. `on_selected(chosen_candidate, outcome)` fires AFTER
    cuRobo picks but BEFORE any motion (so callers mark viz / FK-check / print the chosen grasp);
    `close_cb()` is the operator-gated hand close, run after settling at the grasp pose; `confirm_cb
    (label)` gates each segment. Reads the `planner.grasp` config block. Returns GraspResult."""
    cands = list(candidates)
    if not cands:
        return GraspResult(False, "no candidates")
    gp = (robot.cfg["planner"].get("grasp") or {})
    _update_collision_world(robot, side)        # depth-ESDF world (gated), before planning
    q0 = robot.arm.get_current_dual_arm_q()
    out = robot.planner.plan_grasp_set_sweep(
        q0, side, [c.wrist_goal for c in cands],
        strategies=gp.get("strategies", [{"approach_offset": -0.10, "lift_offset": 0.10}]),
        approach_axis=gp.get("approach_axis", "y"), lift_axis=gp.get("lift_axis", "z"),
        approach_in_tool_frame=gp.get("approach_in_tool_frame", True),
        lift_in_tool_frame=gp.get("lift_in_tool_frame", False),
        hold_idle=gp.get("hold_idle_arm", True),
        disable_collision_links=gp.get("disable_collision_links"))
    if not out.success:
        return GraspResult(False, f"plan_grasp failed: {out.status}", out.chosen_index, out)

    chosen = cands[out.chosen_index] if 0 <= out.chosen_index < len(cands) else None
    if on_selected is not None and chosen is not None:
        on_selected(chosen, out)

    for label, traj in (("approach", out.approach), ("grasp", out.grasp)):
        if traj is None:
            continue
        if confirm_cb is not None:
            confirm_cb(label)
        r = robot.executor.run(traj)
        if not r.success:
            return GraspResult(False, f"{label}: {r.reason}", out.chosen_index, out)

    if close_cb is not None:
        # Settle at the grasp pose first: the fingers close on a converged pose, and the droop
        # accumulated during the (multi-second) close won't trip the lift prime's abort.
        grasp_traj = out.grasp if out.grasp is not None else out.approach
        if grasp_traj is not None:
            robot.executor.settle(grasp_traj.q[-1])
        close_cb()

    if out.lift is not None:
        if confirm_cb is not None:
            confirm_cb("lift")
        r = robot.executor.run(out.lift)
        if not r.success:
            return GraspResult(False, f"lift: {r.reason}", out.chosen_index, out)

    return GraspResult(True, "ok", out.chosen_index, out)


def detect(robot, target: str = "block", frames: int = 5) -> Optional[Detection]:
    """Detect `target` from the head camera; return a Detection whose `.pose` is the
    object pose in the pelvis frame (or None if not found). Pulls up to `frames` head
    images to fill the detector's median filter; detectors with no camera
    (ground_truth) ignore the frames. Compose with move: `move(robot, side, det.pose)`."""
    arm = getattr(robot, "arm", None)
    q14 = arm.get_current_dual_arm_q() if arm is not None else None
    cam = getattr(robot, "camera", None)
    got, attempts, last = 0, 0, None
    while got < max(1, frames) and attempts < max(1, frames) * 4:
        attempts += 1
        # RGB-first: detectors get RGB (AprilTagDetector grayscales itself); future
        # RGB perception primitives consume get_rgb_frame() the same way.
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
                  "opened" if ok else ("open failed" if verify else "open commanded"))


def close_hand(robot, side: str, verify: bool = False, fraction: float = 1.0) -> Result:
    grasped = robot.hand.close(side, verify=verify, fraction=fraction)
    return Result(bool(grasped) if verify else True,
                  "grasped" if grasped else ("no grasp" if verify else "close commanded"))
