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
