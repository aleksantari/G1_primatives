"""Action primitives — the LLM-agent tool surface.

Three thin, composable verbs over the cuRobo planner + DDS hand control:
    move(robot, side, goal_pose)   — wrist of `side` to goal_pose (pelvis frame)
    open_hand(robot, side)
    close_hand(robot, side)
Each returns a Result(ok, info). Tasks are built by composing these. Simple by
design — keep new behavior out of here unless it's genuinely a new primitive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g1_classical_manip.spatial.pose import Pose
from g1_classical_manip.ee.hand_base import LEFT, RIGHT  # noqa: F401 (re-export sides)


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


def open_hand(robot, side: str, verify: bool = False) -> Result:
    ok = robot.hand.open(side, verify=verify)
    return Result(bool(ok) if verify else True,
                  "opened" if ok else ("open failed" if verify else "open commanded"))


def close_hand(robot, side: str, verify: bool = False) -> Result:
    grasped = robot.hand.close(side, verify=verify)
    return Result(bool(grasped) if verify else True,
                  "grasped" if grasped else ("no grasp" if verify else "close commanded"))
