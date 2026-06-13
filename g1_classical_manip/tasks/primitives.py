"""Task primitives: move / pick / place / handover.

Planning (build waypoints -> JointPath -> JointTrajectory) is separated from
execution (stream via the executor) so the planning paths run offline (no robot).
All EE poses are pin.SE3 in the pelvis frame; the moving arm is named explicitly
and the other arm holds its current EE pose (CartesianWaypoint side = None),
never by slicing the 14-vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pinocchio as pin

from g1_classical_manip.perception.transforms import (
    from_xyz_rpy, top_down_grasp_offset)
from g1_classical_manip.motion.planner_base import (
    Goal, CartesianWaypoint, World, Box, JointPath, JointTrajectory, DOF)
from g1_classical_manip.motion.retimer import retime_from_config
from g1_classical_manip.ee.hand_base import LEFT, RIGHT

OTHER = {LEFT: RIGHT, RIGHT: LEFT}


# --------------------------------------------------------------- helpers
def home_q(robot) -> np.ndarray:
    return np.deg2rad(robot.cfg["pick_place"]["home_q14_deg"])


def current_q(robot, start_q=None) -> np.ndarray:
    if start_q is not None:
        return np.asarray(start_q, float).reshape(DOF)
    if robot.connected and robot.arm is not None:
        return robot.arm.get_current_dual_arm_q()
    return home_q(robot)


def world_from_cfg(robot) -> World:
    wb = robot.cfg["pick_place"].get("workspace_box")
    box = Box(np.asarray(wb["min"], float), np.asarray(wb["max"], float)) if wb else None
    return World(workspace_box=box)


def _arm_index(side: str) -> slice:
    return slice(0, 7) if side == LEFT else slice(7, 14)


def _waypoint(side: str, pose: pin.SE3, label: str) -> CartesianWaypoint:
    return CartesianWaypoint(left=pose if side == LEFT else None,
                             right=pose if side == RIGHT else None, label=label)


def grasp_target(robot, block_pose: pin.SE3, side: str, clearance: float,
                 q14=None) -> pin.SE3:
    """L_ee/R_ee IK target for a top-down grasp `clearance` above the block."""
    return robot.frames.grasp_ee_target(
        block_pose, side, top_down_grasp_offset(clearance), q14)


# --------------------------------------------------------------- planning
def plan_cartesian(robot, start_q: np.ndarray, waypoints: List[CartesianWaypoint],
                   world: Optional[World] = None) -> JointTrajectory:
    path = robot.planner.plan(start_q, Goal(waypoints), world or world_from_cfg(robot))
    return retime_from_config(path, robot.planner_cfg)


def plan_joint_move(robot, start_q: np.ndarray, goal_q: np.ndarray) -> JointTrajectory:
    """Direct joint-space move (e.g. to the named home pose)."""
    path = JointPath(np.vstack([np.asarray(start_q, float).reshape(DOF),
                                np.asarray(goal_q, float).reshape(DOF)]),
                     meta={"waypoint_index": [(1, "joint_goal")]})
    return retime_from_config(path, robot.planner_cfg)


def plan_reach(robot, block_pose: pin.SE3, side: str, clearance: float,
               start_q=None) -> JointTrajectory:
    """Single Cartesian move to `clearance` above the block (top-down)."""
    q0 = current_q(robot, start_q)
    tgt = grasp_target(robot, block_pose, side, clearance, q0)
    return plan_cartesian(robot, q0, [_waypoint(side, tgt, f"reach_{clearance:.3f}")])


def plan_pick_approach(robot, block_pose: pin.SE3, side: str,
                       start_q=None) -> JointTrajectory:
    g = robot.cfg["pick_place"]["grasp"]
    q0 = current_q(robot, start_q)
    hover = grasp_target(robot, block_pose, side, g["hover_offset_z"], q0)
    descend = grasp_target(robot, block_pose, side, g["descend_clearance"], q0)
    return plan_cartesian(robot, q0, [_waypoint(side, hover, "hover"),
                                      _waypoint(side, descend, "descend")])


def plan_pick_lift(robot, block_pose: pin.SE3, side: str,
                   start_q=None) -> JointTrajectory:
    g = robot.cfg["pick_place"]["grasp"]
    q0 = current_q(robot, start_q)
    lift = grasp_target(robot, block_pose, side,
                        g["hover_offset_z"] + g["lift_height"], q0)
    return plan_cartesian(robot, q0, [_waypoint(side, lift, "lift")])


def plan_place(robot, place_pose: pin.SE3, side: str, start_q=None) -> JointTrajectory:
    q0 = current_q(robot, start_q)
    g = robot.cfg["pick_place"]["grasp"]
    above = place_pose * pin.SE3(np.eye(3), np.array([0, 0, g["hover_offset_z"]]))
    return plan_cartesian(robot, q0, [_waypoint(side, above, "place_above"),
                                      _waypoint(side, place_pose, "place")])


def plan_handover(robot, start_q=None) -> JointTrajectory:
    """Both wrists to the configured handover pose pair in ONE IK problem."""
    hp = robot.cfg["handover"]["handover_pose"]
    q0 = current_q(robot, start_q)
    left = from_xyz_rpy(hp["left"]["xyz"], hp["left"]["rpy"])
    right = from_xyz_rpy(hp["right"]["xyz"], hp["right"]["rpy"])
    wp = CartesianWaypoint(left=left, right=right, label="handover")
    return plan_cartesian(robot, q0, [wp], world=World())  # torso-front, skip box


# --------------------------------------------------------------- execution
@dataclass
class StepResult:
    ok: bool
    info: str = ""


def _exec(robot, traj: JointTrajectory) -> StepResult:
    if not robot.connected or robot.executor is None:
        return StepResult(False, "not connected (offline)")
    res = robot.executor.run(traj)
    return StepResult(res.success, res.reason)


def move_to_home(robot, start_q=None) -> StepResult:
    return _exec(robot, plan_joint_move(robot, current_q(robot, start_q), home_q(robot)))


def pick(robot, block_pose: pin.SE3, side: str) -> StepResult:
    """HOME-context pick: open -> approach -> close+verify -> lift. Returns ok if
    the block is verified grasped (no lift on a failed grasp)."""
    robot.hand.open(side, verify=False)
    r = _exec(robot, plan_pick_approach(robot, block_pose, side))
    if not r.ok:
        return r
    grasped = robot.hand.close(side, verify=True)
    if not grasped:
        robot.hand.open(side, verify=False)
        return StepResult(False, "grasp verification failed")
    return _exec(robot, plan_pick_lift(robot, block_pose, side))


def place(robot, place_pose: pin.SE3, side: str) -> StepResult:
    r = _exec(robot, plan_place(robot, place_pose, side))
    if not r.ok:
        return r
    robot.hand.open(side, verify=True)
    return StepResult(True, "placed")


def handover(robot, giver: str, receiver: str) -> StepResult:
    """Move both wrists to the handover pair; right(receiver) close -> verify ->
    left(giver) open -> verify. Receiver-fail => giver does NOT open (no drops)."""
    r = _exec(robot, plan_handover(robot))
    if not r.ok:
        return r
    robot.hand.preset(receiver, "open") if hasattr(robot.hand, "preset") \
        else robot.hand.open(receiver, verify=False)
    received = robot.hand.close(receiver, verify=True)
    if not received:
        return StepResult(False, "receiver grasp failed -- giver still holding (no drop)")
    released = robot.hand.open(giver, verify=True)
    return StepResult(released, "handover complete" if released else "giver release failed")
