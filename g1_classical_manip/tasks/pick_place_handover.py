"""Pick-place and pick->handover->place task graphs, composed from primitives.

Pick-place FSM (G1_CLASSICAL_MANIP_PLAN.md sec 5):
  HOME -> PERCEIVE -> PREGRASP(hover) -> REPERCEIVE -> DESCEND -> GRASP(verify)
       -> LIFT -> TRANSPORT -> RELEASE -> RETREAT -> HOME -> DONE
Failure edges: perception miss / grasp-verify fail loop back to PERCEIVE, or
ABORT-to-home after k retries.

The state handlers delegate motion to tasks.primitives; perception reads come
from robot.perception. Transitions are logged via the FSM on_transition hook.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pinocchio as pin

from g1_classical_manip.tasks.fsm import FSM, DONE, ABORT
from g1_classical_manip.tasks import primitives as P
from g1_classical_manip.perception.transforms import from_xyz_rpy


@dataclass
class PickPlaceCtx:
    robot: object
    side: str
    tag_id: int = 0
    block_pose: Optional[pin.SE3] = None
    place_pose: Optional[pin.SE3] = None
    perceive_tries: int = 0
    grasp_tries: int = 0
    last: P.StepResult = field(default_factory=lambda: P.StepResult(True))


def _perceive(ctx) -> Optional[pin.SE3]:
    return ctx.robot.perception.block_pose(ctx.tag_id) if ctx.robot.perception else None


def build_pick_place_fsm(on_transition=None) -> FSM:
    fsm = FSM(on_transition=on_transition)
    cfg_key = "pick_place"

    @fsm.state("HOME")
    def home(ctx):
        ctx.last = P.move_to_home(ctx.robot)
        return "PERCEIVE" if ctx.last.ok else ABORT

    @fsm.state("PERCEIVE")
    def perceive(ctx):
        r = ctx.robot.cfg[cfg_key]["retries"]
        pose = _perceive(ctx)
        if pose is None:
            ctx.perceive_tries += 1
            return "PERCEIVE" if ctx.perceive_tries <= r["max_perception"] else ABORT
        ctx.block_pose = pose
        return "PREGRASP"

    @fsm.state("PREGRASP")
    def pregrasp(ctx):
        g = ctx.robot.cfg[cfg_key]["grasp"]
        ctx.last = P._exec(ctx.robot, P.plan_reach(
            ctx.robot, ctx.block_pose, ctx.side, g["hover_offset_z"]))
        return "REPERCEIVE" if ctx.last.ok else "PERCEIVE"

    @fsm.state("REPERCEIVE")
    def reperceive(ctx):
        pose = _perceive(ctx)
        if pose is not None:
            ctx.block_pose = pose   # correction with the closer view
        return "DESCEND"

    @fsm.state("DESCEND")
    def descend(ctx):
        g = ctx.robot.cfg[cfg_key]["grasp"]
        ctx.last = P._exec(ctx.robot, P.plan_reach(
            ctx.robot, ctx.block_pose, ctx.side, g["descend_clearance"]))
        return "GRASP" if ctx.last.ok else "PERCEIVE"

    @fsm.state("GRASP")
    def grasp(ctx):
        r = ctx.robot.cfg[cfg_key]["retries"]
        if ctx.robot.hand.close(ctx.side, verify=True):
            return "LIFT"
        ctx.robot.hand.open(ctx.side, verify=False)
        ctx.grasp_tries += 1
        return "PERCEIVE" if ctx.grasp_tries <= r["max_grasp"] else ABORT

    @fsm.state("LIFT")
    def lift(ctx):
        ctx.last = P._exec(ctx.robot, P.plan_pick_lift(ctx.robot, ctx.block_pose, ctx.side))
        return "TRANSPORT" if ctx.last.ok else ABORT

    @fsm.state("TRANSPORT")
    def transport(ctx):
        ctx.last = P._exec(ctx.robot, P.plan_place(ctx.robot, ctx.place_pose, ctx.side))
        return "RELEASE" if ctx.last.ok else ABORT

    @fsm.state("RELEASE")
    def release(ctx):
        ctx.robot.hand.open(ctx.side, verify=True)
        return "RETREAT"

    @fsm.state("RETREAT")
    def retreat(ctx):
        ctx.last = P.move_to_home(ctx.robot)
        return DONE

    return fsm


def run_pick_place(robot, on_transition=None) -> str:
    cfg = robot.cfg["pick_place"]
    side = cfg.get("arm", "left")
    place_pose = from_xyz_rpy(cfg["place"]["xyz"], cfg["place"]["rpy"])
    ctx = PickPlaceCtx(robot=robot, side=side, place_pose=place_pose)
    fsm = build_pick_place_fsm(on_transition=on_transition)
    return fsm.run("HOME", ctx)


def run_pick_handover_place(robot, on_transition=None) -> str:
    """Pick with the giver arm, hand to the receiver, place with the receiver."""
    pp = robot.cfg["pick_place"]
    ho = robot.cfg["handover"]
    giver = ho.get("giver", "left")
    receiver = P.OTHER[giver]
    place_pose = from_xyz_rpy(pp["place"]["xyz"], pp["place"]["rpy"])

    ctx = PickPlaceCtx(robot=robot, side=giver, place_pose=place_pose)
    fsm = build_pick_place_fsm(on_transition=on_transition)

    # reuse pick states up to LIFT, then branch into handover + receiver place
    @fsm.state("LIFT")
    def lift(c):
        c.last = P._exec(c.robot, P.plan_pick_lift(c.robot, c.block_pose, c.side))
        return "HANDOVER" if c.last.ok else ABORT

    @fsm.state("HANDOVER")
    def handover(c):
        c.last = P.handover(c.robot, giver=giver, receiver=receiver)
        return "PLACE_RECV" if c.last.ok else ABORT

    @fsm.state("PLACE_RECV")
    def place_recv(c):
        c.last = P.place(c.robot, c.place_pose, receiver)
        return "RETREAT" if c.last.ok else ABORT

    return fsm.run("HOME", ctx)
