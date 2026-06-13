import numpy as np
import pytest

from g1_classical_manip.motion.planner_base import (
    Goal, CartesianWaypoint, World, Box)
from g1_classical_manip.motion.cartesian_planner import CartesianPlanner, PlanningError


def _shift(T, dxyz):
    M = T.copy(); M.translation = T.translation + np.array(dxyz, float); return M


def test_plan_continuity(ik, home_q):
    planner = CartesianPlanner(ik, continuity_jump_rad=0.35)
    TL, _ = ik.fk(home_q)
    wps = [CartesianWaypoint(left=_shift(TL, [0.05, 0, 0.10]), label="hover"),
           CartesianWaypoint(left=_shift(TL, [0.05, 0, 0.0]), label="descend"),
           CartesianWaypoint(left=TL, label="home")]
    path = planner.plan(home_q, Goal(wps))
    assert path.n > 10
    assert path.max_consecutive_jump() <= 0.35
    # right arm held: stays near home
    assert np.max(np.abs(path.q[:, 7:] - home_q[7:])) < 0.1


def test_workspace_box_rejects(ik, home_q):
    planner = CartesianPlanner(ik)
    box = World(workspace_box=Box(np.array([0.0, -0.3, 0.0]),
                                  np.array([0.4, 0.3, 0.4])))
    far = CartesianWaypoint(left=__import__("pinocchio").SE3(
        np.eye(3), np.array([2.0, 0.0, 0.0])), label="far")
    with pytest.raises(PlanningError):
        planner.plan(home_q, Goal([far]), box)


def test_single_arm_holds_other(ik, home_q):
    planner = CartesianPlanner(ik)
    TL, TR = ik.fk(home_q)
    wp = CartesianWaypoint(left=_shift(TL, [0.0, 0.0, 0.08]), label="up")
    path = planner.plan(home_q, Goal([wp]))
    # right EE should barely move (held)
    _, TRf = ik.fk(path.q[-1])
    assert np.linalg.norm(TRf.translation - TR.translation) < 0.05
