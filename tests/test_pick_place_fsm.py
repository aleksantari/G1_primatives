"""Offline FSM integration: real planning (IK + planner + retimer) with fake
execution / perception / hand, driving the pick-place graph to DONE and ABORT."""
import numpy as np

from g1_classical_manip.perception.transforms import from_xyz_rpy
from g1_classical_manip.tasks.pick_place_handover import run_pick_place
from g1_classical_manip.tasks import primitives as P


class _Res:
    success = True
    reason = "ok"


class _Exec:
    def run(self, traj):
        return _Res()


class _Hand:
    def __init__(self, grasp=True):
        self._grasp = grasp

    def close(self, side, verify=True):
        return self._grasp

    def open(self, side, verify=True):
        return True

    def preset(self, side, name):
        pass


class _Perc:
    def __init__(self, pose):
        self.pose = pose

    def block_pose(self, tag):
        return self.pose


def _wire(robot, grasp=True):
    block = from_xyz_rpy([0.34, 0.16, 0.06], [0, 0, 0.0])
    robot.connected = True
    robot.executor = _Exec()
    robot.hand = _Hand(grasp=grasp)
    robot.perception = _Perc(block)
    return block


def test_pick_place_reaches_done(robot):
    _wire(robot, grasp=True)
    trace = []
    end = run_pick_place(robot, on_transition=lambda s, c: trace.append(s))
    assert end == "DONE"
    for st in ("PERCEIVE", "PREGRASP", "DESCEND", "GRASP", "LIFT", "RELEASE"):
        assert st in trace


def test_grasp_failure_aborts_after_retries(robot):
    _wire(robot, grasp=False)
    trace = []
    end = run_pick_place(robot, on_transition=lambda s, c: trace.append(s))
    assert end == "ABORT"
    assert trace.count("PERCEIVE") >= 2     # retried perception/grasp
