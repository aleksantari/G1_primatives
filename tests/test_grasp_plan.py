"""Native cuRobo plan_grasp adoption: JointState->trajectory conversion, goalset build,
strategy sweep, and the grasp_motion primitive. All GPU-free -- the planner is built via
__new__ (skipping cuRobo init) and curobo.types / the trim op are monkeypatched in, exactly
like test_grasp_source.py fakes the planner."""
import sys
import types

import numpy as np
import pytest

from g1_classical_manip.spatial.pose import Pose
from g1_classical_manip.ee.hand_base import RIGHT
from g1_classical_manip.motion.curobo_planner import (
    CuroboArmPlanner, GraspPlanOutcome, PlanningError, REPO_ARM, WRIST_FRAME)

TOOL_FRAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link", "d435_link"]


# --------------------------------------------------------------- fakes
class FakeTensor:
    """numpy array masquerading as a torch tensor (.detach().cpu().numpy())."""
    def __init__(self, arr):
        self._a = np.asarray(arr, float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class FakeJS:
    def __init__(self, position, velocity=None, acceleration=None, dt=0.025,
                 joint_names=None):
        self.position = FakeTensor(position)
        self.velocity = FakeTensor(velocity) if velocity is not None else None
        self.acceleration = FakeTensor(acceleration) if acceleration is not None else None
        self.dt = dt
        self.joint_names = joint_names or list(reversed(REPO_ARM))


class FakeFlag:
    """A success tensor: .any() -> bool, not None."""
    def __init__(self, v):
        self.v = bool(v)

    def any(self):
        return self.v


class FakeIdx:
    """A goalset_index tensor: .view(-1)[0].item() -> int."""
    def __init__(self, v):
        self.v = int(v)

    def view(self, *a):
        return self

    def __getitem__(self, i):
        return self

    def item(self):
        return self.v


def _planner(mp):
    """A CuroboArmPlanner with cuRobo __init__ skipped; _grasp_planner returns the fake `mp`
    (the single-tool-frame grasp planner) and the helpers plan_grasp_set needs are stubbed."""
    p = CuroboArmPlanner.__new__(CuroboArmPlanner)
    p._max_goalset = 128
    p._cw_enabled = False                                              # collision world off by default
    p._joint_state = lambda q: "START"
    p._tensor_k = lambda arr: np.asarray(arr, float)                    # keep numpy for asserts
    p._jointstate_to_trajectory = lambda js, last, lbl, dt=None: f"traj:{lbl}"
    p._hold_idle = lambda traj, side, hold: traj                        # conversion is stubbed
    p._grasp_planner = lambda side: mp
    return p


def _inject_curobo_types(monkeypatch):
    """Monkeypatch `from curobo.types import GoalToolPose, Pose` (leaf-only; parents not
    imported because curobo.types is in sys.modules). Returns the recorder dicts."""
    rec = {"from_poses": None}

    class FakeCuPose:
        def __init__(self, position=None, quaternion=None):
            self.position = position
            self.quaternion = quaternion

    class FakeGoalToolPose:
        @staticmethod
        def from_poses(pose_dict, ordered_tool_frames=None, num_goalset=None):
            rec["from_poses"] = {"pose_dict": pose_dict, "ordered": ordered_tool_frames,
                                 "num_goalset": num_goalset}
            return "GOAL"

    mod = types.ModuleType("curobo.types")
    mod.GoalToolPose = FakeGoalToolPose
    mod.Pose = FakeCuPose
    monkeypatch.setitem(sys.modules, "curobo.types", mod)
    return rec


class FakeMP:
    def __init__(self, result):
        self.result = result
        self.kw = None

    def compute_kinematics(self, start):
        return "KS"

    def plan_grasp(self, goal, start, **kw):
        self.kw = kw
        self.goal, self.start = goal, start
        return self.result


def _grasp_result(idx=1, ok=True, approach=True, grasp=True, lift=True):
    r = types.SimpleNamespace()
    r.goalset_index = FakeIdx(idx)
    r.success = FakeFlag(ok)
    r.approach_success = FakeFlag(approach)
    r.grasp_success = FakeFlag(grasp)
    r.lift_success = FakeFlag(lift)
    r.approach_interpolated_trajectory = "AJ" if approach else None
    r.grasp_interpolated_trajectory = "GJ" if grasp else None
    r.lift_interpolated_trajectory = "LJ" if lift else None
    r.approach_interpolated_last_tstep = [5]
    r.grasp_interpolated_last_tstep = [5]
    r.lift_interpolated_last_tstep = [5]
    r.status = "Planning to lift pose succeeded."
    return r


# --------------------------------------------------------------- _jointstate_to_trajectory
def test_jointstate_to_trajectory_remaps_and_times():
    p = CuroboArmPlanner.__new__(CuroboArmPlanner)
    names = list(reversed(REPO_ARM))                 # cuRobo order = reversed repo order
    pos = np.tile(np.arange(14, dtype=float), (3, 1))  # row t: column k holds value k
    js = FakeJS(pos, dt=0.025, joint_names=names)
    traj = p._jointstate_to_trajectory(js, None, "seg")    # last_tstep None -> no trim
    assert traj.q.shape == (3, 14)
    # repo column i pulls cuRobo column names.index(REPO_ARM[i]) = 13 - i
    assert np.allclose(traj.q[0], [13 - i for i in range(14)])
    assert np.allclose(traj.t, [0.0, 0.025, 0.05])
    assert traj.meta["dt"] == 0.025 and traj.meta["duration"] == pytest.approx(0.05)
    assert np.allclose(traj.qd, 0.0) and np.allclose(traj.qdd, 0.0)   # None vel/acc -> zeros


def test_jointstate_to_trajectory_trims_to_last_tstep(monkeypatch):
    trims = {}

    def fake_trim(js, start, end):
        trims["args"] = (start, end)
        sliced = js.position.numpy()[:int(end)]
        return FakeJS(sliced, dt=js.dt, joint_names=js.joint_names)

    mod = types.ModuleType("curobo._src.state.state_joint_trajectory_ops")
    mod.trim_joint_state_trajectory = fake_trim
    monkeypatch.setitem(sys.modules, "curobo._src.state.state_joint_trajectory_ops", mod)

    p = CuroboArmPlanner.__new__(CuroboArmPlanner)
    pos = np.tile(np.arange(14, dtype=float), (8, 1))
    js = FakeJS(pos, dt=0.01, joint_names=list(reversed(REPO_ARM)))
    traj = p._jointstate_to_trajectory(js, [3], "seg")     # length-1 tstep -> trim(js, 0, 3)
    assert trims["args"] == (0, 3)
    assert traj.q.shape == (3, 14)                          # trimmed to 3 rows


# --------------------------------------------------------------- plan_grasp_set
def test_plan_grasp_set_outcome_and_kwargs(monkeypatch):
    _inject_curobo_types(monkeypatch)
    mp = FakeMP(_grasp_result(idx=2))
    p = _planner(mp)
    goals = [Pose(np.eye(3), [0.4, -0.2 + 0.01 * i, 0.8]) for i in range(4)]
    out = p.plan_grasp_set(np.zeros(14), RIGHT, goals,
                           approach_axis="x", approach_offset=-0.10,
                           lift_axis="z", lift_offset=0.15)
    assert isinstance(out, GraspPlanOutcome)
    assert out.chosen_index == 2 and out.success
    assert out.approach == "traj:right_approach" and out.grasp == "traj:right_grasp"
    assert out.lift == "traj:right_lift" and "succeeded" in out.status
    # the unsigned axis + signed offset + world-frame lift reach plan_grasp
    assert mp.kw["grasp_approach_axis"] == "x" and mp.kw["grasp_approach_offset"] == -0.10
    assert mp.kw["grasp_approach_in_tool_frame"] is True
    assert mp.kw["grasp_lift_axis"] == "z" and mp.kw["grasp_lift_offset"] == 0.15
    assert mp.kw["grasp_lift_in_tool_frame"] is False
    # config has grasp_contact_link_names: null -> we disable the active wrist link explicitly
    assert mp.kw["disable_collision_links"] == ["right_wrist_yaw_link"]


def test_plan_grasp_set_skips_unplanned_segments(monkeypatch):
    _inject_curobo_types(monkeypatch)
    mp = FakeMP(_grasp_result(idx=0, lift=False))     # grasp-at-goal / no lift
    out = _planner(mp).plan_grasp_set(
        np.zeros(14), RIGHT, [Pose.Identity()],
        approach_axis="x", approach_offset=-0.05, lift_axis="z", lift_offset=0.0,
        plan_lift=False)
    assert out.lift is None and out.lift_success is False
    assert out.approach == "traj:right_approach" and out.grasp == "traj:right_grasp"


def test_plan_grasp_set_goalset_single_active_link(monkeypatch):
    rec = _inject_curobo_types(monkeypatch)
    mp = FakeMP(_grasp_result())
    p = _planner(mp)
    K = 3
    goals = [Pose(np.eye(3), [0.4, -0.2, 0.8 + 0.01 * i]) for i in range(K)]
    p.plan_grasp_set(np.zeros(14), RIGHT, goals, approach_axis="x", approach_offset=-0.1,
                     lift_axis="z", lift_offset=0.1)
    fp = rec["from_poses"]
    # ONLY the active wrist is a goal link: plan_grasp offsets EVERY goal frame to form the
    # approach/lift pose, so the held idle arm + torso-locked camera must NOT be included
    # (offsetting the immovable camera goal makes the approach/lift IK infeasible).
    assert fp["num_goalset"] == K and fp["ordered"] == ["right_wrist_yaw_link"]
    assert list(fp["pose_dict"]) == ["right_wrist_yaw_link"]
    active = fp["pose_dict"]["right_wrist_yaw_link"]
    assert active.position.shape == (K, 3) and active.quaternion.shape == (K, 4)


def test_plan_grasp_set_clamps_k_to_max_goalset(monkeypatch):
    rec = _inject_curobo_types(monkeypatch)
    p = _planner(FakeMP(_grasp_result()))
    p._max_goalset = 2
    goals = [Pose.Identity() for _ in range(5)]
    p.plan_grasp_set(np.zeros(14), RIGHT, goals, approach_axis="x", approach_offset=-0.1,
                     lift_axis="z", lift_offset=0.1)
    assert rec["from_poses"]["num_goalset"] == 2          # clamped to max_goalset


def test_plan_grasp_set_duplicates_lone_goal(monkeypatch):
    # cuRobo's num_goalset=1 path is cold (warmup only primes the goalset path), so a lone
    # candidate is duplicated to num_goalset=2 (robust goalset path) and the chosen index is
    # clamped back to the single real candidate even if cuRobo returns the duplicate (index 1).
    rec = _inject_curobo_types(monkeypatch)
    out = _planner(FakeMP(_grasp_result(idx=1))).plan_grasp_set(
        np.zeros(14), RIGHT, [Pose(np.eye(3), [0.4, -0.2, 0.8])],
        approach_axis="x", approach_offset=-0.1, lift_axis="z", lift_offset=0.1)
    assert rec["from_poses"]["num_goalset"] == 2                    # lone goal duplicated
    active = rec["from_poses"]["pose_dict"]["right_wrist_yaw_link"]
    assert active.position.shape == (2, 3) and np.allclose(active.position[0], active.position[1])
    assert out.chosen_index == 0                                    # clamped to the real candidate


def test_plan_grasp_set_empty_raises():
    with pytest.raises(PlanningError):
        _planner(FakeMP(_grasp_result())).plan_grasp_set(
            np.zeros(14), RIGHT, [], approach_axis="x", approach_offset=-0.1,
            lift_axis="z", lift_offset=0.1)


# --------------------------------------------------------------- depth-ESDF collision world
def test_plan_grasp_set_disables_hand_links_when_cw_on(monkeypatch):
    from g1_classical_manip.motion.curobo_planner import HAND_LINKS
    _inject_curobo_types(monkeypatch)
    mp = FakeMP(_grasp_result())
    p = _planner(mp)
    p._cw_enabled = True                          # collision world on -> open hand may touch the ESDF
    p.plan_grasp_set(np.zeros(14), RIGHT, [Pose(np.eye(3), [0.4, -0.2, 0.8])],
                     approach_axis="y", approach_offset=-0.1, lift_axis="z", lift_offset=0.1)
    dl = mp.kw["disable_collision_links"]
    assert dl[0] == "right_wrist_yaw_link"
    assert set(HAND_LINKS[RIGHT]).issubset(set(dl))     # active hand links disabled for the grasp
    assert "right_hand_palm_link" in dl and "right_hand_thumb_2_link" in dl


def test_plan_grasp_set_no_hand_links_when_cw_off(monkeypatch):
    _inject_curobo_types(monkeypatch)
    mp = FakeMP(_grasp_result())
    _planner(mp).plan_grasp_set(np.zeros(14), RIGHT, [Pose(np.eye(3), [0.4, -0.2, 0.8])],
                                approach_axis="y", approach_offset=-0.1, lift_axis="z", lift_offset=0.1)
    assert mp.kw["disable_collision_links"] == ["right_wrist_yaw_link"]   # unchanged when world off


def test_hand_links_map():
    from g1_classical_manip.motion.curobo_planner import HAND_LINKS
    from g1_classical_manip.ee.hand_base import LEFT
    assert HAND_LINKS[RIGHT] == [
        "right_hand_palm_link", "right_hand_thumb_0_link", "right_hand_thumb_1_link",
        "right_hand_thumb_2_link", "right_hand_index_0_link", "right_hand_index_1_link",
        "right_hand_middle_0_link", "right_hand_middle_1_link"]
    assert all(lk.startswith("left_hand_") for lk in HAND_LINKS[LEFT])


def test_set_collision_world_toggles_and_drops_cache():
    p = CuroboArmPlanner.__new__(CuroboArmPlanner)
    p._cw_enabled = False
    p._cw_cfg = {}
    p._grasp_mp = {"right": "stale"}              # a cached non-voxel grasp planner
    p._esdf_mapper = "stale"
    assert p.collision_world_enabled is False
    p.set_collision_world(True, {"enabled": True, "extent_m": [1, 1, 1]})
    assert p.collision_world_enabled is True
    assert p._grasp_mp == {} and p._esdf_mapper is None      # dropped -> rebuilds voxel-capable
    assert p._cw_cfg["extent_m"] == [1, 1, 1]


def test_cw_params_defaults():
    p = CuroboArmPlanner.__new__(CuroboArmPlanner)
    p._cw_cfg = {}
    pr = p._cw_params()
    assert pr["grid_center"] == [0.4, 0.0, 0.2] and pr["extent_m"] == [1.2, 1.2, 1.0]
    assert pr["esdf_voxel_size"] == 0.02 and pr["depth_max_m"] == 2.0


def test_update_grasp_world_noops():
    p = CuroboArmPlanner.__new__(CuroboArmPlanner)
    p._cw_enabled = False
    assert p.update_grasp_world(RIGHT, np.zeros((4, 4)), {}, None) is False   # world off
    p._cw_enabled = True
    assert p.update_grasp_world(RIGHT, None, {}, None) is False               # no depth


# --------------------------------------------------------------- _hold_idle
def test_hold_idle_pins_idle_arm():
    from g1_classical_manip.motion.planner_base import JointTrajectory
    from g1_classical_manip.ee.hand_base import LEFT
    p = CuroboArmPlanner.__new__(CuroboArmPlanner)
    # a 4-step trajectory where every joint ramps (so drift would show); idle arm must be pinned.
    q = np.tile(np.linspace(1.0, 2.0, 14), (4, 1)) + np.arange(4)[:, None] * 0.1
    traj = JointTrajectory(np.arange(4) * 0.025, q.copy(), np.ones((4, 14)), np.ones((4, 14)))
    hold = np.arange(14, dtype=float) * 0.01            # the start/home config to pin to
    out = p._hold_idle(traj, RIGHT, hold)               # active = right (7:14), idle = left (0:7)
    assert np.allclose(out.q[:, 0:7], hold[0:7])        # idle (left) pinned to hold, every step
    assert np.allclose(out.qd[:, 0:7], 0.0) and np.allclose(out.qdd[:, 0:7], 0.0)
    assert np.allclose(out.q[:, 7:14], q[:, 7:14])      # active (right) untouched
    assert out.meta["idle_held"] is True
    # LEFT active -> right (7:14) is the idle arm that gets pinned
    traj2 = JointTrajectory(np.arange(4) * 0.025, q.copy(), np.ones((4, 14)), np.ones((4, 14)))
    out2 = p._hold_idle(traj2, LEFT, hold)
    assert np.allclose(out2.q[:, 7:14], hold[7:14]) and np.allclose(out2.q[:, 0:7], q[:, 0:7])


def test_hold_idle_passthrough_none():
    p = CuroboArmPlanner.__new__(CuroboArmPlanner)
    assert p._hold_idle(None, RIGHT, np.zeros(14)) is None


# --------------------------------------------------------------- plan_grasp_set_sweep
def test_plan_grasp_set_sweep_first_success():
    p = CuroboArmPlanner.__new__(CuroboArmPlanner)
    seen, good = [], GraspPlanOutcome(True, 0, "a", "g", "l", True, True, True, "ok")
    bad = GraspPlanOutcome(False, -1, None, None, None, False, False, False, "fail")

    def fake_set(q, side, goals, **kw):
        seen.append(kw["approach_offset"])
        return good if len(seen) == 3 else bad

    p.plan_grasp_set = fake_set
    strategies = [{"approach_offset": x} for x in (-0.15, -0.10, -0.07, -0.05)]
    out = p.plan_grasp_set_sweep(np.zeros(14), RIGHT, [Pose.Identity()], strategies,
                                 approach_axis="x", lift_axis="z")
    assert out is good and seen == [-0.15, -0.10, -0.07]   # stopped at first success


def test_plan_grasp_set_sweep_returns_last_failure():
    p = CuroboArmPlanner.__new__(CuroboArmPlanner)
    bad = GraspPlanOutcome(False, -1, None, None, None, False, False, False, "fail")
    p.plan_grasp_set = lambda *a, **k: bad
    out = p.plan_grasp_set_sweep(np.zeros(14), RIGHT, [Pose.Identity()],
                                 [{"approach_offset": -0.1}, {"approach_offset": -0.05}],
                                 approach_axis="x", lift_axis="z")
    assert out is bad and not out.success
    with pytest.raises(PlanningError):
        p.plan_grasp_set_sweep(np.zeros(14), RIGHT, [], [], approach_axis="x", lift_axis="z")


# --------------------------------------------------------------- grasp_motion primitive
class _Seg:
    def __init__(self, name):
        self.name = name
        self.q = np.zeros((2, 14))


class FakeExecutor:
    def __init__(self, events, fail=None):
        self.events = events
        self.fail = fail                       # segment name to fail on (or None)

    def run(self, traj):
        self.events.append(f"run:{traj.name}")
        from types import SimpleNamespace
        ok = traj.name != self.fail
        return SimpleNamespace(success=ok, reason=("ok" if ok else f"{traj.name} aborted"))

    def settle(self, q, *a, **k):
        self.events.append("settle")
        return 0.0


class FakeArm2:
    def get_current_dual_arm_q(self):
        return np.zeros(14)


class FakePlannerSweep:
    def __init__(self, outcome):
        self.outcome = outcome
        self.kw = None

    def plan_grasp_set_sweep(self, q, side, goals, **kw):
        self.kw = kw
        return self.outcome


class FakeRobot2:
    def __init__(self, outcome, fail=None):
        self.events = []
        self.arm = FakeArm2()
        self.planner = FakePlannerSweep(outcome)
        self.executor = FakeExecutor(self.events, fail=fail)
        self.cfg = {"planner": {"grasp": {"approach_axis": "x", "lift_axis": "z",
                                          "approach_in_tool_frame": True,
                                          "lift_in_tool_frame": False,
                                          "strategies": [{"approach_offset": -0.1}]}}}


def _cands(n=3):
    from g1_classical_manip.grasp.base import GraspCandidate
    return [GraspCandidate(wrist_goal=Pose(np.eye(3), [0.4, 0.0, 0.8]),
                           confidence=1.0 - 0.1 * i,
                           grasp_pose=Pose(np.eye(3), [0.4, 0.0, 0.8])) for i in range(n)]


def test_grasp_motion_orders_segments_and_callbacks():
    from g1_classical_manip import primitives as P
    out = GraspPlanOutcome(True, 1, _Seg("approach"), _Seg("grasp"), _Seg("lift"),
                           True, True, True, "ok")
    robot = FakeRobot2(out)
    cands = _cands(3)
    sel = {}

    def on_selected(chosen, outcome):
        robot.events.append("selected")
        sel["chosen"], sel["idx"] = chosen, outcome.chosen_index

    res = P.grasp_motion(robot, RIGHT, cands,
                         close_cb=lambda: robot.events.append("close"),
                         confirm_cb=lambda lbl: robot.events.append(f"confirm:{lbl}"),
                         on_selected=on_selected)
    assert res.ok and res.chosen_index == 1
    assert sel["chosen"] is cands[1] and sel["idx"] == 1     # chosen_index -> the candidate
    assert robot.events == ["selected",
                            "confirm:approach", "run:approach",
                            "confirm:grasp", "run:grasp",
                            "settle", "close",
                            "confirm:lift", "run:lift"]
    # config strategies + axes flow into the sweep
    assert robot.planner.kw["approach_axis"] == "x" and robot.planner.kw["lift_axis"] == "z"


def test_grasp_motion_short_circuits_on_failed_segment():
    from g1_classical_manip import primitives as P
    out = GraspPlanOutcome(True, 0, _Seg("approach"), _Seg("grasp"), _Seg("lift"),
                           True, True, True, "ok")
    robot = FakeRobot2(out, fail="grasp")                   # grasp segment aborts
    res = P.grasp_motion(robot, RIGHT, _cands(2),
                         close_cb=lambda: robot.events.append("close"))
    assert not res.ok and "grasp" in res.info
    assert "close" not in robot.events and "run:lift" not in robot.events   # never closed/lifted


def test_grasp_motion_fails_when_plan_grasp_fails():
    from g1_classical_manip import primitives as P
    bad = GraspPlanOutcome(False, -1, None, None, None, False, False, False, "no plan")
    robot = FakeRobot2(bad)
    res = P.grasp_motion(robot, RIGHT, _cands(2), close_cb=lambda: robot.events.append("close"))
    assert not res.ok and "no plan" in res.info and robot.events == []   # nothing executed


def test_grasp_motion_no_candidates():
    from g1_classical_manip import primitives as P
    robot = FakeRobot2(GraspPlanOutcome(True, 0, _Seg("a"), _Seg("g"), _Seg("l"),
                                        True, True, True, "ok"))
    res = P.grasp_motion(robot, RIGHT, [])
    assert not res.ok and "no candidates" in res.info
