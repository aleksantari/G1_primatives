"""Grasp sources + multi-candidate planning, all with fakes (no GPU/robot/zmq)."""
import numpy as np
import pytest

from g1_classical_manip.spatial.pose import Pose, rpy_to_matrix
from g1_classical_manip.ee.hand_base import LEFT, RIGHT
from g1_classical_manip.perception.base import Detection
from g1_classical_manip.perception.depth import deproject_depth
from g1_classical_manip.perception.segment import Segmenter
from g1_classical_manip.grasp.apriltag_source import AprilTagGraspSource
from g1_classical_manip.grasp.graspgenx_source import GraspGenXGraspSource
from g1_classical_manip.grasp.tool_transform import palm_offset, build_T_wristyaw_grasp

PALM = [0.1192, -0.0346, 0.0]
GCFG = {"palm_offset_xyz": PALM, "wristyaw_grasp_rpy": [0.0, np.pi / 2, 0.0],
        "gripper_name": "unitree_g1", "num_grasps": 50, "topk": 20,
        "grasp_threshold": -1.0, "voxel_m": 0.003}
CAM_CFG = {"intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0}}


class FakeArm:
    def get_current_dual_arm_q(self):
        return np.zeros(14)


class FakePlanner:
    def __init__(self, R):
        self.R = R

    def fk(self, side, q):
        return Pose(self.R, [0.0, 0.0, 0.0])


_DEFAULT_RGB = object()


class FakeCam:
    def __init__(self, depth, rgb=_DEFAULT_RGB):
        self._d = depth
        self._rgb = np.zeros((3, 3, 3), np.uint8) if rgb is _DEFAULT_RGB else rgb

    def get_depth_frame(self):
        return self._d

    def get_rgb_frame(self):
        return self._rgb


class FakeFrames:
    def T_pelvis_camera(self, q14=None):
        return Pose.Identity()


class FakeSeg(Segmenter):
    def __init__(self, mask):
        self._mask = mask

    def mask(self, rgb):
        return self._mask


class FakeClient:
    def __init__(self, grasps, conf):
        self.grasps, self.conf, self.sent = grasps, conf, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def infer(self, points, **kw):
        self.sent = np.asarray(points)
        return self.grasps, self.conf


class FakeRobot:
    def __init__(self, planner=None, camera=None):
        self.arm = FakeArm()
        self.planner = planner
        self.camera = camera


# ----------------------------------------------------------------- AprilTag source
def test_apriltag_source_matches_07(monkeypatch):
    from g1_classical_manip import primitives as P
    center = np.array([0.45, -0.2, 0.85])
    monkeypatch.setattr(P, "detect",
                        lambda robot, target: Detection(pose=Pose(np.eye(3), center)))
    R0 = rpy_to_matrix(0.1, 0.2, -0.3)
    src = AprilTagGraspSource(PALM, [0.0, 0.0, 0.02], grasp_rpy=None)
    cands = src.grasps(FakeRobot(planner=FakePlanner(R0)), RIGHT, "block")
    assert len(cands) == 1 and cands[0].confidence == 1.0
    off = palm_offset(PALM, RIGHT) + np.array([0.0, 0.0, 0.02])
    np.testing.assert_allclose(cands[0].wrist_goal.translation, center + R0 @ (-off), atol=1e-9)
    np.testing.assert_allclose(cands[0].wrist_goal.rotation, R0, atol=1e-12)


def test_apriltag_source_empty_when_no_detection(monkeypatch):
    from g1_classical_manip import primitives as P
    monkeypatch.setattr(P, "detect", lambda robot, target: None)
    src = AprilTagGraspSource(PALM, [0.0, 0.0, 0.02])
    assert src.grasps(FakeRobot(planner=FakePlanner(np.eye(3))), RIGHT, "block") == []


# ----------------------------------------------------------------- GraspGenX source
DEPTH = np.full((3, 3), 500.0, np.float32)


def test_graspgenx_source_ranks_and_transforms():
    g0 = Pose(rpy_to_matrix(0.1, 0.2, 0.3), [0.41, -0.21, 0.81]).homogeneous.astype(np.float32)
    g1 = Pose(rpy_to_matrix(-0.2, 0.1, 0.0), [0.39, -0.19, 0.79]).homogeneous.astype(np.float32)
    conf = np.array([0.3, 0.9], np.float32)        # g1 better -> must come first
    client = FakeClient(np.stack([g0, g1]), conf)
    src = GraspGenXGraspSource(FakeFrames(), FakeSeg(np.ones((3, 3), bool)),
                               lambda: client, GCFG, CAM_CFG)
    cands = src.grasps(FakeRobot(camera=FakeCam(DEPTH)), RIGHT, "block")
    assert len(cands) == 2
    assert cands[0].confidence == pytest.approx(0.9) and cands[1].confidence == pytest.approx(0.3)
    assert client.sent is not None and client.sent.shape[1] == 3 and client.sent.shape[0] > 0
    T_wg = build_T_wristyaw_grasp(PALM, RIGHT, rpy_to_matrix(*GCFG["wristyaw_grasp_rpy"]))
    for cand in cands:
        expected = (cand.grasp_pose * T_wg.inverse()).homogeneous
        np.testing.assert_allclose(cand.wrist_goal.homogeneous, expected, atol=1e-6)


def test_graspgenx_source_masks_depth_before_deproject():
    m = np.zeros((3, 3), bool)
    m[1, 1] = True
    m[0, 2] = True
    client = FakeClient(np.eye(4, dtype=np.float32)[None], np.array([1.0], np.float32))
    src = GraspGenXGraspSource(FakeFrames(), FakeSeg(m), lambda: client, GCFG, CAM_CFG)
    src.grasps(FakeRobot(camera=FakeCam(DEPTH)), RIGHT, "block")
    exp = deproject_depth(DEPTH, CAM_CFG["intrinsics"], Pose.Identity(),
                          voxel_m=GCFG["voxel_m"], mask=m).points
    assert client.sent.shape[0] == int(m.sum())            # only the masked pixels became points
    got = client.sent[np.lexsort(client.sent.T)]           # order-independent compare (voxel sorts)
    np.testing.assert_allclose(got, exp[np.lexsort(exp.T)], atol=1e-6)


def test_graspgenx_source_empty_on_missing_inputs():
    src = GraspGenXGraspSource(FakeFrames(), FakeSeg(np.ones((3, 3), bool)),
                               lambda: None, GCFG, CAM_CFG)
    assert src.grasps(FakeRobot(camera=FakeCam(None)), RIGHT, "block") == []          # no depth
    assert src.grasps(FakeRobot(camera=FakeCam(DEPTH, rgb=None)), RIGHT, "block") == []  # no rgb
    assert src.grasps(FakeRobot(camera=None), RIGHT, "block") == []                   # no camera
    # all-False mask -> empty cloud -> no grasps (loud, not whole-scene)
    src2 = GraspGenXGraspSource(FakeFrames(), FakeSeg(np.zeros((3, 3), bool)),
                                lambda: None, GCFG, CAM_CFG)
    assert src2.grasps(FakeRobot(camera=FakeCam(DEPTH)), RIGHT, "block") == []


# ----------------------------------------------------------------- plan_to_pose_set
def test_plan_to_pose_set_returns_first_success():
    from g1_classical_manip.motion.curobo_planner import CuroboArmPlanner, PlanningError
    planner = CuroboArmPlanner.__new__(CuroboArmPlanner)   # skip cuRobo __init__
    calls = []

    def fake(q, side, goal):
        calls.append(goal)
        if len(calls) < 3:
            raise PlanningError("nope")
        return "TRAJ"

    planner.plan_to_pose = fake
    goals = [Pose.Identity() for _ in range(4)]
    assert planner.plan_to_pose_set(np.zeros(14), RIGHT, goals) == "TRAJ"
    assert len(calls) == 3                                 # stopped at first success


def test_plan_to_pose_set_all_fail_raises():
    from g1_classical_manip.motion.curobo_planner import CuroboArmPlanner, PlanningError
    planner = CuroboArmPlanner.__new__(CuroboArmPlanner)

    def fake(q, side, goal):
        raise PlanningError("nope")

    planner.plan_to_pose = fake
    with pytest.raises(PlanningError):
        planner.plan_to_pose_set(np.zeros(14), RIGHT, [Pose.Identity(), Pose.Identity()])
    with pytest.raises(PlanningError):
        planner.plan_to_pose_set(np.zeros(14), RIGHT, [])
