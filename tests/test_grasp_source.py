"""Grasp sources, all with fakes (no GPU/robot/zmq)."""
import numpy as np
import pytest

from g1_classical_manip.spatial.pose import Pose, rpy_to_matrix
from g1_classical_manip.ee.hand_base import RIGHT
from g1_classical_manip.perception.depth import deproject_depth
from g1_classical_manip.perception.segment import Segmenter
from g1_classical_manip.grasp.graspgenx_source import GraspGenXGraspSource
from g1_classical_manip.grasp.tool_transform import build_T_wristyaw_grasp

PALM = [0.1192, -0.0346, 0.0]
GCFG = {"palm_offset_xyz": PALM, "wristyaw_grasp_rpy": [0.0, np.pi / 2, 0.0],
        "gripper_name": "unitree_g1", "num_grasps": 50, "topk": 20,
        "grasp_threshold": -1.0, "voxel_m": 0.003}
CAM_CFG = {"intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0}}


class FakeArm:
    def get_current_dual_arm_q(self):
        return np.zeros(14)


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
    def __init__(self, grasps, conf, tags=None):
        self.grasps, self.conf, self.tags = grasps, conf, (tags or [])
        self.sent, self.kw = None, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def infer(self, points, **kw):                  # v2: returns a 3-tuple (+ branch_tags)
        self.sent = np.asarray(points)
        self.kw = kw
        return self.grasps, self.conf, self.tags


class FakeRobot:
    def __init__(self, planner=None, camera=None):
        self.arm = FakeArm()
        self.planner = planner
        self.camera = camera


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


def test_graspgenx_source_passes_planner_and_stashes_branch_tags():
    # Protocol v2: the planner kwargs from cfg reach infer, and each grasp's branch_tag
    # follows it through the confidence sort (g1 conf 0.9 ranks first, carrying "diff").
    g0 = Pose(np.eye(3), [0.41, -0.21, 0.81]).homogeneous.astype(np.float32)
    g1 = Pose(np.eye(3), [0.39, -0.19, 0.79]).homogeneous.astype(np.float32)
    client = FakeClient(np.stack([g0, g1]), np.array([0.3, 0.9], np.float32),
                        tags=["obb", "diff"])
    gcfg = {**GCFG, "planner": "topdown", "obb_density": "dense", "skip_obb_rule": "never"}
    src = GraspGenXGraspSource(FakeFrames(), FakeSeg(np.ones((3, 3), bool)),
                               lambda: client, gcfg, CAM_CFG)
    cands = src.grasps(FakeRobot(camera=FakeCam(DEPTH)), RIGHT, "block")
    assert client.kw["planner"] == "topdown"
    assert client.kw["obb_density"] == "dense" and client.kw["skip_obb_rule"] == "never"
    assert cands[0].confidence == pytest.approx(0.9) and cands[0].extra["branch_tag"] == "diff"
    assert cands[1].confidence == pytest.approx(0.3) and cands[1].extra["branch_tag"] == "obb"


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


def test_graspgenx_source_retains_last_mask_for_exclude_object():
    # the segmenter's mask must be RETAINED on the source (collision_world.exclude_object reads
    # it to cut the target out of the ESDF), and RESET at every grasps() call so an early
    # return (e.g. camera gone) can never leave a stale mask behind.
    m = np.zeros((3, 3), bool)
    m[1, 1] = True
    client = FakeClient(np.eye(4, dtype=np.float32)[None], np.array([1.0], np.float32))
    src = GraspGenXGraspSource(FakeFrames(), FakeSeg(m), lambda: client, GCFG, CAM_CFG)
    assert src.last_mask is None                            # nothing segmented yet
    src.grasps(FakeRobot(camera=FakeCam(DEPTH)), RIGHT, "block")
    assert src.last_mask is not None and (src.last_mask == m).all()
    src.grasps(FakeRobot(camera=None), RIGHT, "block")      # early return BEFORE segmentation
    assert src.last_mask is None                            # stale mask cleared
