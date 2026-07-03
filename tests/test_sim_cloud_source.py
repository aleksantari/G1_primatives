"""Sim ground-truth grasp source: cube sampler + ranking/transform (no robot/zmq/Isaac)."""
import numpy as np
import pytest

from g1_primitives.spatial.pose import Pose, rpy_to_matrix
from g1_primitives.ee.hand_base import RIGHT
from g1_primitives.grasp.tool_transform import build_T_wristyaw_grasp
from g1_primitives.grasp.sim_cloud_source import SimCloudGraspSource, sample_cube

PALM = [0.1192, -0.0346, 0.0]
GCFG = {"palm_offset_xyz": PALM, "wristyaw_grasp_rpy": [0.0, np.pi / 2, 0.0],
        "gripper_name": "unitree_g1", "num_grasps": 50, "topk": 20, "grasp_threshold": -1.0,
        "voxel_m": 0.003,
        "sim_cloud": {"object_size_m": 0.06, "n_points": 600, "camera_facing_cull": False}}


class FakePoseSource:
    def __init__(self, pose):
        self._pose = pose

    def block_pose(self, label="block"):
        return self._pose


class FakeFrames:
    def T_pelvis_camera(self, q14=None):
        return Pose.Identity()                     # head cam at the pelvis origin


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


def test_sample_cube_lies_on_surface():
    pts, normals = sample_cube(0.06, 600)
    h = 0.03
    assert pts.shape[1] == 3 and normals.shape == pts.shape
    assert np.all(np.abs(pts) <= h + 1e-6)                       # inside the cube bounds
    assert np.allclose(np.max(np.abs(pts), axis=1), h, atol=1e-6)  # every point on a face
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)    # unit axis normals


def test_sim_cloud_source_ranks_and_transforms():
    g0 = Pose(rpy_to_matrix(0.1, 0.2, 0.3), [0.41, -0.21, 0.81]).homogeneous.astype(np.float32)
    g1 = Pose(rpy_to_matrix(-0.2, 0.1, 0.0), [0.39, -0.19, 0.79]).homogeneous.astype(np.float32)
    conf = np.array([0.3, 0.9], np.float32)        # g1 better -> first
    client = FakeClient(np.stack([g0, g1]), conf)
    src = SimCloudGraspSource(FakeFrames(), FakePoseSource(Pose(np.eye(3), [0.4, -0.2, 0.8])),
                              lambda: client, GCFG)
    cands = src.grasps(robot=None, side=RIGHT, target="block")   # robot unused (no camera)
    assert len(cands) == 2
    assert cands[0].confidence == pytest.approx(0.9) and cands[1].confidence == pytest.approx(0.3)
    assert client.sent is not None and client.sent.shape[1] == 3 and client.sent.shape[0] > 0
    T_wg = build_T_wristyaw_grasp(PALM, RIGHT, rpy_to_matrix(*GCFG["wristyaw_grasp_rpy"]))
    for c in cands:
        np.testing.assert_allclose(c.wrist_goal.homogeneous,
                                   (c.grasp_pose * T_wg.inverse()).homogeneous, atol=1e-6)


def test_sim_cloud_source_passes_planner_and_stashes_branch_tags():
    # The sim de-risk path must request top-down too: planner kwargs (from the shared graspgenx
    # cfg) reach infer, and each branch_tag follows its grasp through the confidence sort.
    g0 = Pose(np.eye(3), [0.41, -0.21, 0.81]).homogeneous.astype(np.float32)
    g1 = Pose(np.eye(3), [0.39, -0.19, 0.79]).homogeneous.astype(np.float32)
    client = FakeClient(np.stack([g0, g1]), np.array([0.3, 0.9], np.float32),
                        tags=["obb", "diff"])
    gcfg = {**GCFG, "planner": "topdown", "obb_density": "dense", "skip_obb_rule": "auto"}
    src = SimCloudGraspSource(FakeFrames(), FakePoseSource(Pose(np.eye(3), [0.4, -0.2, 0.8])),
                              lambda: client, gcfg)
    cands = src.grasps(robot=None, side=RIGHT, target="block")
    assert client.kw["planner"] == "topdown" and client.kw["obb_density"] == "dense"
    assert client.kw["skip_obb_rule"] == "auto"
    assert cands[0].confidence == pytest.approx(0.9) and cands[0].extra["branch_tag"] == "diff"
    assert cands[1].extra["branch_tag"] == "obb"


def test_sim_cloud_source_empty_without_pose():
    src = SimCloudGraspSource(FakeFrames(), FakePoseSource(None), lambda: None, GCFG)
    assert src.grasps(robot=None, side=RIGHT, target="block") == []


def test_camera_facing_cull_drops_far_faces():
    # block in front along +x, cam at the origin -> the far (+x) faces get culled.
    cfg = {**GCFG, "voxel_m": None,
           "sim_cloud": {"object_size_m": 0.06, "n_points": 600, "camera_facing_cull": True}}
    client = FakeClient(np.eye(4, dtype=np.float32)[None], np.array([1.0], np.float32))
    src = SimCloudGraspSource(FakeFrames(), FakePoseSource(Pose(np.eye(3), [0.5, 0.0, 0.0])),
                              lambda: client, cfg)
    src.grasps(robot=None, side=RIGHT, target="block")
    full = sample_cube(0.06, 600)[0].shape[0]
    assert 0 < client.sent.shape[0] < full          # some but not all points survive the cull


# ----------------------------------------------------------------- color from the sim RGB
class FakeCamRGB:
    def __init__(self, rgb):
        self._rgb = rgb

    def get_rgb_frame(self):
        return self._rgb


class FakeRobotCam:
    def __init__(self, camera):
        self.camera = camera


class FakeViz:
    def __init__(self):
        self.colors = "unset"
        self.branch_tags = "unset"

    def show_candidates(self, points, grasps, conf, colors=None, branch_tags=None):
        self.colors = colors
        self.branch_tags = branch_tags


def test_sim_cloud_source_colors_from_sim_rgb():
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[:] = (200, 30, 30)                                  # all-red sim render
    g0 = Pose(np.eye(3), [0.0, 0.0, 0.5]).homogeneous.astype(np.float32)
    viz = FakeViz()
    cam_cfg = {"intrinsics": {"fx": 243.2, "fy": 243.2, "cx": 320.0, "cy": 240.0}}
    src = SimCloudGraspSource(FakeFrames(), FakePoseSource(Pose(np.eye(3), [0.0, 0.0, 0.5])),
                              lambda: FakeClient(g0[None], np.array([1.0], np.float32)),
                              {**GCFG, "voxel_m": None}, cam_cfg, viz=viz)
    src.grasps(FakeRobotCam(FakeCamRGB(rgb)), RIGHT, "block")   # block in front of the head cam
    assert viz.colors is not None and viz.colors.dtype == np.uint8 and viz.colors.shape[1] == 3
    assert (viz.colors == (200, 30, 30)).all(axis=1).any()     # sampled the red sim render


def test_sim_cloud_source_no_camera_no_color():
    g0 = Pose(np.eye(3), [0.0, 0.0, 0.5]).homogeneous.astype(np.float32)
    viz = FakeViz()
    src = SimCloudGraspSource(FakeFrames(), FakePoseSource(Pose(np.eye(3), [0.0, 0.0, 0.5])),
                              lambda: FakeClient(g0[None], np.array([1.0], np.float32)),
                              {**GCFG, "voxel_m": None}, camera_cfg=None, viz=viz)
    src.grasps(FakeRobotCam(camera=None), RIGHT, "block")       # no camera -> flat (colors None)
    assert viz.colors is None


def test_sim_cloud_source_snapshot_carries_gt_cloud():
    # the perception-validation tool reads last_snapshot.cloud as the GT reference.
    client = FakeClient(np.eye(4, dtype=np.float32)[None], np.array([1.0], np.float32))
    src = SimCloudGraspSource(FakeFrames(), FakePoseSource(Pose(np.eye(3), [0.4, -0.2, 0.8])),
                              lambda: client, GCFG)
    cands = src.grasps(robot=None, side=RIGHT, target="block")
    snap = src.last_snapshot
    assert snap is not None and snap.target == "block"
    assert snap.mask is None                              # no 2D mask on the GT path
    assert snap.cloud is not None and snap.cloud.points.shape[0] > 0
    assert snap.n_candidates == len(cands)
    src2 = SimCloudGraspSource(FakeFrames(), FakePoseSource(None), lambda: None, GCFG)
    src2.grasps(robot=None, side=RIGHT, target="block")   # no pose -> early return
    assert src2.last_snapshot is None
