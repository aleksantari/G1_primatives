"""Sim ground-truth grasp source: cube sampler + ranking/transform (no robot/zmq/Isaac)."""
import numpy as np
import pytest

from g1_classical_manip.spatial.pose import Pose, rpy_to_matrix
from g1_classical_manip.ee.hand_base import RIGHT
from g1_classical_manip.grasp.tool_transform import build_T_wristyaw_grasp
from g1_classical_manip.grasp.sim_cloud_source import SimCloudGraspSource, sample_cube

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
    def __init__(self, grasps, conf):
        self.grasps, self.conf, self.sent = grasps, conf, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def infer(self, points, **kw):
        self.sent = np.asarray(points)
        return self.grasps, self.conf


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
