"""perception.validation on synthetic fixtures (GPU-free): shifted-cube cloud errors,
pinhole mask projection, IoU, grasp-implied object error, fingertip FK plumbing."""
import numpy as np
import pytest

from g1_primitives.spatial.pose import Pose
from g1_primitives.spatial.pointcloud import PointCloud
from g1_primitives.grasp.base import GraspCandidate
from g1_primitives.grasp.sim_cloud_source import sample_cube
from g1_primitives.perception import validation as V


def _cube(center=(0.0, 0.0, 0.0), edge=0.06, n=1500):
    pts, _ = sample_cube(edge, n)
    return PointCloud((pts + np.asarray(center, np.float32)).astype(np.float32))


# ----------------------------------------------------------------- cloud_error
def test_cloud_error_identical_cube():
    c = _cube()
    e = V.cloud_error(c, c)
    assert e.centroid_mm < 1e-6
    assert e.chamfer_mean_mm < 1e-6
    assert e.inlier_frac == 1.0
    assert np.allclose(e.bbox_delta_mm, 0.0)


def test_cloud_error_shifted_cube_reports_the_shift():
    gt = _cube()
    shifted = _cube(center=(0.02, 0.0, 0.0))            # 20 mm in +x
    e = V.cloud_error(shifted, gt, inlier_tol_m=0.01)
    assert abs(e.centroid_mm - 20.0) < 0.5
    assert abs(e.centroid_delta_mm[0] - 20.0) < 0.5     # signed, per-axis
    assert 0.0 < e.chamfer_mean_mm <= 20.5
    assert e.inlier_frac < 1.0                          # 10 mm tol misses a 20 mm shift
    assert np.allclose(e.bbox_delta_mm, 0.0, atol=1e-3)  # same size, just moved (float32 dust)


def test_cloud_error_bbox_delta_catches_scale():
    gt = _cube(edge=0.06)
    big = _cube(edge=0.08)                              # +20 mm per axis
    e = V.cloud_error(big, gt)
    assert np.allclose(e.bbox_delta_mm, 20.0, atol=0.5)


def test_cloud_error_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        V.cloud_error(np.zeros((0, 3)), _cube())


# ------------------------------------------------------- project_mask / mask_iou
_K = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}


def test_project_mask_pinhole_center_and_behind():
    T = Pose(np.eye(3), [0.0, 0.0, 0.0])                # camera optical = pelvis
    pts = np.array([[0.0, 0.0, 1.0],                    # on-axis, 1 m ahead -> (50, 50)
                    [0.1, 0.0, 1.0],                    # +x -> u = 60
                    [0.0, 0.0, -1.0]])                  # behind the camera -> dropped
    mask = V.project_mask(pts, _K, T, (100, 100))
    assert mask[50, 50] and mask[50, 60]
    assert mask.sum() == 2                              # the behind-point never lands


def test_project_mask_dilate_grows_the_blob():
    T = Pose(np.eye(3), [0.0, 0.0, 0.0])
    pts = np.array([[0.0, 0.0, 1.0]])
    assert V.project_mask(pts, _K, T, (100, 100)).sum() == 1
    assert V.project_mask(pts, _K, T, (100, 100), dilate_px=2).sum() > 1


def test_mask_iou():
    a = np.zeros((10, 30), bool); a[:, :20] = True
    b = np.zeros((10, 30), bool); b[:, 10:] = True
    assert abs(V.mask_iou(a, b) - (10 / 30)) < 1e-9     # overlap 10 cols / union 30
    assert V.mask_iou(a, a) == 1.0
    assert V.mask_iou(a, np.zeros_like(a)) == 0.0
    assert V.mask_iou(None, a) == 0.0
    with pytest.raises(ValueError, match="shapes"):
        V.mask_iou(a, np.zeros((5, 5), bool))


# ---------------------------------------------------------- candidate_set_error
def _cand(grasp_t, conf=1.0):
    gp = Pose(np.eye(3), grasp_t)                       # approach = +Z (identity rotation)
    return GraspCandidate(wrist_goal=gp, confidence=conf, grasp_pose=gp)


def test_candidate_set_error_implied_object_point():
    gt = np.array([0.30, 0.00, 0.10])
    perfect = _cand(gt - np.array([0.0, 0.0, 0.07]))    # origin 7 cm back along approach
    off = _cand(gt - np.array([0.0, 0.0, 0.07]) + np.array([0.05, 0.0, 0.0]))
    out = V.candidate_set_error([off, perfect], gt, fingertip_depth_m=0.07)
    assert out["n"] == 2
    assert out["top1_mm"] == pytest.approx(50.0, abs=0.5)   # top-1 is the OFF one
    assert out["best_mm"] == pytest.approx(0.0, abs=0.5)


def test_candidate_set_error_no_grasp_poses():
    c = GraspCandidate(wrist_goal=Pose(np.eye(3), [0, 0, 0]), grasp_pose=None)
    out = V.candidate_set_error([c], [0.3, 0.0, 0.1])
    assert out["n"] == 0 and out["top1_mm"] is None


# ------------------------------------------------------- fingertip_contact_error
def test_fingertip_contact_error_zero_at_own_contact():
    # place the GT exactly at OUR FK contact point -> the error must be ~0.
    from g1_primitives.ee.hand_kinematics import Dex3Kinematics
    from g1_primitives.config import load_configs
    side = "right"
    q7 = load_configs()["hands"]["dex3"]["presets"]["power_close"][side]
    wrist = Pose(np.eye(3), [0.30, -0.10, 0.05])
    contact_local = Dex3Kinematics().contact_point(side, np.asarray(q7, float))
    gt = (wrist * Pose(np.eye(3), contact_local)).translation
    out = V.fingertip_contact_error(side, wrist, gt, q7)
    assert out["contact_mm"] == pytest.approx(0.0, abs=0.1)
    assert out["thumb_mm"] > 0.0 and out["fingers_mm"] > 0.0
