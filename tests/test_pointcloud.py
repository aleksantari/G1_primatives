import numpy as np
import pytest

from g1_classical_manip.spatial.pointcloud import PointCloud
from g1_classical_manip.spatial.pose import Pose, rpy_to_matrix


def test_validation_and_dtype():
    pc = PointCloud(np.zeros((5, 3)))
    assert pc.n == 5 and pc.points.dtype == np.float32 and pc.frame == "pelvis"
    assert not pc.is_empty()
    assert PointCloud(np.zeros((0, 3))).is_empty()
    with pytest.raises(ValueError):
        PointCloud(np.zeros((4, 2)))


def test_transformed_matches_manual():
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], np.float32)
    R = rpy_to_matrix(0.0, 0.0, np.pi / 2)          # +90deg about z
    t = np.array([0.5, -1.0, 2.0])
    out = PointCloud(pts, frame="camera").transformed(Pose(R, t), frame="pelvis")
    expected = (pts @ R.T + t).astype(np.float32)
    assert out.frame == "pelvis"
    np.testing.assert_allclose(out.points, expected, atol=1e-6)
    # +90deg about z sends +x -> +y
    np.testing.assert_allclose(out.points[0], R @ pts[0] + t, atol=1e-6)


def test_voxel_downsample_collapses_and_centroids():
    # two tight clusters >> one voxel apart -> 2 points; each is its cluster centroid
    a = np.array([[0.001, 0.0, 0.0], [0.002, 0.0, 0.0], [0.0, 0.001, 0.0]], np.float32)
    b = a + np.array([1.0, 0.0, 0.0], np.float32)
    pc = PointCloud(np.vstack([a, b])).voxel_downsampled(0.05)
    assert pc.n == 2
    cents = pc.points[np.argsort(pc.points[:, 0])]
    np.testing.assert_allclose(cents[0], a.mean(0), atol=1e-5)
    np.testing.assert_allclose(cents[1], b.mean(0), atol=1e-5)
    # no-op cases
    assert PointCloud(a).voxel_downsampled(0).n == 3
    assert PointCloud(np.zeros((0, 3))).voxel_downsampled(0.01).is_empty()


def test_cropped_box():
    pts = np.array([[0, 0, 0], [0.05, 0, 0], [0.2, 0, 0], [0, 0.3, 0]], np.float32)
    out = PointCloud(pts).cropped_box(center=[0, 0, 0], half_extent=0.1)
    assert out.n == 2                                # only the two within 0.1 of origin
    out2 = PointCloud(pts).cropped_box(center=[0, 0, 0], half_extent=[0.1, 0.4, 0.1])
    assert out2.n == 3                               # the y=0.3 point now inside


def test_colors_validation_and_default():
    assert PointCloud(np.zeros((5, 3))).colors is None       # optional, off by default
    pc = PointCloud(np.zeros((5, 3)), colors=np.zeros((5, 3)))
    assert pc.colors.shape == (5, 3) and pc.colors.dtype == np.uint8
    with pytest.raises(ValueError):                          # colors must match points
        PointCloud(np.zeros((5, 3)), colors=np.zeros((4, 3)))


def test_colors_carry_through_transform_and_crop():
    pts = np.array([[0, 0, 0], [0.05, 0, 0], [0.2, 0, 0]], np.float32)
    cols = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], np.uint8)
    pc = PointCloud(pts, colors=cols)
    # rigid transform leaves color untouched, same order
    out = pc.transformed(Pose(rpy_to_matrix(0, 0, np.pi / 2), [1, 0, 0]), frame="pelvis")
    np.testing.assert_array_equal(out.colors, cols)
    # crop selects color with the same mask
    cropped = pc.cropped_box(center=[0, 0, 0], half_extent=0.1)
    assert cropped.n == 2
    np.testing.assert_array_equal(cropped.colors, cols[:2])


def test_colors_averaged_in_voxel_downsample():
    # two points in one voxel (avg color) + one far point (its own voxel, color kept)
    pts = np.array([[0.001, 0, 0], [0.002, 0, 0], [1.0, 0, 0]], np.float32)
    cols = np.array([[0, 0, 0], [100, 200, 40], [10, 20, 30]], np.uint8)
    pc = PointCloud(pts, colors=cols).voxel_downsampled(0.05)
    assert pc.n == 2 and pc.colors.shape == (2, 3) and pc.colors.dtype == np.uint8
    by_x = np.argsort(pc.points[:, 0])
    np.testing.assert_array_equal(pc.colors[by_x][0], [50, 100, 20])   # mean of the two
    np.testing.assert_array_equal(pc.colors[by_x][1], [10, 20, 30])    # lone point unchanged
    # XYZ-only cloud still downsamples without color
    assert PointCloud(pts).voxel_downsampled(0.05).colors is None
