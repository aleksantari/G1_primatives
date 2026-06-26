import numpy as np
import pytest

from g1_classical_manip.perception.depth import deproject_depth
from g1_classical_manip.spatial.pose import Pose, rpy_to_matrix

H, W = 3, 3
FX = FY = 100.0
CX = CY = 1.0          # principal point at pixel (1,1) -> that pixel deprojects to [0,0,z]
INTR = {"fx": FX, "fy": FY, "cx": CX, "cy": CY}


def _expected(depth_mm, mask=None):
    """Row-major (np.nonzero order) closed-form pinhole points for the valid pixels."""
    vv, uu = np.mgrid[0:H, 0:W]
    vs, us = vv.ravel(), uu.ravel()
    z = depth_mm.ravel() / 1000.0
    keep = np.isfinite(z) & (z > 0) if mask is None else mask.ravel()
    z, us, vs = z[keep], us[keep], vs[keep]
    x = (us - CX) / FX * z
    y = (vs - CY) / FY * z
    return np.stack([x, y, z], axis=1).astype(np.float32)


def test_constant_plane_closed_form():
    depth = np.full((H, W), 500.0, np.float32)            # 0.5 m plane
    cloud = deproject_depth(depth, INTR, Pose.Identity())
    assert cloud.frame == "pelvis" and cloud.n == 9
    np.testing.assert_allclose(cloud.points, _expected(depth), atol=1e-6)
    # principal-point pixel (v=1,u=1) -> row-major index 4 -> [0,0,0.5]
    np.testing.assert_allclose(cloud.points[4], [0.0, 0.0, 0.5], atol=1e-6)


def test_invalid_pixels_dropped():
    depth = np.full((H, W), 500.0, np.float32)
    depth[0, 0] = np.nan
    depth[0, 1] = 0.0
    depth[2, 2] = np.inf
    cloud = deproject_depth(depth, INTR, Pose.Identity())
    assert cloud.n == 6


def test_z_range_filter():
    depth = np.full((H, W), 500.0, np.float32)
    depth[1, 1] = 30.0                                    # 0.03 m < z_min 0.05 -> dropped
    cloud = deproject_depth(depth, INTR, Pose.Identity(), z_min_m=0.05, z_max_m=2.0)
    assert cloud.n == 8


def test_extrinsic_applied():
    depth = np.full((H, W), 500.0, np.float32)
    T = Pose(rpy_to_matrix(0.0, 0.0, np.pi / 2), [1.0, 0.0, 0.0])   # +90deg z, +x shift
    cloud = deproject_depth(depth, INTR, T)
    expected_optical = _expected(depth)
    expected_pelvis = expected_optical @ T.rotation.T + T.translation
    np.testing.assert_allclose(cloud.points, expected_pelvis, atol=1e-5)


def test_mask_drops_pixels():
    depth = np.full((H, W), 500.0, np.float32)             # all valid -> only mask gates
    mask = np.zeros((H, W), bool)
    mask[1, 1] = True
    mask[2, 0] = True
    cloud = deproject_depth(depth, INTR, Pose.Identity(), mask=mask)
    assert cloud.n == 2 == int(mask.sum())
    np.testing.assert_allclose(cloud.points, _expected(depth, mask=mask), atol=1e-6)


def test_mask_wrong_shape_raises():
    depth = np.full((H, W), 500.0, np.float32)
    with pytest.raises(ValueError):
        deproject_depth(depth, INTR, Pose.Identity(), mask=np.ones((H + 1, W), bool))


def test_rgb_attaches_per_point_color():
    depth = np.full((H, W), 500.0, np.float32)
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[..., 0] = np.arange(W)[None, :]              # R varies by column
    rgb[..., 1] = np.arange(H)[:, None]              # G varies by row
    cloud = deproject_depth(depth, INTR, Pose.Identity(), rgb=rgb)
    assert cloud.colors is not None and cloud.colors.shape == (H * W, 3)
    vv, uu = np.mgrid[0:H, 0:W]
    np.testing.assert_array_equal(cloud.colors, rgb[vv.ravel(), uu.ravel()])   # row-major
    # XYZ unaffected by the color path
    np.testing.assert_allclose(cloud.points, _expected(depth), atol=1e-6)


def test_rgb_shape_mismatch_raises():
    depth = np.full((H, W), 500.0, np.float32)
    with pytest.raises(ValueError):
        deproject_depth(depth, INTR, Pose.Identity(), rgb=np.zeros((H + 1, W, 3), np.uint8))
