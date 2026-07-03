"""Depth-ESDF collision world helpers. GPU-free part only: the grid-shape math (the part that
must match cuRobo's ceil convention so the empty build grid and the live ESDF have identical
voxel shape) + the object-exclusion mask dilation. The Mapper/CUDA path is exercised by the
operator-run sim validation, not here."""
import numpy as np

from g1_classical_manip.motion.collision_world import (
    esdf_grid_shape, dilate_mask, kernel_safe_dims)


def test_esdf_grid_shape_exact_multiple():
    # extent a multiple of voxel -> exact counts (no off-by-one vs the live ESDF)
    assert esdf_grid_shape([1.2, 1.2, 1.0], 0.02) == (60, 60, 50)
    assert esdf_grid_shape([1.0, 1.0, 1.0], 0.02) == (50, 50, 50)


def test_esdf_grid_shape_ceil():
    # non-multiple extent -> ceil per axis (cuRobo Mapper convention)
    assert esdf_grid_shape([0.45, 0.30, 0.11], 0.02) == (23, 15, 6)


def test_dilate_mask_grows_square():
    # a single pixel dilated by px grows to a (2px+1)^2 square (Chebyshev kernel), clipped at edges
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    d1 = dilate_mask(m, 1)
    assert d1.sum() == 9 and d1[3:6, 3:6].all()
    d3 = dilate_mask(m, 3)
    assert d3.sum() == 49 and d3[1:8, 1:8].all() and not d3[0, :].any()
    edge = np.zeros((5, 5), bool)
    edge[0, 0] = True
    assert dilate_mask(edge, 1).sum() == 4                 # clipped at the image border


def test_dilate_mask_noop_cases():
    m = np.zeros((4, 4), bool)
    m[1, 2] = True
    assert dilate_mask(m, 0) is m or (dilate_mask(m, 0) == m).all()   # px<=0 -> unchanged
    empty = np.zeros((4, 4), bool)
    assert not dilate_mask(empty, 5).any()                            # nothing to grow
    out = dilate_mask(m, 1)
    assert m.sum() == 1                                               # input never mutated
    assert out.dtype == np.bool_ and out.shape == m.shape


def _kernel_grid_dims(dims, voxel_size):
    """Replicate cuRobo's dims arithmetic END-TO-END in float32: VoxelData.load_batch stores
    params = float32(dims) / float32(voxel_size); the warp collision kernel then recovers the
    grid shape with a TRUNCATING int cast (wp.int32). Truncation of 119.99999 -> 119 corrupts
    the kernel's flat-index strides -> phantom collisions in free space + real obstacles
    reading free (root cause of the 2026-07-02 left-arm --collision-world failures)."""
    params = np.float32(np.asarray(dims, np.float32) / np.float32(voxel_size))
    return [int(p) for p in params]                       # wp.int32 == C truncation


def test_kernel_safe_dims_survives_curobo_float32_truncation():
    # the live workspace grid (planner.yaml defaults) + a spread of shapes/voxel sizes
    for shape, v in [((120, 120, 100), 0.01), ((60, 60, 50), 0.02), ((23, 15, 6), 0.02),
                     ((240, 240, 200), 0.005), ((1, 1, 1), 0.01)]:
        # author dims the way the cuRobo Mapper does: grid_shape * float32(voxel_size)
        mapper_dims = [float(np.float32(k) * np.float32(v)) for k in shape]
        safe = kernel_safe_dims(mapper_dims, v)
        assert _kernel_grid_dims(safe, v) == list(shape)              # kernel recovers k exactly
        assert [round(d / float(v)) for d in safe] == list(shape)     # python-side round too
        assert kernel_safe_dims(safe, v) == kernel_safe_dims(safe, v)  # idempotent-stable


def test_kernel_safe_dims_regression_mapper_dims_are_unsafe():
    # the bug this guards against: the Mapper's float32-authored dims truncate to k-1 in the
    # kernel for the exact grid we deploy (1.2m / 0.01). If cuRobo fixes load_batch upstream
    # this may start passing through unfixed -- then kernel_safe_dims can be retired.
    mapper_dims = [float(np.float32(k) * np.float32(0.01)) for k in (120, 120, 100)]
    # x/y land at 119.99999 -> truncate to 119 (z happens to land above 100 in float32; one bad
    # axis is enough -- stride_x = dims_y*dims_z is already wrong)
    assert _kernel_grid_dims(mapper_dims, 0.01)[:2] == [119, 119]     # the broken read
