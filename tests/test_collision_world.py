"""Depth-ESDF collision world helpers. GPU-free part only: the grid-shape math (the part that
must match cuRobo's ceil convention so the empty build grid and the live ESDF have identical
voxel shape) + the object-exclusion mask dilation. The Mapper/CUDA path is exercised by the
operator-run sim validation, not here."""
import numpy as np

from g1_classical_manip.motion.collision_world import esdf_grid_shape, dilate_mask


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
