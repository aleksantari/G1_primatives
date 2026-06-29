"""Depth-ESDF collision world helpers. GPU-free part only: the grid-shape math (the part that
must match cuRobo's ceil convention so the empty build grid and the live ESDF have identical
voxel shape). The Mapper/CUDA path is exercised by the operator-run sim validation, not here."""
from g1_classical_manip.motion.collision_world import esdf_grid_shape


def test_esdf_grid_shape_exact_multiple():
    # extent a multiple of voxel -> exact counts (no off-by-one vs the live ESDF)
    assert esdf_grid_shape([1.2, 1.2, 1.0], 0.02) == (60, 60, 50)
    assert esdf_grid_shape([1.0, 1.0, 1.0], 0.02) == (50, 50, 50)


def test_esdf_grid_shape_ceil():
    # non-multiple extent -> ceil per axis (cuRobo Mapper convention)
    assert esdf_grid_shape([0.45, 0.30, 0.11], 0.02) == (23, 15, 6)
