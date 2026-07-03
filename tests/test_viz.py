"""Gripper-geometry loader for the grasp viz (no viser / no GUI -- the GUI itself is a
manual check). Anchored on the repo root so it's cwd-independent."""
import os

import numpy as np

from g1_primitives.viz.gripper_geom import load_gripper_geom, GripperGeom

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(_REPO_ROOT, "assets", "grippers")


def test_load_unitree_g1_geom():
    geom = load_gripper_geom(ASSETS, "unitree_g1")
    assert isinstance(geom, GripperGeom) and geom.name == "unitree_g1"
    # sweep_volume = [extents(3), offset(3)] from config.json
    assert geom.sweep_volume is not None
    assert np.asarray(geom.sweep_volume).shape == (6,)
    # coll_mesh.obj ships with the asset -> mesh overlay available
    assert geom.has_mesh and geom.mesh is not None and len(geom.mesh.vertices) > 0


def test_missing_dir_degrades_gracefully():
    geom = load_gripper_geom("/no/such/asset/dir", "unitree_g1")
    assert isinstance(geom, GripperGeom)
    assert geom.sweep_volume is None and geom.has_mesh is False and geom.mesh is None


def test_unknown_gripper_degrades_gracefully():
    geom = load_gripper_geom(ASSETS, "no_such_gripper_xyz")
    assert geom.sweep_volume is None and geom.has_mesh is False
