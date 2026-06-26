"""Torch-free loader for the gripper geometry the grasp viz needs.

Replicates ONLY the slice of GraspGenX's ``x_grippers.get_gripper_info`` that the
visualization uses -- the sweep volume (for gripper-shaped markers) and a mesh
(for the overlay) -- by parsing the gripper-description asset files directly
(``config.json`` + ``coll_mesh.obj`` / ``vis_mesh.obj``). Imports NOTHING from
``graspgenx`` (that would pull torch + the auto-download hook). Deps: json,
trimesh, numpy.

Asset layout (matches gripper_descriptions): ``<asset_dir>/<name>/config.json``
and a mesh file in the same dir. ``config.json`` carries
``sweep_volume.extents`` (3) + ``sweep_volume.offset`` (3); we concatenate them
into the 6-vec that ``visualize_x_grasp`` expects via ``gripper_info.sweep_volume``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh


@dataclass
class GripperGeom:
    """Minimal gripper geometry for visualization.

    ``sweep_volume`` is ``[extents(3), offset(3)]`` (the attribute name and shape
    that ``viser_primitives.visualize_x_grasp`` reads). ``mesh`` is the gripper
    surface in the canonical grasp frame, transformed by each grasp pose for the
    overlay. ``has_mesh`` is False when only the wireframe markers are available.
    """
    name: str
    sweep_volume: Optional[np.ndarray]
    mesh: Optional[trimesh.Trimesh]
    has_mesh: bool


def _load_mesh(asset_dir: str, name: str) -> Optional[trimesh.Trimesh]:
    """Prefer coll_mesh.obj, then vis_mesh.obj; return None if neither loads."""
    for fname in ("coll_mesh.obj", "vis_mesh.obj"):
        path = os.path.join(asset_dir, name, fname)
        if not os.path.exists(path):
            continue
        try:
            mesh = trimesh.load(path, force="mesh")
            if getattr(mesh, "vertices", None) is not None and len(mesh.vertices):
                return mesh
        except Exception as e:  # noqa: BLE001 - viz must never crash the grasp flow
            print(f"[gripper_geom] failed to load {path}: {e}")
    return None


def load_gripper_geom(asset_dir: str, name: str) -> GripperGeom:
    """Load sweep volume + mesh for ``name`` from ``<asset_dir>/<name>/``.

    Degrades gracefully: a missing/blank config yields ``sweep_volume=None``
    (markers fall back to a generic jaw) and a missing mesh yields
    ``has_mesh=False`` (markers only). Never raises on missing assets.
    """
    sweep_volume = None
    config_path = os.path.join(asset_dir, name, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            sv = config.get("sweep_volume", {})
            extents = sv.get("extents")
            offset = sv.get("offset")
            if extents is not None and offset is not None:
                sweep_volume = np.concatenate(
                    [np.asarray(extents, float), np.asarray(offset, float)]
                )
        except Exception as e:  # noqa: BLE001
            print(f"[gripper_geom] failed to parse {config_path}: {e}")
    else:
        print(
            f"[gripper_geom] no config.json for {name!r} under {asset_dir!r}; "
            "markers will use a generic gripper shape."
        )

    mesh = _load_mesh(asset_dir, name)
    return GripperGeom(
        name=name,
        sweep_volume=sweep_volume,
        mesh=mesh,
        has_mesh=mesh is not None,
    )
