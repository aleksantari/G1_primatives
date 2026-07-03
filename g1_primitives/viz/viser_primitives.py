"""Viser drawing primitives for grasp visualization.

VENDORED from GraspGenX's ``graspgenx/utils/viser_utils.py`` (NVIDIA, Apache-2.0)
and trimmed to the functions this repo needs. Imports NOTHING from the
``graspgenx`` package on purpose: importing ``graspgenx`` triggers a multi-GB
checkpoint/asset auto-download and pulls torch -- we don't want either on the
robot box. Same convention as ``grasp/graspgenx_client.py``. Deps: viser,
trimesh, numpy (all already in the ``g1_curobo`` env via cuRobo).

Coordinate convention matches the model: grasp frame has +Z = approach into the
object, +X = closing direction. The gripper-shaped markers are built from the
gripper's sweep volume so they look like the demo (scripts/demo_object_pc.py).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh  # noqa: F401  (kept for type clarity; meshes are trimesh.Trimesh)
import viser
import viser.transforms as vtf


def is_rotation_matrix(M, tol: float = 1e-4) -> bool:
    """Check if matrix M is a valid rotation matrix."""
    I = np.identity(M.shape[0])
    return bool(
        np.linalg.norm(np.matmul(M, M.T) - I) < tol
        and np.abs(np.linalg.det(M) - 1) < tol
    )


def get_color_from_score(labels, use_255_scale: bool = False):
    """Convert score label(s) to RGB color(s) (red=low, green=high)."""
    scale = 255.0 if use_255_scale else 1.0
    if type(labels) in (np.float32, float):
        return scale * np.array([1 - labels, labels, 0])
    score = scale * np.stack(
        [np.ones(labels.shape[0]) - labels, labels, np.zeros(labels.shape[0])],
        axis=1,
    )
    return score.astype(np.int32)


def matrix_to_wxyz_position(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """4x4 homogeneous transform -> (wxyz quaternion, position)."""
    wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz
    return wxyz, T[:3, 3]


def create_visualizer(clear: bool = True, port: int = 8080) -> viser.ViserServer:
    """Create a viser server (web GUI at http://localhost:<port>)."""
    print(f"Starting viser server on http://localhost:{port}")
    server = viser.ViserServer(port=port)
    if clear:
        server.scene.reset()
    return server


def make_frame(
    vis: viser.ViserServer,
    name: str,
    h: float = 0.15,
    radius: float = 0.01,
    T: Optional[np.ndarray] = None,
):
    """Add an RGB coordinate triad."""
    if vis is None:
        return
    wxyz = (1.0, 0.0, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)
    if T is not None:
        if not is_rotation_matrix(T[:3, :3]):
            raise ValueError("viser_primitives: invalid rotation in transform T")
        wxyz, position = matrix_to_wxyz_position(T)
    vis.scene.add_frame(
        name, show_axes=True, axes_length=h, axes_radius=radius,
        wxyz=wxyz, position=position,
    )


def visualize_mesh(
    vis: viser.ViserServer,
    name: str,
    mesh,
    color: Optional[List[int]] = None,
    transform: Optional[np.ndarray] = None,
):
    """Visualize a trimesh.Trimesh (vertices+faces) at an optional transform."""
    if vis is None:
        return None
    if color is None:
        color = np.random.randint(low=0, high=256, size=3).tolist()
    if isinstance(color, np.ndarray):
        color = color.tolist()
    color_tuple = tuple(int(c) for c in color[:3])

    wxyz = (1.0, 0.0, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)
    if transform is not None:
        wxyz, position = matrix_to_wxyz_position(transform)

    return vis.scene.add_mesh_simple(
        name,
        vertices=mesh.vertices.astype(np.float32),
        faces=mesh.faces.astype(np.uint32),
        color=color_tuple,
        wxyz=wxyz,
        position=position,
    )


def add_collision_spheres(vis, centers, radii, color=(60, 200, 90), opacity: float = 0.4,
                          name: str = "collision_spheres") -> int:
    """Add cuRobo collision spheres -- ``centers (N,3)`` + ``radii (N,)`` in the pelvis frame -- to a
    viser server as ONE semi-transparent mesh (all spheres concatenated into a single scene node, so
    it stays cheap). Returns the count drawn (0 = nothing). Shared by GraspViz.show_collision_spheres
    and tools/check_world so a 'Start or End state in collision' is visible: a sphere inside the red ESDF
    voxels (world collision) or two spheres overlapping (self-collision)."""
    if vis is None:
        return 0
    c = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    r = np.asarray(radii, dtype=np.float32).reshape(-1)
    unit = trimesh.creation.icosphere(subdivisions=1, radius=1.0)      # low-poly; one per sphere
    parts = []
    for ci, ri in zip(c, r):
        if ri <= 0:
            continue
        m = unit.copy()
        m.apply_scale(float(ri))
        m.apply_translation(ci.astype(float))
        parts.append(m)
    if not parts:
        return 0
    mesh = trimesh.util.concatenate(parts)
    col = tuple(int(x) for x in color)
    try:                                   # transparent when the viser build supports opacity
        vis.scene.add_mesh_simple(
            name, vertices=mesh.vertices.astype(np.float32), faces=mesh.faces.astype(np.uint32),
            color=col, opacity=float(opacity), wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0))
    except TypeError:                      # older viser: no opacity kwarg -> opaque fallback
        visualize_mesh(vis, name, mesh, color=list(col))
    return len(parts)


def visualize_bbox(
    vis: viser.ViserServer,
    name: str,
    dims: np.ndarray,
    T: Optional[np.ndarray] = None,
    color: Optional[List[int]] = None,
):
    """Visualize an axis-aligned (or transformed) wireframe box of size ``dims``."""
    if vis is None:
        return
    if color is None:
        color = [255, 0, 0]
    if isinstance(color, np.ndarray):
        color = color.tolist()
    color_tuple = tuple(int(c) for c in color[:3])

    wxyz = (1.0, 0.0, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)
    if T is not None:
        wxyz, position = matrix_to_wxyz_position(T)
    if isinstance(dims, np.ndarray):
        dims = tuple(float(d) for d in dims)

    vis.scene.add_box(
        name, color=color_tuple, dimensions=dims, wireframe=True,
        wxyz=wxyz, position=position,
    )


def visualize_pointcloud(
    vis: viser.ViserServer,
    name: str,
    pc: np.ndarray,
    color: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    size: float = 0.01,
):
    """Visualize an (N,3) point cloud, optional per-point or single color."""
    if vis is None:
        return
    if pc.ndim == 3:
        pc = pc.reshape(-1, pc.shape[-1])
    if pc.shape[-1] != 3:
        pc = pc[:, :3]
    num_points = pc.shape[0]

    if color is not None:
        color = np.array(color)
        if color.ndim == 3:
            color = color.reshape(-1, color.shape[-1])
        if color.ndim == 1:
            color = np.tile(np.array(color).flatten()[:3], (num_points, 1))
        elif color.ndim == 2:
            if color.shape[-1] > 3:
                color = color[:, :3]
            if color.shape[0] != num_points:
                if color.shape[0] > num_points:
                    color = color[:num_points]
                else:
                    pad = np.tile(color[-1:], (num_points - color.shape[0], 1))
                    color = np.vstack([color, pad])
        if np.issubdtype(color.dtype, np.floating):
            color = np.clip(color * 255.0, 0, 255).astype(np.uint8)
        else:
            color = np.clip(color, 0, 255).astype(np.uint8)
    else:
        color = np.full((num_points, 3), 255, dtype=np.uint8)

    wxyz = (1.0, 0.0, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)
    if transform is not None:
        wxyz, position = matrix_to_wxyz_position(transform)

    vis.scene.add_point_cloud(
        name, points=pc.astype(np.float32), colors=color, point_size=size,
        wxyz=wxyz, position=position,
    )


def create_gripper_control_points_for_viz(
    width: float, depth: float, height: float = 0.0
) -> List[np.ndarray]:
    """Canonical parallel-jaw control-point path (fallback when no sweep volume)."""
    hw = width / 2
    hh = height / 2
    left_tip = np.array([-hw, 0, depth, 1])
    right_tip = np.array([hw, 0, depth, 1])
    left_mid = np.array([-hw, 0, depth * 0.6, 1])
    right_mid = np.array([hw, 0, depth * 0.6, 1])
    base = np.array([0, 0, 0, 1])
    mid_point = np.array([0, -hh, depth, 1])
    return [np.array([left_mid, left_tip, mid_point, base, mid_point, right_tip, right_mid])]


def generate_control_points_from_sweep_volume(sweep_volume: Dict) -> List[np.ndarray]:
    """Gripper-shaped control-point path from a sweep volume {extents(3), offset(3)}."""
    sv_extents = sweep_volume["extents"]
    f = sweep_volume["offset"][2]
    w, _d, h = sv_extents[0], sv_extents[1], sv_extents[2]
    control_points = np.array(
        [
            [w / 2, 0, h / 2 + f, 1],
            [w / 2, 0, -h / 2 + f, 1],
            [0, 0, -h / 2 + f, 1],
            [0, 0, 0, 1],
            [0, 0, -h / 2 + f, 1],
            [-w / 2, 0, -h / 2 + f, 1],
            [-w / 2, 0, h / 2 + f, 1],
        ]
    )
    return [control_points]


def visualize_x_grasp(
    vis: viser.ViserServer,
    name: str,
    transform: np.ndarray,
    color: List[int] = [255, 0, 0],
    gripper_info=None,
    width: float = None,
    depth: float = None,
    height: float = 0.0,
    sweep_volume: Dict = None,
    linewidth: float = 2.0,
) -> list:
    """Draw a gripper-shaped grasp marker at ``transform``.

    Prefers ``gripper_info.sweep_volume`` (a 6-vec [extents(3), offset(3)]) so the
    marker matches the real gripper; falls back to a generic parallel jaw.
    Returns the list of viser line-segment handles (for visibility toggling).
    """
    if vis is None:
        return []

    if (
        gripper_info is not None
        and getattr(gripper_info, "sweep_volume", None) is not None
    ):
        sv = np.asarray(gripper_info.sweep_volume)
        sv_dict = {"extents": sv[:3].tolist(), "offset": sv[3:].tolist()}
        grasp_vertices = generate_control_points_from_sweep_volume(sv_dict)
    elif sweep_volume is not None:
        grasp_vertices = generate_control_points_from_sweep_volume(sweep_volume)
    elif width is not None and depth is not None:
        grasp_vertices = create_gripper_control_points_for_viz(width, depth, height)
    else:
        grasp_vertices = create_gripper_control_points_for_viz(0.08, 0.05, 0.0)

    if isinstance(color, np.ndarray):
        color = color.tolist()
    color_tuple = tuple(int(c) for c in color[:3])

    wxyz, position = matrix_to_wxyz_position(transform.astype(float))

    handles = []
    for i, ctrl_pts in enumerate(grasp_vertices):
        ctrl_pts = np.array(ctrl_pts, dtype=np.float32)
        if ctrl_pts.ndim == 1:
            continue
        if ctrl_pts.shape[0] == 4 and ctrl_pts.shape[1] != 4:
            ctrl_pts = ctrl_pts.T
        points_3d = ctrl_pts[:, :3]
        num_points = points_3d.shape[0]
        if num_points < 2:
            continue
        segments = np.zeros((num_points - 1, 2, 3), dtype=np.float32)
        for j in range(num_points - 1):
            segments[j, 0, :] = points_3d[j]
            segments[j, 1, :] = points_3d[j + 1]
        handle = vis.scene.add_line_segments(
            f"{name}/{i}", points=segments, colors=color_tuple,
            line_width=linewidth, wxyz=wxyz, position=position,
        )
        handles.append(handle)
    return handles
