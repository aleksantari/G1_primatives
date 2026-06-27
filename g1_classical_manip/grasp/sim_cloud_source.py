"""Ground-truth grasp source for the Isaac sim (no camera / depth / SAM3).

Builds a synthetic object point cloud from the LIVE `rt/sim_state` block pose (via a
`SimStateDetector`) instead of deprojecting a depth frame, then runs the same GraspGenX ->
tool-transform path as the real source. Lets us validate the grasp->motion half (the
EMPIRICAL `wristyaw_grasp_rpy`, candidate planning, collision-free 6-DoF execution) in sim
before the robot. GraspGenX is trained on simulated clouds, so a GT cube cloud is
in-distribution; the cloud is object-only by construction (no table). See the sim plan.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from g1_classical_manip.spatial.pose import rpy_to_matrix
from g1_classical_manip.spatial.pointcloud import PointCloud
from g1_classical_manip.perception.depth import colorize_from_image
from g1_classical_manip.grasp.base import GraspSource, GraspCandidate
from g1_classical_manip.grasp.tool_transform import candidates_from_grasps


def sample_cube(edge_m: float, n_points: int):
    """~`n_points` points on a cube surface (edge `edge_m`, centered at the origin) in the
    OBJECT frame, with per-point outward face normals. Returns ``(pts (M,3), normals (M,3))``
    float32. Analytic (no mesh / Isaac) -- the red block is a known cube."""
    h = edge_m / 2.0
    side = max(2, int(round((max(6, n_points) / 6) ** 0.5)))   # grid points per face edge
    g = np.linspace(-h, h, side)
    a, b = (x.ravel() for x in np.meshgrid(g, g))
    o = np.ones_like(a)
    faces = [
        (np.stack([o * h, a, b], 1), (1.0, 0.0, 0.0)),    # +x
        (np.stack([-o * h, a, b], 1), (-1.0, 0.0, 0.0)),  # -x
        (np.stack([a, o * h, b], 1), (0.0, 1.0, 0.0)),    # +y
        (np.stack([a, -o * h, b], 1), (0.0, -1.0, 0.0)),  # -y
        (np.stack([a, b, o * h], 1), (0.0, 0.0, 1.0)),    # +z
        (np.stack([a, b, -o * h], 1), (0.0, 0.0, -1.0)),  # -z
    ]
    pts = np.concatenate([f for f, _ in faces], 0).astype(np.float32)
    normals = np.concatenate([np.tile(n, (len(f), 1)) for f, n in faces], 0).astype(np.float32)
    return pts, normals


class SimCloudGraspSource(GraspSource):
    def __init__(self, frames, pose_source, client_factory, gcfg: dict, camera_cfg=None, viz=None):
        self.frames = frames
        self.pose_source = pose_source                  # SimStateDetector (.block_pose -> pelvis Pose)
        self.client_factory = client_factory            # () -> GraspGenXClient (ctx manager)
        self.gcfg = gcfg
        self.intrinsics = (camera_cfg or {}).get("intrinsics", {})   # sim cam K (for color)
        self.palm_offset_xyz = np.asarray(gcfg["palm_offset_xyz"], float)
        self.R_wristyaw_grasp = rpy_to_matrix(*gcfg["wristyaw_grasp_rpy"])
        sc = gcfg.get("sim_cloud", {}) or {}
        self.edge_m = float(sc.get("object_size_m", 0.06))
        self.n_points = int(sc.get("n_points", 2000))
        self.camera_facing_cull = bool(sc.get("camera_facing_cull", True))
        self.color_fallback = tuple(sc.get("color_fallback", (128, 128, 128)))
        self.viz = viz

    @staticmethod
    def _grab_rgb(cam, timeout_s: float = 2.0):
        """Best-effort sim RGB (the SUB may be cold on the first call). None if unavailable."""
        if cam is None:
            return None
        t0, rgb = time.time(), cam.get_rgb_frame()
        while rgb is None and time.time() - t0 < timeout_s:
            time.sleep(0.05)
            rgb = cam.get_rgb_frame()
        return rgb

    def grasps(self, robot, side: str, target: str) -> List[GraspCandidate]:
        pose = self.pose_source.block_pose(target)      # pelvis-frame object pose (live), or None
        if pose is None:
            return []
        pts, normals = sample_cube(self.edge_m, self.n_points)
        R, t = pose.rotation, pose.translation
        pts_pelvis = pts @ R.T + t                       # object -> pelvis
        T_pc = self.frames.T_pelvis_camera(None)         # head cam optical pose in pelvis
        if self.camera_facing_cull:                      # keep only faces visible to the head cam
            cam = np.asarray(T_pc.translation, float)
            facing = np.einsum("ij,ij->i", normals @ R.T, cam[None, :] - pts_pelvis) > 0.0
            if facing.any():
                pts_pelvis = pts_pelvis[facing]
        # color the cube from the live sim RGB (viz-only; XYZ unaffected). Best-effort: a cold
        # SUB or off-image points fall back to color_fallback rather than blocking.
        colors = None
        rgb = self._grab_rgb(getattr(robot, "camera", None))
        if rgb is not None and self.intrinsics:
            colors = colorize_from_image(pts_pelvis, rgb, self.intrinsics, T_pc, self.color_fallback)
        cloud = PointCloud(pts_pelvis.astype(np.float32), frame="pelvis", colors=colors)
        if self.gcfg.get("voxel_m"):
            cloud = cloud.voxel_downsampled(self.gcfg["voxel_m"])
        if cloud.is_empty():
            return []

        with self.client_factory() as client:
            grasps, conf = client.infer(
                cloud.points, gripper_name=self.gcfg.get("gripper_name", "unitree_g1"),
                num_grasps=int(self.gcfg.get("num_grasps", 200)),
                grasp_threshold=float(self.gcfg.get("grasp_threshold", -1.0)),
                topk_num_grasps=int(self.gcfg.get("topk", 100)))
        grasps = np.asarray(grasps, dtype=np.float32)
        conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        if min(grasps.shape[0], conf.shape[0]) == 0:
            return []
        if self.viz is not None:                         # cloud + all grasps (pelvis frame)
            self.viz.show_candidates(cloud.points, grasps, conf, colors=cloud.colors)
        return candidates_from_grasps(grasps, conf, side,
                                      self.palm_offset_xyz, self.R_wristyaw_grasp)
