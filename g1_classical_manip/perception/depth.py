"""Depth -> point cloud (the RGB-D producer). Pinhole-unprojects a head-camera depth
map into a metric, pelvis-frame ``PointCloud``.

This APPLIES the camera extrinsic that ``perception/transforms.py`` owns — it never
constructs a frame conversion (CLAUDE.md rule 2). The caller passes the optical->pelvis
``Pose`` (``robot.frames.T_pelvis_camera(q14)``) and the camera intrinsics
(``camera_cfg["intrinsics"]``, the left-eye values matching the depth map). Output is in
METERS, pelvis frame, +Z up — exactly what a grasp model / point-cloud consumer wants.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from g1_classical_manip.spatial.pose import Pose
from g1_classical_manip.spatial.pointcloud import PointCloud


def deproject_depth(depth_mm: np.ndarray, intrinsics: dict, T_pelvis_camera: Pose,
                    voxel_m: Optional[float] = None,
                    z_min_m: float = 0.05, z_max_m: float = 2.0,
                    mask: Optional[np.ndarray] = None,
                    rgb: Optional[np.ndarray] = None) -> PointCloud:
    """``(H,W)`` float depth in MILLIMETERS (NaN/inf/0 = invalid) -> pelvis-frame
    ``PointCloud`` in meters. Drops invalid + out-of-[z_min,z_max] pixels, unprojects with
    the pinhole intrinsics ``{fx,fy,cx,cy}``, maps optical->pelvis via the passed-in
    ``T_pelvis_camera``, and (optionally) voxel-downsamples to ``voxel_m``. An optional
    ``mask`` ((H,W) bool, same shape as depth) ANDs into the validity gate so only masked-in
    pixels become points (segmentation applied pre-deproject). An optional ``rgb`` ((H,W,3)
    uint8, pixel-aligned with depth) attaches per-point colors for visualization — XYZ is
    unaffected, so consumers reading ``.points`` (e.g. GraspGenX) never see color."""
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W); got {depth.shape}")
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])

    z_mm = depth
    valid = np.isfinite(z_mm) & (z_mm > 0) & (z_mm >= z_min_m * 1e3) & (z_mm <= z_max_m * 1e3)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError(f"mask shape {mask.shape} != depth shape {depth.shape}")
        valid &= mask
    vs, us = np.nonzero(valid)                       # pixel rows (v) / cols (u)
    z = z_mm[vs, us] / 1000.0                        # mm -> m
    x = (us.astype(np.float32) - cx) / fx * z        # optical: +x right, +y down, +z fwd
    y = (vs.astype(np.float32) - cy) / fy * z
    pts_optical = np.stack([x, y, z], axis=1).astype(np.float32)   # (N,3), (0,3) if none

    colors = None
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[:2] != depth.shape:
            raise ValueError(f"rgb must be (H, W, 3) matching depth {depth.shape}; got {rgb.shape}")
        colors = rgb[vs, us][:, :3].astype(np.uint8)   # per-point RGB (N,3)

    cloud = PointCloud(pts_optical, frame="camera_optical", colors=colors).transformed(
        T_pelvis_camera, frame="pelvis")
    return cloud.voxel_downsampled(voxel_m) if voxel_m else cloud
