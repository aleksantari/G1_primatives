"""Point-cloud type (numpy-only) — the shared geometry vocabulary between perception
(which produces clouds) and the grasp layer (which consumes them), exactly as ``Pose``
is the shared vocabulary between perception and motion.

Pure data + pure geometry: no torch, no I/O, no frame *construction* (that stays in
``perception/frames.py``). ``transformed`` only *applies* an existing ``Pose``.
Points are ``(N, 3)`` float32; ``frame`` is a string label (e.g. ``"pelvis"``) asserted
at boundaries that care (the GraspGenX cloud must be sent in the planning frame).

``colors`` is an OPTIONAL ``(N, 3)`` uint8 RGB array, a *parallel* attribute kept in
lockstep with ``points`` (carried by ``transformed``, averaged by ``voxel_downsampled``,
masked by ``cropped_box``). It is purely for visualization — consumers that want XYZ
(e.g. GraspGenX) always read ``.points``, never the fused color, so color can never reach
the grasp model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from g1_primitives.spatial.pose import Pose


@dataclass(frozen=True, eq=False)
class PointCloud:
    points: np.ndarray                       # (N, 3) float32
    frame: str = "pelvis"
    colors: Optional[np.ndarray] = None      # (N, 3) uint8 RGB, optional (viz only)

    def __post_init__(self):
        pts = np.asarray(self.points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must be (N, 3); got {pts.shape}")
        object.__setattr__(self, "points", np.ascontiguousarray(pts))
        if self.colors is not None:
            cols = np.asarray(self.colors, dtype=np.uint8)
            if cols.shape != pts.shape:
                raise ValueError(f"colors must be (N, 3) matching points {pts.shape}; "
                                 f"got {cols.shape}")
            object.__setattr__(self, "colors", np.ascontiguousarray(cols))

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    def is_empty(self) -> bool:
        return self.n == 0

    def transformed(self, pose: Pose, frame: str) -> "PointCloud":
        """Apply an EXISTING rigid transform to every point: ``p' = R @ p + t``.
        ``pose`` maps THIS cloud's frame into ``frame`` (e.g. ``T_pelvis_camera`` maps a
        camera-frame cloud into ``"pelvis"``). Does not construct a transform. Color is
        unchanged by a rigid transform, so it rides along."""
        R, t = pose.rotation, pose.translation
        return PointCloud((self.points @ R.T + t).astype(np.float32), frame=frame,
                          colors=self.colors)

    def voxel_downsampled(self, voxel_m: float) -> "PointCloud":
        """Collapse points to one per-voxel CENTROID on a ``voxel_m`` grid (numpy; no
        open3d), averaging color per voxel in lockstep. No-op for an empty cloud or a
        non-positive voxel size."""
        if voxel_m is None or voxel_m <= 0 or self.is_empty():
            return self
        keys = np.floor(self.points / voxel_m).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        m = int(inv.max()) + 1
        counts = np.bincount(inv, minlength=m).reshape(-1, 1)
        sums = np.zeros((m, 3), np.float64)
        np.add.at(sums, inv, self.points.astype(np.float64))
        pts = (sums / counts).astype(np.float32)
        cols = None
        if self.colors is not None:
            csum = np.zeros((m, 3), np.float64)
            np.add.at(csum, inv, self.colors.astype(np.float64))
            cols = np.round(csum / counts).astype(np.uint8)
        return PointCloud(pts, frame=self.frame, colors=cols)

    def cropped_box(self, center, half_extent) -> "PointCloud":
        """Keep points within an axis-aligned box: ``|p - center| <= half_extent`` per
        axis (``half_extent`` may be a scalar or a 3-vector). Same frame; color masked
        with the same selection."""
        c = np.asarray(center, dtype=np.float32).reshape(3)
        h = np.broadcast_to(np.asarray(half_extent, dtype=np.float32), (3,))
        mask = np.all(np.abs(self.points - c) <= h, axis=1)
        cols = self.colors[mask] if self.colors is not None else None
        return PointCloud(self.points[mask], frame=self.frame, colors=cols)
