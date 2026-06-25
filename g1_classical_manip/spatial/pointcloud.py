"""Point-cloud type (numpy-only) — the shared geometry vocabulary between perception
(which produces clouds) and the grasp layer (which consumes them), exactly as ``Pose``
is the shared vocabulary between perception and motion.

Pure data + pure geometry: no torch, no I/O, no frame *construction* (that stays in
``perception/transforms.py``). ``transformed`` only *applies* an existing ``Pose``.
Points are ``(N, 3)`` float32; ``frame`` is a string label (e.g. ``"pelvis"``) asserted
at boundaries that care (the GraspGenX cloud must be sent in the planning frame).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g1_classical_manip.spatial.pose import Pose


@dataclass(frozen=True, eq=False)
class PointCloud:
    points: np.ndarray            # (N, 3) float32
    frame: str = "pelvis"

    def __post_init__(self):
        pts = np.asarray(self.points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must be (N, 3); got {pts.shape}")
        object.__setattr__(self, "points", np.ascontiguousarray(pts))

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    def is_empty(self) -> bool:
        return self.n == 0

    def transformed(self, pose: Pose, frame: str) -> "PointCloud":
        """Apply an EXISTING rigid transform to every point: ``p' = R @ p + t``.
        ``pose`` maps THIS cloud's frame into ``frame`` (e.g. ``T_pelvis_camera`` maps a
        camera-frame cloud into ``"pelvis"``). Does not construct a transform."""
        R, t = pose.rotation, pose.translation
        return PointCloud((self.points @ R.T + t).astype(np.float32), frame=frame)

    def voxel_downsampled(self, voxel_m: float) -> "PointCloud":
        """Collapse points to one per-voxel CENTROID on a ``voxel_m`` grid (numpy; no
        open3d). No-op for an empty cloud or a non-positive voxel size."""
        if voxel_m is None or voxel_m <= 0 or self.is_empty():
            return self
        keys = np.floor(self.points / voxel_m).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        m = int(inv.max()) + 1
        sums = np.zeros((m, 3), np.float64)
        np.add.at(sums, inv, self.points.astype(np.float64))
        counts = np.bincount(inv, minlength=m).reshape(-1, 1)
        return PointCloud((sums / counts).astype(np.float32), frame=self.frame)

    def cropped_box(self, center, half_extent) -> "PointCloud":
        """Keep points within an axis-aligned box: ``|p - center| <= half_extent`` per
        axis (``half_extent`` may be a scalar or a 3-vector). Same frame."""
        c = np.asarray(center, dtype=np.float32).reshape(3)
        h = np.broadcast_to(np.asarray(half_extent, dtype=np.float32), (3,))
        mask = np.all(np.abs(self.points - c) <= h, axis=1)
        return PointCloud(self.points[mask], frame=self.frame)
