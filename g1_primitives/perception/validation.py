"""Perception validation: quantify how well the perceived pipeline output matches a
reference (in practice: sim ground truth from ``rt/sim_state``).

Four independent probes, each localizing a different failure:

  * ``cloud_error``            perceived object cloud vs the GT cloud (centroid offset,
                               chamfer distances, inlier fraction, AABB size delta) --
                               catches extrinsics/depth-scale errors and over/under-masking.
  * ``mask_iou`` (+ ``project_mask``)  the SAM3 mask vs the GT object projected into the
                               image -- separates SEGMENTATION error from extrinsics error
                               (a good IoU with a bad cloud_error points at T_pelvis_camera).
  * ``candidate_set_error``    where the grasp model thinks the object is (grasp origin +
                               fingertip_depth * approach) vs the GT center -- catches a
                               wrong grasp frame convention even when the cloud is perfect.
  * ``fingertip_contact_error`` FK of OUR Dex3 fingertips at a wrist goal + close preset vs
                               the GT center -- validates the grasp->wrist tool transform
                               with the real finger geometry (the check a gripper-mesh
                               overlay cannot do). Generalizes the pick example's FK check.

All GPU-free (numpy + scipy cKDTree + the URDF-parsed hand FK). The sim regression gate
is ``scripts/tools/validate_perception.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


def _points(cloud) -> np.ndarray:
    """Accept a PointCloud or a raw (N,3) array."""
    pts = getattr(cloud, "points", cloud)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N,3) points; got {pts.shape}")
    return pts


@dataclass
class CloudError:
    n_perceived: int
    n_gt: int
    centroid_delta_mm: np.ndarray            # (3,) perceived - GT centroid, mm
    centroid_mm: float                       # |centroid_delta|
    chamfer_mean_mm: float                   # symmetric mean nearest-neighbour distance
    chamfer_p95_mm: float                    # p95 of perceived->GT distances (tail outliers)
    inlier_frac: float                       # perceived points within inlier_tol of GT
    bbox_delta_mm: np.ndarray                # (3,) perceived AABB size - GT AABB size, mm

    def as_dict(self) -> Dict:
        return {"n_perceived": self.n_perceived, "n_gt": self.n_gt,
                "centroid_mm": round(self.centroid_mm, 2),
                "centroid_delta_mm": np.round(self.centroid_delta_mm, 2).tolist(),
                "chamfer_mean_mm": round(self.chamfer_mean_mm, 2),
                "chamfer_p95_mm": round(self.chamfer_p95_mm, 2),
                "inlier_frac": round(self.inlier_frac, 3),
                "bbox_delta_mm": np.round(self.bbox_delta_mm, 2).tolist()}


def cloud_error(perceived, gt, inlier_tol_m: float = 0.01) -> CloudError:
    """Compare a perceived object cloud against the GT cloud (same frame, meters).
    Chamfer is the symmetric mean of nearest-neighbour distances (cKDTree); the p95 is
    one-directional perceived->GT, so stray background points show up in the tail."""
    from scipy.spatial import cKDTree
    p, g = _points(perceived), _points(gt)
    if len(p) == 0 or len(g) == 0:
        raise ValueError(f"empty cloud (perceived {len(p)}, gt {len(g)})")
    d_pg = cKDTree(g).query(p)[0]            # perceived -> GT
    d_gp = cKDTree(p).query(g)[0]            # GT -> perceived (coverage)
    delta = (p.mean(0) - g.mean(0)) * 1000.0
    bbox = ((p.max(0) - p.min(0)) - (g.max(0) - g.min(0))) * 1000.0
    return CloudError(
        n_perceived=len(p), n_gt=len(g),
        centroid_delta_mm=delta, centroid_mm=float(np.linalg.norm(delta)),
        chamfer_mean_mm=float(0.5 * (d_pg.mean() + d_gp.mean()) * 1000.0),
        chamfer_p95_mm=float(np.percentile(d_pg, 95) * 1000.0),
        inlier_frac=float((d_pg < inlier_tol_m).mean()),
        bbox_delta_mm=bbox)


def project_mask(points_pelvis, intrinsics: Dict, T_pelvis_camera,
                 image_hw, dilate_px: int = 0) -> np.ndarray:
    """Project pelvis-frame points into the image -> an (H,W) bool mask of hit pixels.
    ``T_pelvis_camera`` is the camera OPTICAL pose in the pelvis frame (a Pose); points
    behind the camera are dropped. ``dilate_px`` grows the sparse point hits into a solid
    blob so it is comparable to a dense segmentation mask."""
    pts = _points(points_pelvis)
    inv = T_pelvis_camera.inverse()
    cam = pts @ inv.rotation.T + inv.translation        # pelvis -> camera optical
    z = cam[:, 2]
    ok = z > 1e-6
    u = intrinsics["fx"] * cam[ok, 0] / z[ok] + intrinsics["cx"]
    v = intrinsics["fy"] * cam[ok, 1] / z[ok] + intrinsics["cy"]
    h, w = int(image_hw[0]), int(image_hw[1])
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    keep = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    mask = np.zeros((h, w), bool)
    mask[vi[keep], ui[keep]] = True
    if dilate_px > 0 and mask.any():
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=int(dilate_px))
    return mask


def mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Intersection-over-union of two (H,W) bool masks (0.0 if either is None/empty)."""
    if a is None or b is None:
        return 0.0
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def candidate_set_error(candidates, gt_center, fingertip_depth_m: float = 0.07) -> Dict:
    """Where the grasp model put the object vs the GT center. GraspGenX's convention
    places the object at (grasp origin + fingertip_depth * approach(+Z)); each
    candidate's ``grasp_pose`` (the raw model frame) gives that implied object point.
    Returns mm distances for the top-1 (highest-confidence) and the best candidate --
    a wrong grasp-frame convention shows up here even when the cloud is perfect."""
    gt = np.asarray(gt_center, float).reshape(3)
    dists = []
    for c in candidates:
        gp = getattr(c, "grasp_pose", None)
        if gp is None:
            continue
        implied = gp.translation + fingertip_depth_m * gp.rotation[:, 2]
        dists.append(float(np.linalg.norm(implied - gt) * 1000.0))
    if not dists:
        return {"n": 0, "top1_mm": None, "best_mm": None, "mean_mm": None}
    return {"n": len(dists), "top1_mm": round(dists[0], 1),
            "best_mm": round(min(dists), 1),
            "mean_mm": round(float(np.mean(dists)), 1)}


def fingertip_contact_error(side: str, wrist_goal, object_center, q7_close) -> Dict:
    """FK OUR Dex3 fingertips at ``wrist_goal`` (pelvis frame) + the given close preset
    and report how close the grasp contact lands to the TRUE object center (mm).
    Validates the grasp pick + the derived tool transform + our real finger geometry
    against GT -- NOT against the grasp pose, which our contact hits by construction."""
    from g1_primitives.ee.hand_kinematics import Dex3Kinematics
    from g1_primitives.spatial.pose import Pose
    q7 = np.asarray(q7_close, float)
    gt = np.asarray(object_center, float).reshape(3)
    kin = Dex3Kinematics()
    tips = kin.fingertips(side, q7)
    to_pelvis = lambda p: (wrist_goal * Pose(np.eye(3), p)).translation
    contact = to_pelvis(kin.contact_point(side, q7))
    fingers_mid = 0.5 * (to_pelvis(tips["index"]) + to_pelvis(tips["middle"]))
    return {"contact_mm": round(float(np.linalg.norm(contact - gt)) * 1000.0, 1),
            "thumb_mm": round(float(np.linalg.norm(to_pelvis(tips["thumb"]) - gt)) * 1000.0, 1),
            "fingers_mm": round(float(np.linalg.norm(fingers_mid - gt)) * 1000.0, 1)}
