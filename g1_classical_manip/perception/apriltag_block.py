"""AprilTag block detection: image -> T_cam_tag -> T_pelvis_block (pelvis frame).

Pose assembly (the frame math) is separated from the detector call so it can be
unit-tested offline without a camera or pupil_apriltags; only ``detect()`` needs a
live frame + the (lazily imported) ``pupil_apriltags`` detector.
"""
from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np

from g1_classical_manip.spatial.pose import Pose, from_xyz_rpy
from g1_classical_manip.perception.transforms import Frames
from g1_classical_manip.perception.base import Detector, Detection

try:
    import cv2 as _cv2
except Exception:
    _cv2 = None


def _to_gray(img: np.ndarray) -> np.ndarray:
    """RGB (H,W,3) -> grayscale (H,W); pass through if already single-channel."""
    a = np.asarray(img)
    if a.ndim == 2:
        return a
    if _cv2 is not None:
        return _cv2.cvtColor(a, _cv2.COLOR_RGB2GRAY)
    return a.mean(axis=-1).astype(np.uint8)


@dataclass
class _Reading:
    tag_id: int
    T_pelvis_block: Pose
    decision_margin: float
    stamp: float


def _pose_median(poses: List[Pose]) -> Pose:
    """Componentwise-median translation; rotation taken from the sample nearest
    that median (robust, avoids quaternion-averaging pitfalls)."""
    T = np.array([p.translation for p in poses])
    med = np.median(T, axis=0)
    k = int(np.argmin(np.linalg.norm(T - med, axis=1)))
    return Pose(poses[k].rotation.copy(), med)


class AprilTagDetector(Detector):
    """Detect tag36h11 markers, fuse to a median-filtered pelvis-frame block pose."""

    def __init__(self, frames: Frames, perception_cfg: dict, camera_cfg: dict, clock=None):
        self.frames = frames
        tag = perception_cfg.get("tag", {})
        self.family = tag.get("family", "tag36h11")
        self.tag_size = float(tag.get("size_m", 0.03))
        ttb = tag.get("tag_to_block", {"xyz": [0, 0, 0], "rpy": [0, 0, 0]})
        self.tag_to_block = from_xyz_rpy(ttb["xyz"], ttb["rpy"])
        self.id_names = {int(k): v for k, v in (tag.get("ids", {}) or {}).items()}
        self._name_to_id = {v: k for k, v in self.id_names.items()}

        flt = perception_cfg.get("filter", {})
        self.median_frames = int(flt.get("median_frames", 5))
        self.max_stale_s = float(flt.get("max_stale_s", 0.5))
        self.min_decision_margin = float(flt.get("min_decision_margin", 30.0))

        k = camera_cfg.get("intrinsics", {})
        self.camera_params = (k.get("fx", 0.0), k.get("fy", 0.0),
                              k.get("cx", 0.0), k.get("cy", 0.0))

        # pupil_apriltags params. quad_decimate=1.0 (no downsampling) so small tags
        # (the sim tag is ~30 px in a 640x480 frame) clear the decode threshold; the
        # library default 2.0 misses them.
        dp = perception_cfg.get("detector_params") or {}
        self.quad_decimate = float(dp.get("quad_decimate", 1.0))
        self.nthreads = int(dp.get("nthreads", 1))
        self._detector = None    # lazy pupil_apriltags.Detector (camera path only)
        self._clock = clock or __import__("time").monotonic
        self._hist: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.median_frames))

    # ----- frame math (offline-testable; no camera / pupil_apriltags) -----
    def cam_tag_pose(self, pose_R, pose_t) -> Pose:
        return Pose(np.asarray(pose_R, float).reshape(3, 3),
                    np.asarray(pose_t, float).reshape(3))

    def pelvis_block_pose(self, T_cam_tag: Pose, q14=None) -> Pose:
        return self.frames.T_pelvis_block(T_cam_tag, self.tag_to_block, q14)

    def ingest_detection(self, tag_id, pose_R, pose_t,
                         decision_margin: float, q14=None) -> Optional[_Reading]:
        if decision_margin < self.min_decision_margin:
            return None
        T_cam_tag = self.cam_tag_pose(pose_R, pose_t)
        r = _Reading(int(tag_id), self.pelvis_block_pose(T_cam_tag, q14),
                     float(decision_margin), self._clock())
        self._hist[int(tag_id)].append(r)
        return r

    def name_of(self, tag_id) -> str:
        return self.id_names.get(int(tag_id), str(tag_id))

    # ----- camera path -----
    def _ensure_detector(self):
        if self._detector is None:
            import pupil_apriltags
            self._detector = pupil_apriltags.Detector(
                families=self.family, quad_decimate=self.quad_decimate,
                nthreads=self.nthreads)
        return self._detector

    def detect(self, gray: np.ndarray, q14=None) -> Dict[str, Detection]:
        """Detect tags in a frame; update history. Returns accepted detections
        keyed by semantic label. No-op on a missing frame (no camera / not ready)."""
        if gray is None:
            return {}
        dets = self._ensure_detector().detect(
            _to_gray(gray), estimate_tag_pose=True,
            camera_params=self.camera_params, tag_size=self.tag_size)
        out: Dict[str, Detection] = {}
        for d in dets:
            if self.id_names and d.tag_id not in self.id_names:
                continue
            r = self.ingest_detection(d.tag_id, d.pose_R, d.pose_t,
                                      d.decision_margin, q14)
            if r is not None:
                label = self.name_of(d.tag_id)
                out[label] = Detection(pose=r.T_pelvis_block, label=label,
                                       score=r.decision_margin, tag_id=int(d.tag_id))
        return out

    def block_pose(self, label: str = "block") -> Optional[Pose]:
        """Median-filtered, staleness-gated pelvis-frame pose for a label, or None."""
        tag_id = self._name_to_id.get(label)
        if tag_id is None:
            try:                       # allow a numeric tag id as the label
                tag_id = int(label)
            except (TypeError, ValueError):
                return None
        hist = self._hist.get(tag_id)
        if not hist:
            return None
        now = self._clock()
        fresh = [r for r in hist if now - r.stamp <= self.max_stale_s]
        if len(fresh) < max(1, self.median_frames // 2):
            return None
        return _pose_median([r.T_pelvis_block for r in fresh])
