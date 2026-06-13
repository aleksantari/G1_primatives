"""AprilTag block detection: image -> T_cam_tag -> T_pelvis_block.

Pose assembly (the frame math) is separated from the detector call so it can be
unit-tested offline without a camera; only ``detect()`` needs a live frame.
"""
from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import pinocchio as pin

from g1_classical_manip.perception.transforms import Frames, from_xyz_rpy, se3


@dataclass
class TagReading:
    tag_id: int
    T_pelvis_block: pin.SE3
    decision_margin: float
    stamp: float


def _se3_median(poses: List[pin.SE3]) -> pin.SE3:
    """Componentwise-median translation; rotation taken from the sample nearest
    that median (robust, avoids quaternion-averaging pitfalls)."""
    T = np.array([p.translation for p in poses])
    med = np.median(T, axis=0)
    k = int(np.argmin(np.linalg.norm(T - med, axis=1)))
    return pin.SE3(poses[k].rotation.copy(), med)


class AprilTagBlockDetector:
    def __init__(self, frames: Frames, perception_cfg: dict, camera_cfg: dict,
                 clock=None):
        import pupil_apriltags
        self.frames = frames
        tag = perception_cfg.get("tag", {})
        self.family = tag.get("family", "tag36h11")
        self.tag_size = float(tag.get("size_m", 0.03))
        ttb = tag.get("tag_to_block", {"xyz": [0, 0, 0], "rpy": [0, 0, 0]})
        self.tag_to_block = from_xyz_rpy(ttb["xyz"], ttb["rpy"])
        self.id_names = {int(k): v for k, v in (tag.get("ids", {}) or {}).items()}

        flt = perception_cfg.get("filter", {})
        self.median_frames = int(flt.get("median_frames", 5))
        self.max_stale_s = float(flt.get("max_stale_s", 0.5))
        self.min_decision_margin = float(flt.get("min_decision_margin", 30.0))

        k = camera_cfg.get("intrinsics", {})
        self.camera_params = (k.get("fx", 0.0), k.get("fy", 0.0),
                              k.get("cx", 0.0), k.get("cy", 0.0))

        self.detector = pupil_apriltags.Detector(families=self.family)
        self._clock = clock or __import__("time").monotonic
        self._hist: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.median_frames))

    # ----- frame math (offline-testable) -----
    def cam_tag_se3(self, pose_R, pose_t) -> pin.SE3:
        return pin.SE3(np.asarray(pose_R, float).reshape(3, 3),
                       np.asarray(pose_t, float).reshape(3))

    def pelvis_block_se3(self, T_cam_tag: pin.SE3, q14=None) -> pin.SE3:
        return self.frames.T_pelvis_block(T_cam_tag, self.tag_to_block, q14)

    def ingest_detection(self, tag_id: int, pose_R, pose_t,
                         decision_margin: float, q14=None) -> Optional[TagReading]:
        if decision_margin < self.min_decision_margin:
            return None
        T_cam_tag = self.cam_tag_se3(pose_R, pose_t)
        T_pelvis_block = self.pelvis_block_se3(T_cam_tag, q14)
        r = TagReading(tag_id, T_pelvis_block, float(decision_margin),
                       self._clock())
        self._hist[tag_id].append(r)
        return r

    # ----- camera path (hardware-gated) -----
    def detect(self, gray_image: np.ndarray, q14=None) -> Dict[int, TagReading]:
        """Detect tags in a grayscale frame; update history. Returns the latest
        accepted reading per tag id."""
        dets = self.detector.detect(
            gray_image, estimate_tag_pose=True,
            camera_params=self.camera_params, tag_size=self.tag_size)
        out: Dict[int, TagReading] = {}
        for d in dets:
            if self.id_names and d.tag_id not in self.id_names:
                continue
            r = self.ingest_detection(d.tag_id, d.pose_R, d.pose_t,
                                      d.decision_margin, q14)
            if r is not None:
                out[d.tag_id] = r
        return out

    def block_pose(self, tag_id: int) -> Optional[pin.SE3]:
        """Median-filtered, staleness-gated pelvis-frame block pose, or None."""
        hist = self._hist.get(tag_id)
        if not hist:
            return None
        now = self._clock()
        fresh = [r for r in hist if now - r.stamp <= self.max_stale_s]
        if len(fresh) < max(1, self.median_frames // 2):
            return None
        return _se3_median([r.T_pelvis_block for r in fresh])

    def name_of(self, tag_id: int) -> str:
        return self.id_names.get(tag_id, str(tag_id))
