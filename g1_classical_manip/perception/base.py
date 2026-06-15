"""Pluggable vision-model seam.

A ``VisionModel`` consumes a dict of RGB frames (keyed by camera name) and emits
a ``PerceptionOutput`` -- an SE3-centric result in the PELVIS frame. AprilTag is
the first implementation (see ``apriltag_block.AprilTagVisionModel``); future
6-DoF pose estimators and learned grasp models implement the same interface and
register in ``perception/registry.py``.

All poses are ``pin.SE3`` in the pelvis frame and MUST be constructed via the
injected ``Frames`` object (CLAUDE.md rule 2 -- transforms.py owns frame math).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import numpy as np
import pinocchio as pin


@dataclass
class PoseEstimate:
    """One named object pose in the pelvis frame (AprilTag tag, 6-DoF pose est.)."""
    label: str
    T_pelvis_object: pin.SE3
    score: float = 1.0                      # confidence / decision_margin
    source_cam: str = "head"
    extra: Dict = field(default_factory=dict)


@dataclass
class GraspCandidate:
    """One ranked grasp pose in the pelvis frame (learned grasp / affordance)."""
    T_pelvis_grasp: pin.SE3
    score: float
    label: str = ""                         # object this grasp targets
    width: Optional[float] = None           # optional gripper opening (m)
    source_cam: str = "head"
    extra: Dict = field(default_factory=dict)


@dataclass
class PerceptionOutput:
    """SE3-centric output. The same shape serves AprilTag (one pose), multi-object
    6-DoF pose estimation (N poses), and grasp prediction (ranked candidates)."""
    model: str
    stamp: float
    poses: List[PoseEstimate] = field(default_factory=list)
    grasps: List[GraspCandidate] = field(default_factory=list)
    source_cams: List[str] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)

    def best_pose(self, label: Optional[str] = None) -> Optional[PoseEstimate]:
        cands = [p for p in self.poses if label is None or p.label == label]
        return max(cands, key=lambda p: p.score, default=None)

    def best_grasp(self, label: Optional[str] = None) -> Optional[GraspCandidate]:
        cands = [g for g in self.grasps if label is None or g.label == label]
        return max(cands, key=lambda g: g.score, default=None)


@dataclass
class CameraFrame:
    """Reserved frame container so a later depth/intrinsics add is additive, not a
    signature change. The pipeline passes bare RGB ndarrays today (RGB-only scope);
    ``frames`` may become ``Dict[str, CameraFrame]`` when depth is plumbed."""
    rgb: np.ndarray                         # (H, W, 3) uint8
    depth: Optional[np.ndarray] = None      # (H, W) uint16, aligned -- not used yet
    intrinsics: Optional[Dict] = None
    stamp: float = 0.0


class VisionModel(ABC):
    """A camera-consuming perception model. Subclasses set ``name`` and implement
    ``process``. ``cameras()`` declares which camera frames the model needs."""

    name: str = "vision_model"

    @abstractmethod
    def process(self, frames: Dict[str, np.ndarray],
                q14: Optional[np.ndarray] = None) -> PerceptionOutput:
        """frames: {cam_name -> (H, W, 3) uint8 RGB}. Returns a pelvis-frame
        SE3 result. Frame conversions go through the injected ``Frames`` only."""

    def cameras(self) -> List[str]:
        return ["head"]
