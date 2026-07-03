"""Pluggable detector seam (lean).

A ``Detector`` consumes a grayscale head frame (+ optional arm config) and returns
named ``Detection`` objects -- a labeled ``Pose`` in the pelvis frame. ``detect()``
updates internal state from one frame; ``block_pose()`` returns the filtered
pelvis-frame pose for one label. ``SimStateDetector`` is the current implementation;
a model-based detector implements the same interface later (registered in the
factory's ``_DETECTORS`` map).

All poses are ``spatial.pose.Pose`` in the pelvis frame, constructed via the
injected ``transforms.Frames`` (CLAUDE.md rule 2 -- transforms.py owns frame math).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np

from g1_classical_manip.spatial.pose import Pose


@dataclass
class Detection:
    """One detected object in the pelvis frame."""
    pose: Pose
    label: str = "block"
    score: float = 1.0
    tag_id: Optional[int] = None
    extra: Dict = field(default_factory=dict)


class Detector(ABC):
    """A head-camera detector returning pelvis-frame object poses."""

    @abstractmethod
    def detect(self, gray: np.ndarray,
               q14: Optional[np.ndarray] = None) -> Dict[str, Detection]:
        """Process one grayscale frame; update internal state. Returns the
        currently-accepted detections keyed by semantic label."""

    @abstractmethod
    def block_pose(self, label: str = "block") -> Optional[Pose]:
        """Filtered pelvis-frame pose for one label, or None if not seen."""
