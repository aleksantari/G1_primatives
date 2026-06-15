"""PerceptionHub: the multi-model perception runtime.

Builds the ``VisionModel``s selected in ``perception.yaml: models:``, runs them
on a dict of RGB frames each tick, and stores the latest ``PerceptionOutput`` per
model. It re-exposes the legacy AprilTag surface (``block_pose``/``detect``/
``detector``/``median_frames``) by delegating to the apriltag model, so the FSM
(``tasks/pick_place_handover.py``) and hardware scripts call ``robot.perception.*``
unchanged.

A per-model ``min_period_s`` throttle lets heavy models (a learned 6-DoF/grasp
net) run slower than the ~50 Hz frame loop without stalling AprilTag. Execution is
synchronous for now; an async worker per slow model is a reserved extension.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Set

import numpy as np

from g1_classical_manip.perception.base import PerceptionOutput
from g1_classical_manip.perception.registry import build_vision_model

_DEFAULT_MODELS = [{"name": "apriltag", "camera": "head"}]


class PerceptionHub:
    def __init__(self, frames, perception_cfg: dict, cameras_cfg: Optional[dict] = None,
                 camera_cfg: Optional[dict] = None, clock=None):
        self.frames = frames
        self.cameras_cfg = cameras_cfg or {}
        self.camera_cfg = camera_cfg or {}
        self._clock = clock or time.monotonic

        specs = perception_cfg.get("models") or _DEFAULT_MODELS
        self.models: Dict[str, object] = {}
        self._min_period: Dict[str, float] = {}
        self._next_due: Dict[str, float] = {}
        for spec in specs:
            cam_cfg = self._camera_cfg_for(spec.get("camera", "head"))
            m = build_vision_model(spec, frames, perception_cfg, cam_cfg,
                                   clock=self._clock)
            self.models[m.name] = m
            self._min_period[m.name] = float(spec.get("min_period_s", 0.0))
            self._next_due[m.name] = 0.0
        self._latest: Dict[str, PerceptionOutput] = {}

    def _camera_cfg_for(self, cam_name: str) -> dict:
        cams = self.cameras_cfg.get("cameras", self.cameras_cfg)
        if cam_name in cams:
            return cams[cam_name]
        return self.camera_cfg          # legacy fallback (camera.yaml head block)

    def required_cameras(self) -> Set[str]:
        out: Set[str] = set()
        for m in self.models.values():
            out.update(m.cameras())
        return out

    def process(self, frames: Dict[str, np.ndarray], q14=None) -> Dict[str, PerceptionOutput]:
        """Run each due model on the frame dict; store + return latest per model."""
        now = self._clock()
        for name, m in self.models.items():
            if now < self._next_due[name]:
                continue
            self._latest[name] = m.process(frames, q14)
            self._next_due[name] = now + self._min_period[name]
        return self._latest

    def latest(self, model: str) -> Optional[PerceptionOutput]:
        return self._latest.get(model)

    # ----- legacy AprilTag surface (delegate to the apriltag model) -----
    @property
    def _apriltag(self):
        return self.models.get("apriltag")

    def block_pose(self, tag_id: int):
        m = self._apriltag
        return m.block_pose(tag_id) if m is not None else None

    def detect(self, image: np.ndarray, q14=None):
        m = self._apriltag
        return m.detect(image, q14=q14) if m is not None else {}

    @property
    def detector(self):
        m = self._apriltag
        return m.detector if m is not None else None

    @property
    def median_frames(self) -> int:
        m = self._apriltag
        return m.median_frames if m is not None else 0
