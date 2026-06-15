"""Vision-model registry (mirrors unitree_lerobot make_robot's ARM_CONFIG/EE_CONFIG
pattern). ``perception.yaml: models:`` selects which ``VisionModel``s the
``PerceptionHub`` builds; add a new CV model by adding one entry here."""
from __future__ import annotations

from typing import Dict, Type

from g1_classical_manip.perception.base import VisionModel
from g1_classical_manip.perception.apriltag_block import AprilTagVisionModel

VISION_MODELS: Dict[str, Type[VisionModel]] = {
    "apriltag": AprilTagVisionModel,
}


def build_vision_model(spec: dict, frames, perception_cfg: dict, camera_cfg: dict,
                       clock=None) -> VisionModel:
    """spec: a model entry from ``perception.yaml: models:`` (e.g.
    ``{name: apriltag, camera: head}``). ``camera_cfg`` is the resolved per-camera
    config block for that model's camera (carries ``intrinsics``)."""
    name = spec.get("name")
    if name not in VISION_MODELS:
        raise ValueError(
            f"unknown vision model: {name!r} (have {list(VISION_MODELS)})")
    return VISION_MODELS[name](frames, perception_cfg, camera_cfg,
                               clock=clock, spec=spec)
