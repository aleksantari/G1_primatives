"""Object segmentation seam: produce a 2D image MASK, applied to the depth map BEFORE
deproject (RGB and the head depth are pixel-aligned, so a mask over the RGB indexes the
depth grid 1:1). A grasp model wants ONE object's points, not the whole table.

``Segmenter.mask(rgb) -> Optional[(H,W) bool]``:
  * a True/False mask  -> deproject keeps only masked-in pixels (the object).
  * an all-False mask  -> "segmented, found nothing" -> empty cloud -> no grasps (loud).
  * ``None``           -> "no segmentation" -> the whole frame (NullSegmenter / sim).

Implementations: ``NullSegmenter`` (no-op), ``Sam3Segmenter`` (one SAM3 call with a fixed
configured prompt), and ``segment_gui.InteractiveSam3Segmenter`` (operator refines
text/box/points in a cv2 GUI; lives with the GUI so this module stays display-free). The
interactive one raises ``SegmentationAborted`` on abort so the caller returns no grasps
rather than silently grasping the whole scene.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class SegmentationAborted(Exception):
    """The operator aborted interactive segmentation (distinct from 'found nothing')."""


def _prompt_kwargs(prompt_cfg: Optional[dict]) -> dict:
    """A grasp.yaml default_prompt ({text|box|points(+point_labels)}) -> Sam3Client.segment
    kwargs (exactly one prompt)."""
    p = dict(prompt_cfg or {})
    if "text" in p:
        return {"text": p["text"]}
    if "box" in p:
        return {"box": p["box"]}
    if "points" in p:
        kw = {"points": p["points"]}
        if p.get("point_labels") is not None:
            kw["point_labels"] = p["point_labels"]
        return kw
    raise ValueError(f"default_prompt needs one of text/box/points; got {sorted(p)}")


class Segmenter(ABC):
    @abstractmethod
    def mask(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """``(H,W,3)`` RGB -> ``(H,W)`` bool object mask, all-False (found nothing), or
        None (no masking / whole frame)."""


class NullSegmenter(Segmenter):
    """No segmentation -> the whole frame becomes the cloud (sim / mode: null)."""

    def mask(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        return None


class Sam3Segmenter(Segmenter):
    """One SAM3 call with the configured default prompt; returns the top mask (or an
    all-False mask if the object was not found)."""

    def __init__(self, client_factory, prompt_cfg: dict, top_k: int = 1):
        self.client_factory = client_factory      # () -> Sam3Client (context manager)
        self.prompt_cfg = prompt_cfg
        self.top_k = int(top_k)

    def mask(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        kw = _prompt_kwargs(self.prompt_cfg)
        with self.client_factory() as c:
            masks, _scores, _labels = c.segment(rgb, top_k=self.top_k, **kw)
        if masks.shape[0] == 0:                    # found nothing -> empty (loud), NOT whole-frame
            return np.zeros(rgb.shape[:2], dtype=bool)
        return masks[0] > 0
