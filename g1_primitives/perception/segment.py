"""Object segmentation seam: produce a 2D image MASK, applied to the depth map BEFORE
deproject (RGB and the head depth are pixel-aligned, so a mask over the RGB indexes the
depth grid 1:1). A grasp model wants ONE object's points, not the whole table.

``Segmenter.mask(rgb, text=None) -> Optional[(H,W) bool]``:
  * a True/False mask  -> deproject keeps only masked-in pixels (the object).
  * an all-False mask  -> "segmented, found nothing" -> empty cloud -> no grasps (loud).
  * ``None``           -> "no segmentation" -> the whole frame (NullSegmenter / sim).

``text`` is the caller's object phrase — the grasp TARGET (`robot.grasp(side, target)`),
threaded through by the grasp source so the call MEANS what it says:
  * auto (``Sam3Segmenter``): a non-empty ``text`` ALWAYS becomes the SAM3 text prompt
    (even over a configured box/points prompt); ``default_prompt`` is the fallback for
    an empty/None text.
  * interactive (``segment_gui.InteractiveSam3Segmenter``): ``text`` SEEDS the GUI (the
    window opens already querying it); the operator refines/overrides as usual.
  * none (``NullSegmenter``): ignored.

The interactive one raises ``SegmentationAborted`` on abort so the caller returns no
grasps rather than silently grasping the whole scene. It lives with the GUI
(``segment_gui``) so this module stays display-free.
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
    def mask(self, rgb: np.ndarray, text: Optional[str] = None) -> Optional[np.ndarray]:
        """``(H,W,3)`` RGB -> ``(H,W)`` bool object mask, all-False (found nothing), or
        None (no masking / whole frame). ``text`` = the caller's object phrase (the
        grasp target); see the module docstring for the per-mode rule."""


class NullSegmenter(Segmenter):
    """No segmentation -> the whole frame becomes the cloud (sim / mode: null)."""

    def mask(self, rgb: np.ndarray, text: Optional[str] = None) -> Optional[np.ndarray]:
        return None


class Sam3Segmenter(Segmenter):
    """One headless SAM3 call; returns the top mask (or an all-False mask if the object
    was not found). A non-empty ``text`` (the grasp target) is THE prompt; the configured
    ``default_prompt`` is only the empty-text fallback."""

    def __init__(self, client_factory, prompt_cfg: dict, top_k: int = 1):
        self.client_factory = client_factory      # () -> Sam3Client (context manager)
        self.prompt_cfg = prompt_cfg
        self.top_k = int(top_k)

    def mask(self, rgb: np.ndarray, text: Optional[str] = None) -> Optional[np.ndarray]:
        if text and text.strip():
            kw = {"text": text.strip()}            # the call's target wins
        else:
            kw = _prompt_kwargs(self.prompt_cfg)   # fallback: configured default_prompt
        with self.client_factory() as c:
            masks, _scores, _labels = c.segment(rgb, top_k=self.top_k, **kw)
        if masks.shape[0] == 0:                    # found nothing -> empty (loud), NOT whole-frame
            return np.zeros(rgb.shape[:2], dtype=bool)
        return masks[0] > 0
