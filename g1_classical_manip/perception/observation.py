"""LeRobot-style observation assembly (mirrors unitree_lerobot's
``observation.images.<cam>`` + ``observation.state`` convention, make_robot.py).

This is stage-1 preprocessing only: raw camera frames -> RGB HWC uint8, keyed by
camera name, plus the proprioceptive state vector. Stage-2 (uint8->float/255,
HWC->CHW, normalization) belongs at policy time and is intentionally NOT done here
-- nothing consumes this dict yet (no policy/eval loop in scope). Kept numpy-only
(no torch) so the offline path stays import-light.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

_HAND_SIDES = ("left", "right")


def build_observation(cameras, arm, hand=None,
                      include_ee_state: bool = False) -> Dict[str, np.ndarray]:
    """Assemble one observation:
        observation.images.<cam> -> (H, W, 3) uint8 RGB   (one per live camera)
        observation.state        -> (14,) float32 dual-arm q  [left7 + right7]
    ``include_ee_state`` appends per-hand q (changes the state dimension a future
    policy expects, so it is OFF by default)."""
    obs: Dict[str, np.ndarray] = {}

    frames = cameras.get_rgb_frames() if cameras is not None else {}
    for name, rgb in frames.items():
        obs[f"observation.images.{name}"] = np.asarray(rgb)

    if arm is not None:
        state = np.asarray(arm.get_current_dual_arm_q(), dtype=np.float32)
    else:
        state = np.zeros(14, dtype=np.float32)

    if include_ee_state and hand is not None:
        state = np.concatenate([state, _ee_state(hand)]).astype(np.float32)

    obs["observation.state"] = state
    return obs


def _ee_state(hand) -> np.ndarray:
    """Per-hand joint positions [left..., right...], best-effort. Returns an empty
    array if the hand controller does not expose state."""
    out = []
    for side in _HAND_SIDES:
        try:
            out.append(np.asarray(hand.ctrl.get_state(side)["q"], dtype=np.float32))
        except Exception:
            pass
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
