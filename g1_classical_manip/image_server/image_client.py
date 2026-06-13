"""Head-camera image client.

The user's deployed image server on PC2 is the ``teleimager`` stack (from
G1_teleop_dex), so this wraps ``teleimager.ImageClient`` when available. The
import is lazy so the module loads on machines without teleimager; the wrapper
exposes a minimal, perception-focused surface (grayscale head frames for
AprilTag, which needs no depth).

NOTE (HARDWARE_TODO): confirm which image server is actually running on PC2 and
that ``request_bgr``/host/port match. unitree_lerobot's ZMQ ImageClient is an
alternative backend with the same `get_gray_frame()` contract.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


class HeadCamera:
    def __init__(self, host: str = "192.168.123.164", request_bgr: bool = True,
                 backend: str = "teleimager", **kwargs):
        self.host = host
        self.backend = backend
        self._client = None
        if backend == "teleimager":
            from teleimager import ImageClient  # lazy; hardware env only
            self._client = ImageClient(host=host, **kwargs)
            if not self._client.has_head_cam():
                raise RuntimeError("Head camera not available on image server.")
        elif backend == "unitree_lerobot":
            from unitree_lerobot.eval_robot.image_server.image_client import ImageClient
            self._client = ImageClient(host=host, request_bgr=request_bgr, **kwargs)
        else:
            raise ValueError(f"unknown image backend: {backend}")

    def get_bgr_frame(self) -> Optional[np.ndarray]:
        if self.backend == "teleimager":
            img, _fps = self._client.get_head_frame()
            return img
        return self._client.get_head_frame()

    def get_gray_frame(self) -> Optional[np.ndarray]:
        bgr = self.get_bgr_frame()
        if bgr is None:
            return None
        if cv2 is None:
            return np.asarray(bgr).mean(axis=-1).astype(np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def shape(self) -> Tuple[int, int]:
        if self.backend == "teleimager":
            return tuple(self._client.get_head_shape())
        bgr = self.get_bgr_frame()
        return (bgr.shape[0], bgr.shape[1]) if bgr is not None else (0, 0)
