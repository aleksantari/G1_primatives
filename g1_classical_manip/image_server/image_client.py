"""Head-camera image client.

Three backends, all exposing the same minimal perception surface (BGR/gray head
frames; AprilTag needs no depth):
  * ``zmq``           -- subscribe directly to the teleimager ZMQ PUB stream
    (JPEG frames). This is what the Isaac sim publishes (640x480 on :55555) and
    needs only pyzmq + cv2, no image-server package. CONFLATE keeps the latest frame.
  * ``teleimager``    -- the PC2 ``teleimager.ImageClient`` (lazy import; hardware).
  * ``unitree_lerobot`` -- unitree_lerobot's ZMQ ImageClient (same contract).

All imports are lazy so the module loads on machines without the optional backends.
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
        if backend == "zmq":
            import zmq  # lazy
            self._zmq = zmq
            port = int(kwargs.get("port", 55555))
            self._sock = zmq.Context.instance().socket(zmq.SUB)
            self._sock.setsockopt(zmq.CONFLATE, 1)        # keep only the latest frame
            self._sock.setsockopt(zmq.SUBSCRIBE, b"")
            self._sock.setsockopt(zmq.RCVTIMEO, int(kwargs.get("recv_timeout_ms", 2000)))
            self._sock.connect(f"tcp://{host}:{port}")
        elif backend == "teleimager":
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
        if self.backend == "zmq":
            try:
                buf = self._sock.recv()
            except self._zmq.Again:
                return None                                # no frame within timeout
            if cv2 is None:
                return None
            return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)  # BGR
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
