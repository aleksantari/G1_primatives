"""Head-camera client wrapper. One small surface over a few transports; frames are
RGB-first (``get_rgb_frame``) for perception, with ``get_bgr_frame`` (cv2 viewers) and
``get_gray_frame`` (AprilTag) alongside.

Backends:
  * ``unitree`` -- the vendored unitree_lerobot ImageClient (``zmq_image_client``):
    threaded SUB + REQ-config(:60000) + TeleImage. This is the REAL-robot head: the ZED
    publishes the stereo pair as ONE side-by-side frame (720x2560), which we slice at
    width//2 to a single 1280x720 eye (``stereo`` / ``stereo_side``).
  * ``zmq``     -- simple direct SUB to a teleimager ZMQ PUB (JPEG), CONFLATE-latest, no
    REQ-config. The Isaac sim publishes 640x480 mono on :55555 -- the proven sim path.
  * ``teleimager`` -- the legacy PC2 ``teleimager.ImageClient`` (lazy import).

All optional imports are lazy. Stereo slicing is shape-driven (``width//2``), so it works
regardless of the exact ZED resolution.
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
                 backend: str = "unitree", **kwargs):
        self.host = host
        self.backend = backend
        self._client = None
        # stereo: slice the side-by-side frame and keep one eye (real ZED head).
        self._stereo = bool(kwargs.get("stereo", False))
        self._stereo_side = str(kwargs.get("stereo_side", "left"))

        if backend == "unitree":
            from g1_classical_manip.image_server.zmq_image_client import ImageClient
            self._client = ImageClient(
                host=host, request_port=int(kwargs.get("request_port", 60000)),
                request_bgr=True)  # request_bgr -> the client decodes BGR in a bg thread
            # default `stereo` from the server's REQ-config binocular flag if not set
            if "stereo" not in kwargs:
                try:
                    head = (self._client.get_cam_config() or {}).get("head_camera", {})
                    self._stereo = bool(head.get("binocular", False))
                except Exception:
                    pass
        elif backend == "zmq":
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
        else:
            raise ValueError(f"unknown image backend: {backend}")

    # ------------------------------------------------------------------ helpers
    def _slice_stereo(self, bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Side-by-side stereo -> one eye. No-op for mono (stereo=False)."""
        if bgr is None or not self._stereo:
            return bgr
        mid = bgr.shape[1] // 2
        return bgr[:, :mid] if self._stereo_side == "left" else bgr[:, mid:]

    # -------------------------------------------------------------------- frames
    def get_bgr_frame(self) -> Optional[np.ndarray]:
        if self.backend == "unitree":
            ti = self._client.get_head_frame()
            return self._slice_stereo(ti.bgr if ti is not None else None)
        if self.backend == "zmq":
            try:
                buf = self._sock.recv()
            except self._zmq.Again:
                return None                                # no frame within timeout
            if cv2 is None:
                return None
            return self._slice_stereo(
                cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR))
        # teleimager
        img, _fps = self._client.get_head_frame()
        return self._slice_stereo(img)

    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """RGB head frame (the perception currency). Future RGB primitives use this."""
        bgr = self.get_bgr_frame()
        if bgr is None:
            return None
        if cv2 is None:
            return np.ascontiguousarray(np.asarray(bgr)[..., ::-1])
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_gray_frame(self) -> Optional[np.ndarray]:
        bgr = self.get_bgr_frame()
        if bgr is None:
            return None
        if cv2 is None:
            return np.asarray(bgr).mean(axis=-1).astype(np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def shape(self) -> Tuple[int, int]:
        bgr = self.get_bgr_frame()
        return (bgr.shape[0], bgr.shape[1]) if bgr is not None else (0, 0)
