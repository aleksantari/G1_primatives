"""Head-camera client wrapper. One small surface over a few transports; frames are
RGB-first (``get_rgb_frame``) for perception, with ``get_bgr_frame`` (cv2 viewers) and
``get_gray_frame`` alongside.

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
        self._depth_sock = None        # zmq backend: optional 2nd SUB for the sim depth stream
        self._depth_hw = None          # (height, width) for reshaping raw float32 depth
        # stereo: slice the side-by-side frame and keep one eye (real ZED head).
        self._stereo = bool(kwargs.get("stereo", False))
        self._stereo_side = str(kwargs.get("stereo_side", "left"))
        # depth: tri-state gate. None -> follow the server cam_config; True/False -> force
        # on/off (a True gate can't conjure a stream the server doesn't advertise).
        self._depth_pref = kwargs.get("depth", None)

        if backend == "unitree":
            from g1_primitives.hardware.zmq_image_client import ImageClient
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
            # optional depth: a 2nd SUB on depth_port. The Isaac sim head camera publishes a
            # raw float32 (height,width) depth map in MILLIMETERS there (mirroring the real
            # ZED), so the same deproject -> PointCloud -> cuRobo Mapper/ESDF path runs in sim.
            # Gate: depth_port configured AND depth not forced off.
            dport = kwargs.get("depth_port")
            if dport is not None and self._depth_pref is not False:
                self._depth_sock = zmq.Context.instance().socket(zmq.SUB)
                self._depth_sock.setsockopt(zmq.CONFLATE, 1)
                self._depth_sock.setsockopt(zmq.SUBSCRIBE, b"")
                self._depth_sock.setsockopt(zmq.RCVTIMEO, int(kwargs.get("recv_timeout_ms", 2000)))
                self._depth_sock.connect(f"tcp://{host}:{int(dport)}")
                self._depth_hw = (int(kwargs.get("depth_height", 480)),
                                  int(kwargs.get("depth_width", 640)))
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

    # --------------------------------------------------------------------- depth
    @property
    def has_depth(self) -> bool:
        """Whether a head depth stream is available on this backend/target. The real ZED
        (``unitree`` backend) publishes depth; the ``zmq`` backend has it when the sim
        advertises a depth_port (Isaac front_camera distance_to_image_plane); teleimager is
        color-only. A ``depth=False`` kwarg forces it off."""
        if self._depth_pref is False:
            return False
        if self.backend == "unitree":
            return self._client is not None and bool(getattr(self._client, "has_depth", False))
        if self.backend == "zmq":
            return self._depth_sock is not None
        return False

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Latest head depth as float32 in MILLIMETERS, NaN/inf = invalid (masking is the
        consumer's job; meters = depth/1000). NOT stereo-sliced -- head depth is a single map.
        Real ZED (``unitree``): (720,1280). Isaac sim (``zmq``): a raw float32 (h,w) buffer on
        depth_port. Returns None when no depth stream exists or no frame has arrived yet."""
        if not self.has_depth:
            return None
        if self.backend == "zmq":
            try:
                buf = self._depth_sock.recv()
            except self._zmq.Again:
                return None                                # no depth frame within the timeout
            h, w = self._depth_hw
            d = np.frombuffer(buf, dtype=np.float32)
            if d.size != h * w:
                return None                                # shape mismatch (check depth_h/w cfg)
            return d.reshape(h, w).copy()                  # writable copy (frombuffer is RO)
        return self._client.get_head_depth_frame()

    def shape(self) -> Tuple[int, int]:
        bgr = self.get_bgr_frame()
        return (bgr.shape[0], bgr.shape[1]) if bgr is not None else (0, 0)
