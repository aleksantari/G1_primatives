"""Thin standalone ZMQ REQ client for the SAM3 segmentation server.

Protocol-identical to the GraspGenX client (g1_primitives/grasp/graspgenx_client.py);
imports NOTHING from the sam3 package (importing sam3 triggers torch + a checkpoint
download). Pure msgpack/ZMQ/numpy shim. The server returns masks in the SAME pixel grid as
the image sent, so a mask over the head-cam RGB applies pixel-for-pixel to the aligned
depth map. See /home/santari/repos/sam3/sam3/serving/README.md.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from g1_primitives.latency import LOG   # latency instrumentation (no-op unless enabled)

msgpack_numpy.patch()                      # numpy arrays serialize natively

logger = logging.getLogger(__name__)


class Sam3Client:
    """ZMQ REQ client round-tripping msgpack payloads to a SAM3 server.

        with Sam3Client(host="127.0.0.1", port=5557) as client:
            masks, scores, labels = client.segment(rgb, text="red block")
            best = masks[0] > 0           # (H,W) bool, pixel-aligned with the depth map
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5557,
                 timeout_ms: Optional[int] = 60_000) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._ctx: Optional[zmq.Context] = None
        self._sock: Optional[zmq.Socket] = None

    @property
    def address(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def __enter__(self) -> "Sam3Client":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        if self._sock is not None:
            return
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        if self.timeout_ms is not None:
            self._sock.setsockopt(zmq.RCVTIMEO, int(self.timeout_ms))
            self._sock.setsockopt(zmq.SNDTIMEO, int(self.timeout_ms))
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.address)
        logger.info("Connected to SAM3 server at %s", self.address)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None

    def _request(self, payload: dict) -> dict:
        if self._sock is None:
            self.connect()
        try:
            self._sock.send(msgpack.packb(payload, use_bin_type=True))
            raw = self._sock.recv()
        except zmq.error.Again as exc:
            self.close()                  # socket is unusable after a timeout; reset
            raise TimeoutError(
                f"SAM3 server at {self.address} did not respond within "
                f"{self.timeout_ms} ms") from exc
        response = msgpack.unpackb(raw, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"SAM3 server error: {response['error']}")
        return response

    def health(self) -> dict:
        return self._request({"action": "health"})

    def metadata(self) -> dict:
        return self._request({"action": "metadata"})

    def segment(self, rgb: np.ndarray, *, box=None, points=None, point_labels=None,
                text: Optional[str] = None, top_k: int = 1,
                return_scores: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Segment ONE object. Provide EXACTLY ONE prompt:
          box=[x0,y0,x1,y1] (pixel XYXY), points=[[x,y],...] (+ optional point_labels
          1=fg/0=bg), or text="red block".
        Returns (masks, scores, labels): masks (M,H,W) uint8 0/255 ranked best-first
        (M may be 0 = object not found), scores (M,) f32 desc, labels list[str]."""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3) uint8; got {rgb.shape}")

        prompt: dict = {}
        if box is not None:
            prompt["box"] = [float(v) for v in np.asarray(box, float).reshape(4)]
        if points is not None:
            prompt["points"] = [[float(x), float(y)] for x, y in points]
            if point_labels is not None:
                prompt["point_labels"] = [int(v) for v in point_labels]
        if text is not None:
            prompt["text"] = str(text)
        if set(prompt) not in ({"box"}, {"text"}, {"points"}, {"points", "point_labels"}):
            raise ValueError("provide exactly one prompt: box, points (+optional "
                             f"point_labels), or text (got keys {sorted(prompt)})")

        with LOG.span("sam3"):                 # pure SAM3 server round-trip (excludes the cv2 GUI)
            response = self._request({
                "action": "segment", "image": rgb, "prompt": prompt,
                "top_k": int(top_k), "return_scores": bool(return_scores)})
        masks = np.asarray(response["masks"], dtype=np.uint8)        # (M,H,W); M may be 0
        scores = np.asarray(response.get("scores", []), dtype=np.float32).reshape(-1)
        labels = list(response.get("labels", []))
        return masks, scores, labels
