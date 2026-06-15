"""CameraRig: named multi-camera RGB source (head / left_wrist / right_wrist).

Generalizes ``HeadCamera`` to the multi-camera topology of unitree_lerobot's
ImageClient (named per-camera getters). Returns **RGB** HWC uint8 frames keyed by
camera name; latest-frame semantics come from the underlying backend client.

Backends:
  * ``teleimager``      -- the deployed PC2 stack; HEAD ONLY (no wrist getters).
  * ``unitree_lerobot`` -- ZMQ ImageClient with head + left/right wrist getters.

Only constructed on the hardware/sim path (factory ``connect_camera=True``), so
the backend imports stay lazy and the offline test path never touches them.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import cv2 as _cv2
except Exception:
    _cv2 = None

_GETTERS = {
    "head": "get_head_frame",
    "left_wrist": "get_left_wrist_frame",
    "right_wrist": "get_right_wrist_frame",
}


def _extract_bgr(obj):
    """Normalize a backend frame to a BGR ndarray (or None). Handles teleimager's
    (img, fps) tuple, unitree_lerobot's TeleImage (.bgr), or a bare ndarray."""
    if obj is None:
        return None
    if hasattr(obj, "bgr"):
        return obj.bgr
    if isinstance(obj, tuple):
        return obj[0]
    return obj


def _bgr_to_rgb(bgr) -> Optional[np.ndarray]:
    if bgr is None:
        return None
    a = np.asarray(bgr)
    if a.ndim == 3 and a.shape[2] == 3:
        return _cv2.cvtColor(a, _cv2.COLOR_BGR2RGB) if _cv2 is not None \
            else a[:, :, ::-1].copy()
    return a


class CameraRig:
    DEFAULT_HOST = "192.168.123.164"

    def __init__(self, cameras: Dict[str, dict], host: str = DEFAULT_HOST,
                 request_bgr: bool = True):
        self.host = host
        self._cams: Dict[str, dict] = {}
        self._clients: Dict[str, object] = {}      # backend -> shared client
        self._getter = {}                          # cam name -> bound getter
        for name, spec in cameras.items():
            if not (spec or {}).get("enabled", False):
                continue
            if name not in _GETTERS:
                raise ValueError(
                    f"camera {name!r}: no getter mapping (known: {list(_GETTERS)})")
            backend = spec.get("backend", "teleimager")
            if backend == "teleimager" and name != "head":
                raise ValueError(
                    f"camera {name!r}: teleimager backend has no {name} getter; "
                    "use the unitree_lerobot backend for wrist cameras")
            client = self._client_for(backend, request_bgr)
            getter_name = _GETTERS[name]
            if not hasattr(client, getter_name):
                raise RuntimeError(
                    f"backend {backend!r} client has no {getter_name}()")
            self._cams[name] = spec
            self._getter[name] = getattr(client, getter_name)

    def _client_for(self, backend: str, request_bgr: bool):
        if backend in self._clients:
            return self._clients[backend]
        if backend == "teleimager":
            from teleimager import ImageClient        # lazy; hardware env only
            c = ImageClient(host=self.host)
            if hasattr(c, "has_head_cam") and not c.has_head_cam():
                raise RuntimeError("Head camera not available on image server.")
        elif backend == "unitree_lerobot":
            from unitree_lerobot.eval_robot.image_server.image_client import ImageClient
            c = ImageClient(host=self.host, request_bgr=request_bgr)
        else:
            raise ValueError(f"unknown image backend: {backend}")
        self._clients[backend] = c
        return c

    @classmethod
    def from_config(cls, cameras_cfg: dict, host: Optional[str] = None,
                    request_bgr: bool = True) -> "CameraRig":
        cams = cameras_cfg.get("cameras", cameras_cfg)
        kw = {"request_bgr": request_bgr}
        if host is not None:
            kw["host"] = host
        return cls(cams, **kw)

    def cameras(self) -> List[str]:
        return list(self._cams.keys())

    def get_rgb_frame(self, name: str) -> Optional[np.ndarray]:
        g = self._getter.get(name)
        return _bgr_to_rgb(_extract_bgr(g())) if g is not None else None

    def get_rgb_frames(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name in self._cams:
            f = self.get_rgb_frame(name)
            if f is not None:
                out[name] = f
        return out
