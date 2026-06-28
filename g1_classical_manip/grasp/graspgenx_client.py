"""Thin standalone ZMQ REQ client for the GraspGenX inference server.

Mirrors ``graspgenx.serving.zmq_client`` but imports NOTHING from the graspgenx package:
importing ``graspgenx`` triggers a multi-GB checkpoint/asset auto-download we don't want
on the robot box. This is a pure msgpack/ZMQ/numpy wire shim — deps: pyzmq, msgpack,
msgpack-numpy, numpy. The server centers the cloud internally and returns grasps in the
SAME frame the cloud was sent in, so send a pelvis-frame cloud (meters) and grasps come
back in the pelvis frame.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import msgpack
import msgpack_numpy
import numpy as np
import zmq

msgpack_numpy.patch()                      # numpy arrays serialize natively

logger = logging.getLogger(__name__)

EXPECTED_PROTOCOL_VERSION = 2              # bump in lockstep with the server's wire schema


class GraspGenXClient:
    """ZMQ REQ client round-tripping msgpack payloads to a GraspGenX server.

        with GraspGenXClient(host="127.0.0.1", port=5556) as client:
            grasps, conf = client.infer(points_xyz, gripper_name="unitree_g1")
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5556,
                 timeout_ms: Optional[int] = 60_000) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._ctx: Optional[zmq.Context] = None
        self._sock: Optional[zmq.Socket] = None

    @property
    def address(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def __enter__(self) -> "GraspGenXClient":
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
        logger.info("Connected to GraspGenX server at %s", self.address)

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
                f"GraspGenX server at {self.address} did not respond within "
                f"{self.timeout_ms} ms") from exc
        response = msgpack.unpackb(raw, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GraspGenX server error: {response['error']}")
        return response

    def health(self) -> dict:
        return self._request({"action": "health"})

    def metadata(self) -> dict:
        meta = self._request({"action": "metadata"})
        pv = meta.get("protocol_version")
        if pv is not None and pv != EXPECTED_PROTOCOL_VERSION:
            logger.warning("GraspGenX server protocol_version=%s; client expects %s "
                           "(upgrade one side)", pv, EXPECTED_PROTOCOL_VERSION)
        return meta

    def infer(self, point_cloud: np.ndarray, gripper_name: Optional[str] = None,
              num_grasps: int = 200, grasp_threshold: float = -1.0,
              topk_num_grasps: int = 100, planner: Optional[str] = None,
              obb_density: Optional[str] = None, skip_obb_rule: Optional[str] = None
              ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Send an ``(N,3)`` float32 cloud, return
        ``(grasps (K,4,4) f32, conf (K,) f32, branch_tags ["diff"|"obb", ...])``
        ranked best-first (may be empty if the model produced nothing).

        ``planner`` selects ``diffusion`` | ``graspmoe`` | ``topdown`` (OBB-only,
        top-down/side grasps); ``obb_density`` (sparse|dense|dense-topandside) and
        ``skip_obb_rule`` (auto|never) tune the GraspMoE OBB branch. Leave any as
        None to use the server default. ``branch_tags[i]`` is "obb" for an OBB
        (top-down/side) grasp, "diff" for a diffusion grasp."""
        pc = np.asarray(point_cloud, dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"point_cloud must be (N, 3); got {pc.shape}")
        payload = {
            "action": "infer",
            "point_cloud": pc,
            "num_grasps": int(num_grasps),
            "grasp_threshold": float(grasp_threshold),
            "topk_num_grasps": int(topk_num_grasps),
        }
        if gripper_name is not None:
            payload["gripper_name"] = gripper_name
        if planner is not None:
            payload["planner"] = str(planner)
        if obb_density is not None:
            payload["obb_density"] = str(obb_density)
        if skip_obb_rule is not None:
            payload["skip_obb_rule"] = str(skip_obb_rule)
        response = self._request(payload)
        grasps = np.asarray(response["grasps"], dtype=np.float32)
        confidences = np.asarray(response["confidences"], dtype=np.float32)
        tags = list(response.get("branch_tags", []))
        return grasps, confidences, tags
