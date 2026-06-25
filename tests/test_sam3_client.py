"""Wire-protocol test for the standalone SAM3 client against an in-process mock
msgpack/ZMQ server. No sam3 import, no model, no GPU."""
import threading

import numpy as np
import pytest
import zmq
import msgpack
import msgpack_numpy

from g1_classical_manip.perception.sam3_client import Sam3Client

msgpack_numpy.patch()

H, W = 12, 16
CANNED_MASKS = np.stack([                       # (2,H,W) uint8 0/255, best-first
    (np.zeros((H, W), np.uint8)),
    (np.zeros((H, W), np.uint8))])
CANNED_MASKS[0, 2:6, 3:7] = 255
CANNED_SCORES = np.array([0.9, 0.4], np.float32)


class MockServer:
    def __init__(self, mode="ok", empty=False):
        self.mode, self.empty = mode, empty
        self.sock = zmq.Context.instance().socket(zmq.REP)
        self.port = self.sock.bind_to_random_port("tcp://127.0.0.1")
        self.last_request = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        while not self._stop.is_set():
            if dict(poller.poll(timeout=50)).get(self.sock) == zmq.POLLIN:
                req = msgpack.unpackb(self.sock.recv(), raw=False)
                self.last_request = req
                self.sock.send(msgpack.packb(self._reply(req), use_bin_type=True))

    def _reply(self, req):
        if self.mode == "error":
            return {"error": "ValueError: boom"}
        action = req.get("action")
        if action == "health":
            return {"status": "ok"}
        if action == "segment":
            img = np.asarray(req["image"])
            assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
            if self.empty:                      # object-not-found: (0,H,W), NOT an error
                h, w = img.shape[:2]
                return {"masks": np.zeros((0, h, w), np.uint8),
                        "scores": np.zeros((0,), np.float32), "labels": [],
                        "timing": {"infer_ms": 1.0}}
            return {"masks": CANNED_MASKS, "scores": CANNED_SCORES,
                    "labels": ["", ""], "timing": {"infer_ms": 1.0}}
        return {"error": f"unknown action {action}"}

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1)
        self.sock.close(0)


def _free_port():
    s = zmq.Context.instance().socket(zmq.REP)
    p = s.bind_to_random_port("tcp://127.0.0.1")
    s.close(0)
    return p


def _rgb():
    return np.zeros((H, W, 3), np.uint8)


def test_health_and_segment_round_trip():
    srv = MockServer()
    try:
        with Sam3Client(host="127.0.0.1", port=srv.port, timeout_ms=2000) as c:
            assert c.health() == {"status": "ok"}
            masks, scores, labels = c.segment(_rgb(), box=[3, 2, 7, 6], top_k=3)
            assert masks.shape == (2, H, W) and masks.dtype == np.uint8
            assert scores.shape == (2,) and scores.dtype == np.float32
            req = srv.last_request
            assert req["action"] == "segment" and req["top_k"] == 3
            assert req["prompt"] == {"box": [3.0, 2.0, 7.0, 6.0]}
            assert np.asarray(req["image"]).shape == (H, W, 3)
    finally:
        srv.stop()


def test_points_prompt_with_labels():
    srv = MockServer()
    try:
        with Sam3Client(host="127.0.0.1", port=srv.port, timeout_ms=2000) as c:
            c.segment(_rgb(), points=[[4, 4], [8, 8]], point_labels=[1, 0])
            assert srv.last_request["prompt"] == {
                "points": [[4.0, 4.0], [8.0, 8.0]], "point_labels": [1, 0]}
    finally:
        srv.stop()


def test_object_not_found_is_not_error():
    srv = MockServer(empty=True)
    try:
        with Sam3Client(host="127.0.0.1", port=srv.port, timeout_ms=2000) as c:
            masks, scores, labels = c.segment(_rgb(), text="ghost")
            assert masks.shape == (0, H, W) and masks.shape[0] == 0
            assert scores.shape == (0,) and labels == []
    finally:
        srv.stop()


def test_error_reply_raises():
    srv = MockServer(mode="error")
    try:
        with Sam3Client(host="127.0.0.1", port=srv.port, timeout_ms=2000) as c:
            with pytest.raises(RuntimeError):
                c.segment(_rgb(), text="x")
    finally:
        srv.stop()


def test_timeout_raises():
    with Sam3Client(host="127.0.0.1", port=_free_port(), timeout_ms=300) as c:
        with pytest.raises(TimeoutError):
            c.segment(_rgb(), text="x")


def test_bad_prompt_and_shape_rejected_client_side():
    with Sam3Client(host="127.0.0.1", port=_free_port(), timeout_ms=300) as c:
        with pytest.raises(ValueError):
            c.segment(_rgb(), box=[0, 0, 1, 1], text="x")     # two prompts
        with pytest.raises(ValueError):
            c.segment(_rgb())                                  # no prompt
        with pytest.raises(ValueError):
            c.segment(np.zeros((H, W), np.uint8), text="x")    # wrong image shape
