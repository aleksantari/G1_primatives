"""Wire-protocol test for the standalone GraspGenX client against an in-process mock
msgpack/ZMQ server. No graspgenx import, no model, no GPU."""
import threading

import numpy as np
import pytest
import zmq
import msgpack
import msgpack_numpy

from g1_classical_manip.grasp.graspgenx_client import GraspGenXClient

msgpack_numpy.patch()

CANNED_GRASPS = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))   # (2,4,4)
CANNED_CONF = np.array([0.9, 0.5], np.float32)
CANNED_TAGS = ["obb", "diff"]                                     # protocol v2 branch_tags


class MockServer:
    """A tiny REP server echoing the GraspGenX protocol on a random loopback port."""

    def __init__(self, mode="ok"):
        self.mode = mode
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
        if action == "infer":
            pc = np.asarray(req["point_cloud"])
            assert pc.ndim == 2 and pc.shape[1] == 3 and pc.dtype == np.float32
            return {"grasps": CANNED_GRASPS, "confidences": CANNED_CONF,
                    "branch_tags": CANNED_TAGS,
                    "gripper_name": req.get("gripper_name", "unitree_g1"),
                    "planner": req.get("planner", "graspmoe"),
                    "timing": {"infer_ms": 1.0}}
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


def test_health_and_infer_round_trip():
    srv = MockServer()
    try:
        with GraspGenXClient(host="127.0.0.1", port=srv.port, timeout_ms=2000) as c:
            assert c.health() == {"status": "ok"}
            pts = np.random.rand(50, 3).astype(np.float32)
            grasps, conf, tags = c.infer(pts, gripper_name="unitree_g1",
                                         num_grasps=10, topk_num_grasps=5,
                                         planner="topdown", obb_density="dense")
            assert grasps.shape == (2, 4, 4) and grasps.dtype == np.float32
            assert conf.shape == (2,) and conf.dtype == np.float32
            assert tags == ["obb", "diff"]                # branch_tags round-trip
            req = srv.last_request
            assert req["action"] == "infer" and req["gripper_name"] == "unitree_g1"
            assert req["num_grasps"] == 10 and req["topk_num_grasps"] == 5
            assert req["planner"] == "topdown" and req["obb_density"] == "dense"
            assert np.asarray(req["point_cloud"]).shape == (50, 3)
    finally:
        srv.stop()


def test_error_reply_raises():
    srv = MockServer(mode="error")
    try:
        with GraspGenXClient(host="127.0.0.1", port=srv.port, timeout_ms=2000) as c:
            with pytest.raises(RuntimeError):
                c.infer(np.zeros((10, 3), np.float32), gripper_name="unitree_g1")
    finally:
        srv.stop()


def test_timeout_raises():
    with GraspGenXClient(host="127.0.0.1", port=_free_port(), timeout_ms=300) as c:
        with pytest.raises(TimeoutError):
            c.infer(np.zeros((10, 3), np.float32), gripper_name="unitree_g1")


def test_bad_shape_rejected_client_side():
    with GraspGenXClient(host="127.0.0.1", port=_free_port(), timeout_ms=300) as c:
        with pytest.raises(ValueError):
            c.infer(np.zeros((10, 2), np.float32))
