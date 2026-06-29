"""HeadCamera zmq-backend depth path (Isaac sim head depth). GPU/socket-free: the planner
is built via __new__ and a fake SUB socket is injected, so we exercise the reshape/gate logic
without a real ZMQ server."""
import numpy as np

from g1_classical_manip.image_server.image_client import HeadCamera


class _Again(Exception):
    pass


class _FakeZmq:
    Again = _Again


class _FakeSock:
    """recv() yields queued payloads; None (or empty) -> raise Again (timeout/no frame)."""
    def __init__(self, frames):
        self._frames = list(frames)

    def recv(self):
        if not self._frames:
            raise _Again()
        f = self._frames.pop(0)
        if f is None:
            raise _Again()
        return f


def _cam(frames, hw=(480, 640), pref=None, sock=True):
    c = HeadCamera.__new__(HeadCamera)
    c.backend = "zmq"
    c._depth_pref = pref
    c._zmq = _FakeZmq()
    c._depth_sock = _FakeSock(frames) if sock else None
    c._depth_hw = hw
    return c


def test_zmq_depth_reshapes_to_hw_mm():
    d = np.arange(480 * 640, dtype=np.float32).reshape(480, 640)
    out = _cam([d.tobytes()]).get_depth_frame()
    assert out.shape == (480, 640) and out.dtype == np.float32
    assert np.array_equal(out, d)
    assert out.flags.writeable                       # frombuffer is read-only -> we .copy()


def test_zmq_depth_timeout_returns_none():
    assert _cam([None]).get_depth_frame() is None


def test_zmq_depth_shape_mismatch_returns_none():
    # wrong-sized buffer (e.g. depth_height/width misconfigured) -> drop, don't crash
    assert _cam([np.zeros(123, np.float32).tobytes()]).get_depth_frame() is None


def test_zmq_has_depth_gates():
    assert _cam([]).has_depth is True                # depth_sock present, pref unset
    assert _cam([], pref=False).has_depth is False   # forced off
    assert _cam([], sock=False).has_depth is False   # no depth_port -> no socket
