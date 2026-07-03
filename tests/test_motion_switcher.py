"""ensure_debug_mode contract (DDS-free via the injected client factory).

The load-bearing case is the FIRST one: a robot the operator already put in debug
mode (CheckMode -> empty name) must be left UNTOUCHED -- zero ReleaseMode calls.
Releasing there drops the robot OUT of low-level control (hardware-observed:
rt/lowcmd publishes into the void, arms never move)."""
from g1_primitives.hardware.motion_switcher import ensure_debug_mode


class FakeMSC:
    """CheckMode() returns the queued names in order (the last repeats forever)."""

    def __init__(self, names):
        self._names = list(names)
        self.release_calls = 0

    def CheckMode(self):
        name = self._names.pop(0) if len(self._names) > 1 else self._names[0]
        return 0, {"name": name}

    def ReleaseMode(self):
        self.release_calls += 1


def test_already_debug_is_left_untouched():
    msc = FakeMSC([""])
    ok, msg = ensure_debug_mode(verbose=False, client_factory=lambda: msc)
    assert ok
    assert msc.release_calls == 0          # the whole point: NEVER blind-release
    assert "untouched" in msg


def test_named_mode_released_then_ok():
    msc = FakeMSC(["ai", ""])
    ok, msg = ensure_debug_mode(verbose=False, client_factory=lambda: msc, poll_s=0.0)
    assert ok
    assert msc.release_calls == 1
    assert "ai" in msg


def test_stuck_mode_times_out_false():
    msc = FakeMSC(["ai"])                  # never clears
    ok, msg = ensure_debug_mode(verbose=False, client_factory=lambda: msc,
                                timeout_s=0.05, poll_s=0.01)
    assert not ok
    assert msc.release_calls >= 1
    assert "ai" in msg


def test_none_result_treated_as_debug():
    # a client whose CheckMode returns (status, None) (the vendored error shape)
    class NoneMSC(FakeMSC):
        def CheckMode(self):
            return 0, None

    msc = NoneMSC([""])
    ok, _ = ensure_debug_mode(verbose=False, client_factory=lambda: msc)
    assert ok and msc.release_calls == 0


def test_client_failure_returns_false_never_raises():
    def boom():
        raise RuntimeError("no dds")

    ok, msg = ensure_debug_mode(verbose=False, client_factory=boom)
    assert not ok
    assert "no dds" in msg
