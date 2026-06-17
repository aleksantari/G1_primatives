import numpy as np
import yaml
import os

from g1_classical_manip.ee.dex3 import Dex3Hand


class FakeCtrl:
    def __init__(self):
        self.state = {}

    def command(self, side, q):
        pass

    def set(self, side, q, dq, tau, press):
        self.state[side] = dict(q=np.array(q, float), dq=np.array(dq, float),
                                tau=np.array(tau, float), press=np.array(press, float))

    def get_state(self, side):
        return {k: v.copy() for k, v in self.state[side].items()}


def _hand(side="left"):
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__),
                                           "..", "configs", "hands.yaml")))["dex3"]
    fc = FakeCtrl()
    hand = Dex3Hand(fc, cfg, clock=lambda: 0.0, sleep=lambda s: None)
    # power_close is per-hand ({left,right}); return the array for `side`.
    return hand, fc, np.asarray(cfg["presets"]["power_close"][side], float)


def test_object_held_is_grasped():
    hand, fc, close_t = _hand("left")
    fc.set("left", close_t - 0.4, [0] * 7, [0.1] * 7, [0] * 7)
    assert hand.close("left") is True


def test_closed_on_air_not_grasped():
    hand, fc, close_t = _hand("left")
    fc.set("left", close_t, [0] * 7, [0.05] * 7, [0] * 7)
    assert hand.close("left") is False


def test_tau_triggers_grasp():
    hand, fc, close_t = _hand("right")
    fc.set("right", close_t, [0] * 7, [0, 0, 0.5, 0, 0, 0, 0], [0] * 7)
    assert hand.close("right") is True


def test_press_triggers_grasp():
    hand, fc, close_t = _hand("right")
    fc.set("right", close_t, [0] * 7, [0.05] * 7, [0, 0, 0, 40, 0, 0, 0])
    assert hand.close("right") is True


def test_open_reached():
    hand, fc, _ = _hand("left")
    fc.set("left", np.zeros(7), [0] * 7, [0] * 7, [0] * 7)
    assert hand.open("left") is True
