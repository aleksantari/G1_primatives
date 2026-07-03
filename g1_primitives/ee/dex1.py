"""Dex1 (1-DoF gripper) hand: open/close presets + stall/tau grasp verification.
Kept to preserve the end-effector swap seam (factory: hand=dex1)."""
from __future__ import annotations

import time

import numpy as np

from g1_primitives.ee.hand_base import Hand


class Dex1Hand(Hand):
    def __init__(self, controller, cfg: dict, clock=time.monotonic, sleep=time.sleep):
        self.ctrl = controller
        p = cfg["presets"]
        self.open_v = float(p["open"])
        self.close_v = float(p["close"])
        self._last_close_target = self.close_v
        v = cfg["verify"]
        self.stall_margin = float(v.get("stall_margin", 0.5))
        self.still_dq = float(v.get("still_dq", 0.05))
        self.tau_threshold = float(v.get("tau_threshold", 0.5))
        self.settle_s = float(v.get("settle_s", 0.3))
        self.close_timeout_s = float(v.get("close_timeout_s", 1.5))
        self.open_timeout_s = float(v.get("open_timeout_s", 1.5))
        self._clock = clock
        self._sleep = sleep

    def command(self, side, q):
        self.ctrl.command(side, q)

    def get_state(self, side):
        return self.ctrl.get_state(side)

    def grasped(self, side) -> bool:
        st = self.ctrl.get_state(side)
        q, dq, tau = float(st["q"][0]), float(st["dq"][0]), float(st["tau"][0])
        settled = abs(dq) < self.still_dq
        stalled_short = abs(q - self._last_close_target) > self.stall_margin
        return bool((settled and stalled_short) or abs(tau) > self.tau_threshold)

    def _wait_settled(self, side, timeout):
        t0 = self._clock()
        while self._clock() - t0 < timeout:
            if abs(self.ctrl.get_state(side)["dq"][0]) < self.still_dq:
                self._sleep(self.settle_s)
                return True
            self._sleep(0.02)
        return False

    def close(self, side, verify=True, fraction=1.0) -> bool:
        target = self.open_v + float(fraction) * (self.close_v - self.open_v)
        self._last_close_target = target
        self.ctrl.command(side, target)
        if not verify:
            self._sleep(self.close_timeout_s)   # block until the gripper actually moves
            return True
        self._wait_settled(side, self.close_timeout_s)
        return self.grasped(side)

    def open(self, side, verify=True) -> bool:
        self.ctrl.command(side, self.open_v)
        if not verify:
            self._sleep(self.open_timeout_s)    # block until the gripper actually moves
            return True
        self._wait_settled(side, self.open_timeout_s)
        return abs(self.ctrl.get_state(side)["q"][0] - self.open_v) < self.stall_margin
