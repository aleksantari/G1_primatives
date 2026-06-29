"""Dex3 hand: named presets + grasp verification.

grasped == any of:
  * motor STALL: fingers settled (|dq| small) but stopped short of the close
    target by more than stall_margin  -> something is blocking them (an object);
  * |tau_est| over threshold on any joint -> contact force;
  * press_sensor pressure over threshold -> tactile contact.

Closing on air drives the fingers all the way to the close preset (no stall, low
tau, no pressure) -> grasped == False.
"""
from __future__ import annotations

import time

import numpy as np

from g1_classical_manip.ee.hand_base import Hand, SIDES


class Dex3Hand(Hand):
    def __init__(self, controller, cfg: dict, clock=time.monotonic, sleep=time.sleep):
        self.ctrl = controller
        # a preset is either a flat 7-vector (both hands) or per-hand
        # {left:[...], right:[...]} -- the dex3 thumb flexion + finger curl are
        # sign-mirrored L vs R (see hands.yaml), so close targets are per-hand.
        self.presets = {
            k: ({s: np.asarray(p[s], float) for s in p}
                if isinstance(p, dict) else np.asarray(p, float))
            for k, p in cfg["presets"].items()}
        v = cfg["verify"]
        self.stall_margin = float(v.get("stall_margin_rad", 0.15))
        self.still_dq = float(v.get("still_dq", 0.05))
        self.tau_threshold = float(v.get("tau_threshold", 0.3))
        self.press_threshold = float(v.get("press_threshold", 20.0))
        self.settle_s = float(v.get("settle_s", 0.3))
        self.close_timeout_s = float(v.get("close_timeout_s", 2.0))
        self.open_timeout_s = float(v.get("open_timeout_s", 1.5))
        self.close_preset = v.get("close_preset", "power_close")
        self._clock = clock
        self._sleep = sleep
        self._last_close_target = {s: self._preset(self.close_preset, s) for s in SIDES}

    # ------------------------------------------------------------------- raw
    def command(self, side, q):
        self.ctrl.command(side, q)

    def get_state(self, side):
        return self.ctrl.get_state(side)

    def get_q(self, side):
        """Latest measured (7,) finger joint positions for `side` (Dex3 get_q order). Used to
        self-filter the hand at its LIVE pose in the depth collision world."""
        return self.ctrl.get_q(side)

    def _preset(self, name, side) -> np.ndarray:
        """Resolve a preset to a 7-vector for `side`: a flat array (same for both
        hands) or per-hand {left,right}. dex3 thumb flexion + finger curl are
        sign-mirrored L vs R, so close/pinch targets differ per hand."""
        p = self.presets[name]
        return p[side] if isinstance(p, dict) else p

    # ----------------------------------------------------------- verification
    def grasped(self, side) -> bool:
        st = self.ctrl.get_state(side)
        q, dq, tau, press = st["q"], st["dq"], st["tau"], st["press"]
        target = self._last_close_target[side]
        settled = np.max(np.abs(dq)) < self.still_dq
        stalled_short = np.max(np.abs(q - target)) > self.stall_margin
        stall_hit = settled and stalled_short
        tau_hit = np.max(np.abs(tau)) > self.tau_threshold
        press_hit = np.max(press) > self.press_threshold
        return bool(stall_hit or tau_hit or press_hit)

    def _wait_settled(self, side, timeout):
        t0 = self._clock()
        while self._clock() - t0 < timeout:
            if np.max(np.abs(self.ctrl.get_state(side)["dq"])) < self.still_dq:
                # allow a short settle window then accept
                self._sleep(self.settle_s)
                return True
            self._sleep(0.02)
        return False

    # ------------------------------------------------------------------- ops
    def close(self, side, verify=True, preset=None, fraction=1.0) -> bool:
        full = self._preset(preset or self.close_preset, side)
        if fraction >= 1.0:
            target = full
        else:   # interpolate open -> close (e.g. fraction=0.5 -> half-closed)
            q_open = self._preset("open", side)
            target = q_open + float(fraction) * (full - q_open)
        self._last_close_target[side] = target
        self.ctrl.command(side, target)
        if not verify:
            self._sleep(self.close_timeout_s)   # block until the fingers actually move
            return True
        self._wait_settled(side, self.close_timeout_s)
        return self.grasped(side)

    def open(self, side, verify=True) -> bool:
        target = self._preset("open", side)
        self.ctrl.command(side, target)
        if not verify:
            self._sleep(self.open_timeout_s)    # block until the fingers actually move
            return True
        self._wait_settled(side, self.open_timeout_s)
        reached = np.max(np.abs(self.ctrl.get_state(side)["q"] - target)) < self.stall_margin
        return bool(reached)

    def preset(self, side, name):
        self.ctrl.command(side, self._preset(name, side))
