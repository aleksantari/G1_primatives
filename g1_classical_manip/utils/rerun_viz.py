"""Thin Rerun logging for FSM transitions + trajectory execution (CLAUDE.md
rule 5). Degrades to a no-op if rerun is unavailable or disabled.
"""
from __future__ import annotations

import numpy as np


class RerunLogger:
    def __init__(self, app: str = "g1_classical_manip", enabled: bool = True,
                 spawn: bool = False, save_path: str = None):
        self.enabled = enabled
        self.rr = None
        if not enabled:
            return
        try:
            import rerun as rr
            self.rr = rr
            rr.init(app, spawn=spawn)
            if save_path:
                rr.save(save_path)
            self._t = 0.0
        except Exception:
            self.rr = None
            self.enabled = False

    def log_transition(self, state: str, ctx) -> None:
        if not self.rr:
            return
        self._t += 1.0
        self.rr.set_time_seconds("fsm", self._t)
        self.rr.log("fsm/state", self.rr.TextLog(state))
        bp = getattr(ctx, "block_pose", None)
        if bp is not None:
            self.rr.log("fsm/block_xyz", self.rr.Points3D([bp.translation]))

    def log_execution(self, t: float, q_des, q_meas, err: float) -> None:
        if not self.rr:
            return
        self.rr.set_time_seconds("exec", float(t))
        for j in range(len(q_des)):
            self.rr.log(f"exec/q_des/{j:02d}", self.rr.Scalar(float(q_des[j])))
            self.rr.log(f"exec/q_meas/{j:02d}", self.rr.Scalar(float(q_meas[j])))
        self.rr.log("exec/tracking_error", self.rr.Scalar(float(err)))
