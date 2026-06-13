"""Trajectory executor: the ONLY holder of the G1_29_ArmController handle.

Streams a JointTrajectory at the control rate, applies optional gravity-comp
feed-forward torque, monitors joint tracking error, and aborts-to-hold if the
error exceeds a configured threshold. If the controller's internal velocity
clip ever activates during nominal execution, the retimer limits are wrong
(treat as a bug) -- the executor logs the symptom via max tracking error.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pinocchio as pin

from g1_classical_manip.motion.planner_base import JointTrajectory, DOF


@dataclass
class ExecutionResult:
    success: bool
    aborted: bool
    max_tracking_error: float
    reason: str = ""


class Executor:
    def __init__(self, arm_controller, ik=None, control_hz: float = 250.0,
                 tracking_error_abort_rad: float = 0.20, gravity_comp: bool = True,
                 rerun_logger=None):
        self.arm = arm_controller
        self.ik = ik
        self.control_hz = float(control_hz)
        self.abort_thresh = float(tracking_error_abort_rad)
        self.gravity_comp = gravity_comp and ik is not None
        self.rr = rerun_logger

    # -------------------------------------------------------------- helpers
    def _tauff(self, q14: np.ndarray) -> np.ndarray:
        if not self.gravity_comp:
            return np.zeros(DOF)
        m = self.ik.reduced_robot.model
        d = self.ik.reduced_robot.data
        return pin.rnea(m, d, q14, np.zeros(m.nv), np.zeros(m.nv))

    def hold(self, q14: np.ndarray):
        self.arm.ctrl_dual_arm(np.asarray(q14, float).reshape(DOF), self._tauff(q14))

    def go_home(self):
        self.arm.ctrl_dual_arm_go_home()

    # ----------------------------------------------------------------- run
    def run(self, traj: JointTrajectory, ramp: bool = True) -> ExecutionResult:
        if hasattr(self.arm, "speed_gradual_max") and ramp:
            self.arm.speed_gradual_max()

        t0 = time.time()
        dt = 1.0 / self.control_hz
        duration = traj.duration
        max_err = 0.0
        next_t = t0
        while True:
            now = time.time()
            elapsed = now - t0
            q_des = traj.sample(min(elapsed, duration))
            self.arm.ctrl_dual_arm(q_des, self._tauff(q_des))

            q_meas = self.arm.get_current_dual_arm_q()
            err = float(np.max(np.abs(q_des - q_meas)))
            max_err = max(max_err, err)
            if self.rr is not None:
                self.rr.log_execution(elapsed, q_des, q_meas, err)
            if err > self.abort_thresh:
                self.hold(q_meas)
                return ExecutionResult(False, True, max_err,
                                       f"tracking error {err:.3f} > "
                                       f"{self.abort_thresh:.3f} rad at t={elapsed:.2f}s")
            if elapsed >= duration:
                break
            next_t += dt
            time.sleep(max(0.0, next_t - time.time()))

        return ExecutionResult(True, False, max_err, "ok")
