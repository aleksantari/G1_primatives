"""Trajectory executor: the ONLY holder of the G1_29_ArmController handle.

Streams a JointTrajectory at the control rate, applies optional gravity-comp
feed-forward torque (cuRobo RNEA via the planner; OFF by default -- see
docs/gravity_comp.md), monitors joint tracking error, and aborts-to-hold if the
error exceeds a configured threshold. If the controller's internal velocity clip
ever activates during nominal execution, the cuRobo joint limits are too aggressive
(treat as a bug) -- the executor logs the symptom via max tracking error.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from g1_primitives.motion.trajectory import JointTrajectory, DOF


@dataclass
class ExecutionResult:
    success: bool
    aborted: bool
    max_tracking_error: float
    reason: str = ""


class Executor:
    def __init__(self, arm_controller, planner=None, control_hz: float = 250.0,
                 tracking_error_abort_rad: float = 0.20, gravity_comp: bool = False,
                 gravity_scale: float = 1.0, prime_timeout_s: float = 3.0,
                 time_dilation: float = 1.0, rerun_logger=None):
        self.arm = arm_controller
        self.planner = planner
        self.control_hz = float(control_hz)
        self.abort_thresh = float(tracking_error_abort_rad)
        # Trajectory playback speed: the cuRobo plan is sampled at (wall-clock * this).
        # < 1.0 slows playback (same path/goal, lower joint velocity & quadratically
        # lower acceleration), so a torque-limited arm -- gravity-only feed-forward, PD
        # torque capped by the velocity clip -- can track the trajectory without the
        # tracking error diverging. 1.0 = cuRobo's native timing.
        self.time_dilation = float(time_dilation)
        # gravity-comp needs the planner's cuRobo dynamics; off by default (sim runs
        # fine on zero feed-forward, and the sign must be confirmed once on hardware).
        self.gravity_comp = bool(gravity_comp) and planner is not None
        self.gravity_scale = float(gravity_scale)
        self.prime_timeout_s = float(prime_timeout_s)
        self.rr = rerun_logger

    # -------------------------------------------------------------- helpers
    def _tauff(self, q14: np.ndarray) -> np.ndarray:
        """Feed-forward torque: zero unless gravity_comp is enabled, then
        gravity_scale * G(q) from the planner's cuRobo RNEA (gravity-only). Sign is
        hardware-validated; see docs/gravity_comp.md."""
        if not self.gravity_comp:
            return np.zeros(DOF)
        return self.gravity_scale * self.planner.gravity_torque(q14)

    def hold(self, q14: np.ndarray):
        self.arm.ctrl_dual_arm(np.asarray(q14, float).reshape(DOF), self._tauff(q14))

    def go_home(self):
        self.arm.ctrl_dual_arm_go_home()

    def go_home_direct(self, q_home: np.ndarray, ramp: bool = True,
                       tol: float = 0.05, timeout: float = 10.0) -> float:
        """Drive both arms to q_home with DIRECT position control (PD), NOT a cuRobo
        plan. This bypasses the collision-aware planner, so it can move *out of* a
        pose cuRobo flags as a self-collision START (e.g. the arms folded at launch,
        where plan_joint/plan_to_pose would refuse to plan). Velocity is bounded by
        the controller's arm_velocity_limit cap, eased up from half-cap via
        speed_gradual_max when ramp=True. It is NOT collision-avoided en route -- the
        path to q_home must be clear (watch the e-stop on first hardware contact).
        Returns the residual error (rad). Use this for the launch home, then the
        collision-aware home()/move() primitives plan from the known-good home."""
        if ramp and hasattr(self.arm, "speed_gradual_max"):
            self.arm.speed_gradual_max()
        return self.settle(q_home, tol=tol, timeout=timeout)

    def settle(self, q14: np.ndarray, tol: float = 0.05, timeout: float = 5.0) -> float:
        """Hold q14 until the measured pose converges within `tol` rad (or timeout).
        Returns the residual error. Useful after a move when a loose (sim) PD needs
        time to catch up before the next plan is started."""
        q14 = np.asarray(q14, float).reshape(DOF)
        t0 = time.time()
        err = float("inf")
        while time.time() - t0 < timeout:
            self.hold(q14)
            err = float(np.max(np.abs(q14 - self.arm.get_current_dual_arm_q())))
            if err < tol:
                return err
            time.sleep(0.02)
        return err

    # ----------------------------------------------------------------- run
    def prime(self, q_start: np.ndarray) -> float:
        """Command the trajectory's first point and wait for the measured pose to
        converge before the clock starts -- planning takes seconds, during which
        the measured pose drifts from the planned start. Returns the residual
        error reached (caller decides whether to proceed)."""
        q_start = np.asarray(q_start, float).reshape(DOF)
        t0 = time.time()
        err = np.inf
        while time.time() - t0 < self.prime_timeout_s:
            self.arm.ctrl_dual_arm(q_start, self._tauff(q_start))
            err = float(np.max(np.abs(q_start - self.arm.get_current_dual_arm_q())))
            if err < self.abort_thresh:
                return err
            time.sleep(0.02)
        return err

    def run(self, traj: JointTrajectory, ramp: bool = True) -> ExecutionResult:
        if hasattr(self.arm, "speed_gradual_max") and ramp:
            self.arm.speed_gradual_max()

        prime_err = self.prime(traj.q[0])
        if prime_err >= self.abort_thresh:
            return ExecutionResult(False, True, prime_err,
                                   f"failed to reach trajectory start: residual "
                                   f"{prime_err:.3f} >= {self.abort_thresh:.3f} rad")

        td = max(1e-3, self.time_dilation)   # playback speed (<1 = slower); see __init__
        t0 = time.time()
        dt = 1.0 / self.control_hz
        duration = traj.duration
        max_err = 0.0
        next_t = t0
        while True:
            now = time.time()
            traj_t = (now - t0) * td          # dilated trajectory clock
            q_des = traj.sample(min(traj_t, duration))
            self.arm.ctrl_dual_arm(q_des, self._tauff(q_des))

            q_meas = self.arm.get_current_dual_arm_q()
            err = float(np.max(np.abs(q_des - q_meas)))
            max_err = max(max_err, err)
            if self.rr is not None:
                self.rr.log_execution(traj_t, q_des, q_meas, err)
            if err > self.abort_thresh:
                self.hold(q_meas)
                return ExecutionResult(False, True, max_err,
                                       f"tracking error {err:.3f} > "
                                       f"{self.abort_thresh:.3f} rad at traj_t={traj_t:.2f}s")
            if traj_t >= duration:
                break
            next_t += dt
            time.sleep(max(0.0, next_t - time.time()))

        return ExecutionResult(True, False, max_err, "ok")
