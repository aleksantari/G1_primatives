"""Shared helpers for the scripts/ ladder: sim/real target selection + a cv2 live
viewer with a headless fallback. (Run scripts as `python scripts/NN_*.py`; the
script's dir is on sys.path so `import _rig` resolves.)"""
from __future__ import annotations

import argparse
import os

import numpy as np

from g1_primitives.factory import make_robot, load_configs
from g1_primitives import primitives as P
from g1_primitives.latency import LOG   # latency instrumentation (no-op unless enabled)

SIM_DOMAIN, SIM_IFACE = 1, "lo"


def add_target_arg(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--target", choices=["sim", "real"], default="sim",
                    help="sim = unitree_sim_isaaclab loopback (domain 1/lo, mode=sim); "
                         "real = robot.yaml DDS + mode=debug (operator sets debug via the remote).")
    return ap


def dds_for(target: str):
    """(domain, interface) for a raw DDS connection (read-only scripts)."""
    if target == "sim":
        return SIM_DOMAIN, SIM_IFACE
    dds = load_configs()["robot"]["dds"]
    return dds["domain_id"], (dds.get("interface") or None)


def camera_config_for(target: str) -> str:
    """Camera yaml for the target: real -> ZED stereo (camera_real.yaml), else sim mono."""
    return "camera_real.yaml" if target == "real" else "camera_sim.yaml"


def _warn_sim_uri(target: str):
    if target == "sim" and not os.environ.get("CYCLONEDDS_URI"):
        print("WARNING: CYCLONEDDS_URI is unset -- sim loopback DDS discovery will likely "
              "fail. Prefix the command with\n  CYCLONEDDS_URI=file://$PWD/configs/"
              "cyclonedds_loopback.xml")


def connect(target: str, **kwargs):
    """make_robot for the target: sim -> loopback DDS + mode=sim; real -> robot.yaml
    DDS + mode=debug (debug mode is set by the OPERATOR via the physical remote; we do
    NOT call MotionSwitcher by default). NOTE: any
    connect_dds=True build homes the arms with DIRECT (un-planned, velocity-capped)
    position control and waits for convergence BEFORE returning -- so the robot starts
    at a known collision-free home, and the planned home()/move() primitives plan from
    there. This physically moves the arms on connect and is NOT collision-avoided en
    route, so ensure the path to home is clear (pass home_on_connect=False to skip)."""
    _warn_sim_uri(target)
    if target == "sim":
        return make_robot(connect_dds=True, dds_domain=SIM_DOMAIN,
                          dds_interface=SIM_IFACE, mode="sim", **kwargs)
    return make_robot(connect_dds=True, mode="debug", **kwargs)


def confirm(step: str, auto: bool):
    """Operator gate before a step. Enter -> run; 'q'/Ctrl-D -> abort (raises
    KeyboardInterrupt -> the caller HOLDS position, no recovery motion). auto=True skips."""
    if auto:
        return
    try:
        with LOG.span(f"wait: {step}", LOG.WAIT):   # human keyboard gate -- NOT pipeline speed
            ans = input(f"  >> next: {step} -- Enter to run, 'q' to abort: ").strip().lower()
    except EOFError:
        raise KeyboardInterrupt("stdin closed")
    if ans in ("q", "quit", "n", "no", "abort"):
        raise KeyboardInterrupt(f"operator aborted before '{step}'")


def do_move(robot, side, goal, label):
    """move() to a wrist goal, then -- after letting the PD CONVERGE -- print the
    achieved-vs-target error (wrist FK vs the commanded goal, pos mm + orientation deg).
    run() returns when the trajectory clock ends, BEFORE the arm finishes settling, so
    settle to the final commanded config first or the error reads high. Shared by 07/09."""
    r = P.move(robot, side, goal)
    robot.executor.settle(robot.arm.q_target, tol=0.02, timeout=1.5)
    ach = robot.planner.fk(side, robot.arm.get_current_dual_arm_q())
    dp_mm = (goal.translation - ach.translation) * 1000.0
    R_err = goal.rotation.T @ ach.rotation
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))))
    print(f"{label:8s}: {r} | reached err: pos {np.linalg.norm(dp_mm):.1f} mm "
          f"{np.round(dp_mm, 1).tolist()}, rot {ang:.1f} deg")
    return r


class Viewer:
    """cv2 live window; falls back to periodic frame saves when there's no display."""

    def __init__(self, name="feed", save_path="/tmp/feed.png", save_every=15):
        self.name, self.save_path, self.save_every = name, save_path, save_every
        self._n = 0
        self._headless = False
        try:
            import cv2
            self._cv2 = cv2
        except Exception:
            self._cv2, self._headless = None, True

    def show(self, bgr) -> bool:
        """Display one BGR frame. Returns False if the user pressed 'q' (quit)."""
        if bgr is None or self._cv2 is None:
            return True
        cv2 = self._cv2
        if not self._headless:
            try:
                cv2.imshow(self.name, bgr)
                return (cv2.waitKey(1) & 0xFF) != ord("q")
            except cv2.error:
                self._headless = True
                print(f"[viewer] no display -> saving frames to {self.save_path}")
        self._n += 1
        if self._n % self.save_every == 0:
            cv2.imwrite(self.save_path, bgr)
        return True

    def close(self):
        if self._cv2 is not None and not self._headless:
            try:
                self._cv2.destroyAllWindows()
            except Exception:
                pass
