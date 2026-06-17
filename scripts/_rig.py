"""Shared helpers for the scripts/ ladder: sim/real target selection + a cv2 live
viewer with a headless fallback. (Run scripts as `python scripts/NN_*.py`; the
script's dir is on sys.path so `import _rig` resolves.)"""
from __future__ import annotations

import argparse
import os

from g1_classical_manip.factory import make_robot, load_configs

SIM_DOMAIN, SIM_IFACE = 1, "lo"


def add_target_arg(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--target", choices=["sim", "real"], default="sim",
                    help="sim = unitree_sim_isaaclab loopback (domain 1/lo, mode=sim); "
                         "real = robot.yaml DDS + mode=debug (auto Enter_Debug_Mode).")
    return ap


def dds_for(target: str):
    """(domain, interface) for a raw DDS connection (read-only scripts)."""
    if target == "sim":
        return SIM_DOMAIN, SIM_IFACE
    dds = load_configs()["robot"]["dds"]
    return dds["domain_id"], (dds.get("interface") or None)


def _warn_sim_uri(target: str):
    if target == "sim" and not os.environ.get("CYCLONEDDS_URI"):
        print("WARNING: CYCLONEDDS_URI is unset -- sim loopback DDS discovery will likely "
              "fail. Prefix the command with\n  CYCLONEDDS_URI=file://$PWD/configs/"
              "cyclonedds_loopback.xml")


def connect(target: str, **kwargs):
    """make_robot for the target: sim -> loopback DDS + mode=sim; real -> robot.yaml
    DDS + mode=debug (which auto-runs MotionSwitcher.Enter_Debug_Mode). NOTE: any
    connect_dds=True build starts the arm controller, so the arms drive to home on
    connect (velocity-clipped on hardware) -- ensure clearance."""
    _warn_sim_uri(target)
    if target == "sim":
        return make_robot(connect_dds=True, dds_domain=SIM_DOMAIN,
                          dds_interface=SIM_IFACE, mode="sim", **kwargs)
    return make_robot(connect_dds=True, mode="debug", **kwargs)


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
