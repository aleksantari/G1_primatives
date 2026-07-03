#!/usr/bin/env python
"""checks/02 - Head-camera feed check. Pulls frames from the head camera and shows a live
window ('q' to quit; no display -> saves to /tmp/head_feed.png).

Camera-only (no control DDS, nothing moves).
  --target sim   Isaac mono 640x480 on :55555  (configs/camera_sim.yaml)
  --target real  ZED stereo, sliced to one 1280x720 eye  (configs/camera_real.yaml)

  bash -ic 'use_conda g1_curobo && python scripts/checks/02_camera.py --target sim'
"""
import argparse
import time

from g1_primitives import Robot
from g1_primitives.api import console


def main():
    ap = console.add_target_arg(argparse.ArgumentParser())
    ap.add_argument("--secs", type=float, default=30.0)
    args = ap.parse_args()

    robot = Robot.offline(args.target, camera=True)   # camera only -> no motion
    print(f"[{args.target}] camera shape (h,w):", robot.camera.shape())
    view = console.Viewer("head cam", save_path="/tmp/head_feed.png")

    t0, n = time.time(), 0
    while time.time() - t0 < args.secs:
        bgr = robot.camera.get_bgr_frame()
        if bgr is None:
            continue
        n += 1
        if not view.show(bgr):
            break
    view.close()
    print(f"shown {n} frames.")


if __name__ == "__main__":
    main()
