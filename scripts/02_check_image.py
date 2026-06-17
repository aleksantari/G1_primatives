#!/usr/bin/env python
"""02 - Head-camera feed check. Pulls frames from the head camera (ZMQ) and shows a
live window ('q' to quit; no display -> saves to /tmp/head_feed.png).

Camera-only (no control DDS, nothing moves). Works against the SIM stream now; the
real-robot image client is not wired yet (HARDWARE_TODO). Camera host/port come from
configs/camera.yaml `stream` (sim: 127.0.0.1:55555).

  CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
  bash -ic 'use_conda g1_curobo && python scripts/02_check_image.py'
"""
import argparse
import time

import _rig
from g1_classical_manip.factory import make_robot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=30.0)
    args = ap.parse_args()

    robot = make_robot(connect_dds=False, connect_camera=True)   # camera only -> no motion
    print("camera shape (h,w):", robot.camera.shape())
    view = _rig.Viewer("head cam", save_path="/tmp/head_feed.png")

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
