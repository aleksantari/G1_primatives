#!/usr/bin/env python
"""[HARDWARE] Report the pelvis-frame block pose from the head camera; move the
tag by hand-measured steps and compare reported deltas (Phase-3 acceptance:
within +/-5 mm of 10 cm steps).

    python scripts/03_static_perception_check.py --host 192.168.123.164 --tag 0
"""
import argparse
import time

import numpy as np

from g1_classical_manip.factory import make_robot
from g1_classical_manip.image_server.image_client import HeadCamera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.123.164")
    ap.add_argument("--tag", type=int, default=0)
    ap.add_argument("--backend", default="teleimager")
    args = ap.parse_args()

    robot = make_robot(connect_dds=False, build_perception=True)
    cam = HeadCamera(host=args.host, backend=args.backend)
    perc = robot.perception
    print("Move the tag and read the pelvis-frame pose. Ctrl-C to stop.")
    prev = None
    try:
        while True:
            for _ in range(perc.median_frames):
                g = cam.get_gray_frame()
                if g is not None:
                    perc.detect(g)
                time.sleep(0.03)
            T = perc.block_pose(args.tag)
            if T is None:
                print("no stable detection"); time.sleep(0.5); continue
            p = T.translation
            msg = f"block @ pelvis = {np.round(p, 4)}"
            if prev is not None:
                msg += f"   delta = {np.round((p - prev) * 1000, 1)} mm"
            print(msg)
            prev = p
            time.sleep(0.7)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
