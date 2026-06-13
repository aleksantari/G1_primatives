#!/usr/bin/env python
"""[HARDWARE] Open/close each Dex3 hand and stream q / tau_est / press during the
close, then report grasp verification. Phase-1 + Phase-4 acceptance.

    python scripts/04_hand_check.py --side left
Place a block in the hand to see grasped=True; close on air for grasped=False.
"""
import argparse
import time

import numpy as np

from g1_classical_manip.factory import make_robot
from g1_classical_manip.ee.hand_base import LEFT, RIGHT


def stream(hand, side, secs=2.0):
    t0 = time.time()
    while time.time() - t0 < secs:
        st = hand.get_state(side)
        print(f"  q={np.round(st['q'],2)} tau={np.round(st['tau'],2)} "
              f"press={np.round(st['press'],1)}")
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=[LEFT, RIGHT], default=LEFT)
    args = ap.parse_args()

    robot = make_robot(connect_dds=True, build_perception=False)
    hand = robot.hand
    side = args.side

    print(f"[{side}] opening..."); hand.open(side, verify=True); stream(hand, side, 1.0)
    input("Place a block (or leave empty) and press Enter to close...")
    print(f"[{side}] closing...")
    grasped = hand.close(side, verify=True)
    stream(hand, side, 1.0)
    print(f"[{side}] grasped = {grasped}")
    input("Press Enter to release...")
    hand.open(side, verify=True)
    print("done.")


if __name__ == "__main__":
    main()
