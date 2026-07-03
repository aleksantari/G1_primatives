#!/usr/bin/env python
"""checks/04 - Hand primitive check: run close_hand then open_hand (the actual primitives)
on one side and report whether the joint state responded.

Dual-target. NOTE: connecting builds the arm controller too, so the arms drive to home
on connect (velocity-clipped on hardware) -- ensure clearance. For low-level command->
state debugging (raw presets, per-finger tau/press), use scripts/tools/hand_loop_diag.py.

  sim : CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
        bash -ic 'use_conda g1_curobo && python scripts/checks/04_hands.py --target sim --side right'
  real: bash -ic 'use_conda g1_curobo && python scripts/checks/04_hands.py --target real --side right'
"""
import argparse

import numpy as np

import g1_primitives as g1
from g1_primitives.api import console


def _q(hand, side):
    return hand.get_state(side)["q"].copy()


def main():
    ap = argparse.ArgumentParser()
    console.add_target_arg(ap)
    ap.add_argument("--side", choices=[g1.LEFT, g1.RIGHT], default=g1.RIGHT)
    args = ap.parse_args()

    robot = g1.connect(args.target, camera=False)
    side = args.side
    hand = robot.hand

    q0 = _q(hand, side)
    print(f"[{args.target}] close_hand({side}):", robot.close_hand(side))
    qc = _q(hand, side)
    print(f"open_hand({side}):", robot.open_hand(side))
    qo = _q(hand, side)

    moved_close = float(np.max(np.abs(qc - q0)))
    moved_open = float(np.max(np.abs(qo - qc)))
    print(f"\nVERDICT ({side}): close moved {moved_close:.2f} rad, open moved {moved_open:.2f} rad")
    print("  q open->close->open:", np.round(q0, 2), np.round(qc, 2), np.round(qo, 2))
    if moved_close < 0.05 and moved_open < 0.05:
        print("  => hand did NOT move (check the stream/topic and that the sim has dex3 dds).")
    else:
        print("  => hand command->state loop OK.")


if __name__ == "__main__":
    main()
