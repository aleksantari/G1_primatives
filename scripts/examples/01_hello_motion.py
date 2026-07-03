#!/usr/bin/env python
"""examples/01 - Hello, motion: the action primitives end to end via the Robot facade.

    home -> move(RIGHT, +10 cm z) -> close_hand -> open_hand -> home

Dual-target. The arms move -- ensure clearance; on hardware start with a low
arm_velocity_limit and gravity_comp off (configs/{robot,planner}.yaml).

  sim : CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
        bash -ic 'use_conda g1_curobo && python scripts/examples/01_hello_motion.py --target sim'
  real: bash -ic 'use_conda g1_curobo && python scripts/examples/01_hello_motion.py --target real'
"""
import argparse

import numpy as np

import g1_primitives as g1
from g1_primitives.api import console


def main():
    ap = argparse.ArgumentParser()
    console.add_target_arg(ap)
    ap.add_argument("--abort", type=float, default=None,
                    help="override executor tracking-error abort threshold (rad)")
    args = ap.parse_args()

    robot = g1.connect(args.target, camera=False)
    # sim's PD lags on the fast initial homing move -> looser abort; real uses the
    # config default (planner.yaml, 0.20) unless overridden.
    robot.set_executor(abort_thresh_rad=(args.abort if args.abort is not None
                                         else (0.40 if args.target == "sim" else None)))

    side = g1.RIGHT
    print("home :", robot.home())

    q = robot.arm.get_current_dual_arm_q()
    cur = robot.planner.fk(side, q)
    goal = cur.copy()
    goal.translation = cur.translation + np.array([0.0, 0.0, 0.10])
    print("goal wrist pos:", np.round(goal.translation, 3))

    print("move :", robot.move(side, goal))
    print("close:", robot.close_hand(side))
    print("open :", robot.open_hand(side))
    print("home :", robot.home())
    print("DONE")


if __name__ == "__main__":
    main()
