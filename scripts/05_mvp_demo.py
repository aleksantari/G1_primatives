#!/usr/bin/env python
"""05 - MVP demo: the cuRobo-native action primitives, end to end.

    home -> move(RIGHT, +10 cm z) -> close_hand -> open_hand -> home

Dual-target. The arms move -- ensure clearance; on hardware start with a low
arm_velocity_limit and gravity_comp off (configs/{robot,planner}.yaml).

  sim : CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
        bash -ic 'use_conda g1_curobo && python scripts/05_mvp_demo.py --target sim'
  real: bash -ic 'use_conda g1_curobo && python scripts/05_mvp_demo.py --target real'
"""
import argparse

import numpy as np
import _rig
from g1_primitives import primitives as P


def main():
    ap = argparse.ArgumentParser()
    _rig.add_target_arg(ap)
    ap.add_argument("--abort", type=float, default=None,
                    help="override executor tracking-error abort threshold (rad)")
    args = ap.parse_args()

    robot = _rig.connect(args.target, connect_hand=True)
    # sim's PD lags on the fast initial homing move -> looser abort; real uses the
    # config default (planner.yaml, 0.20) unless overridden.
    if args.abort is not None:
        robot.executor.abort_thresh = args.abort
    elif args.target == "sim":
        robot.executor.abort_thresh = 0.40

    side = P.RIGHT
    print("home :", P.home(robot))

    q = robot.arm.get_current_dual_arm_q()
    cur = robot.planner.fk(side, q)
    goal = cur.copy()
    goal.translation = cur.translation + np.array([0.0, 0.0, 0.10])
    print("goal wrist pos:", np.round(goal.translation, 3))

    print("move :", P.move(robot, side, goal))
    print("close:", P.close_hand(robot, side))
    print("open :", P.open_hand(robot, side))
    print("home :", P.home(robot))
    print("DONE")


if __name__ == "__main__":
    main()
