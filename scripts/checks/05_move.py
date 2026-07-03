#!/usr/bin/env python
"""checks/05 - Arm motion check: home, then (optionally) a small Cartesian lift of one wrist.

Default is HOME ONLY. Dual-target. The arms move -- ensure clearance. On hardware,
start with a low arm_velocity_limit (configs/robot.yaml) and gravity_comp off.

  home only : ... python scripts/checks/05_move.py --target sim
  home+lift : ... python scripts/checks/05_move.py --target sim --side right --dz 0.10
"""
import argparse

import numpy as np

import g1_primitives as g1
from g1_primitives.api import console


def main():
    ap = argparse.ArgumentParser()
    console.add_target_arg(ap)
    ap.add_argument("--side", choices=[g1.LEFT, g1.RIGHT], default=g1.RIGHT)
    ap.add_argument("--dz", type=float, default=0.0,
                    help="after home, lift this wrist by dz metres in +z (0 = home only)")
    ap.add_argument("--gravity-scale", type=float, default=None,
                    help="enable cuRobo gravity-comp feed-forward at this scale "
                         "(start 0.5 to confirm the sign, ramp to 1.0; negative flips "
                         "the sign). Omit = use planner.yaml default (off).")
    ap.add_argument("--abort", type=float, default=None,
                    help="override executor tracking-error abort threshold (rad). "
                         "Raise (e.g. 0.30) for fast/large moves where gravity-only "
                         "feed-forward lets the arm lag the trajectory transiently.")
    ap.add_argument("--speed", type=float, default=None,
                    help="trajectory playback time-dilation (<1 = slower; same path/goal). "
                         "Lowers joint velocity & quadratically lowers acceleration so a "
                         "torque-limited arm can track the plan. Try 0.5 if a move aborts.")
    args = ap.parse_args()

    gkw = {} if args.gravity_scale is None else dict(
        gravity_comp=True, gravity_scale=args.gravity_scale)
    robot = g1.connect(args.target, camera=False, **gkw)
    robot.set_executor(speed=args.speed, abort_thresh_rad=args.abort)
    print(f"[{args.target}] home:", robot.home())

    if abs(args.dz) > 1e-6:
        q = robot.arm.get_current_dual_arm_q()
        cur = robot.planner.fk(args.side, q)
        goal = cur.copy()
        goal.translation = cur.translation + np.array([0.0, 0.0, args.dz])
        print(f"goal {args.side} wrist:", np.round(goal.translation, 3))
        console.do_move(robot, args.side, goal, f"lift+{args.dz:.2f}")
        print("home:", robot.home())


if __name__ == "__main__":
    main()
