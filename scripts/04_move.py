#!/usr/bin/env python
"""04 - Arm motion check: home, then (optionally) a small Cartesian lift of one wrist.

Default is HOME ONLY. Dual-target. The arms move -- ensure clearance. On hardware,
start with a low arm_velocity_limit (configs/robot.yaml) and gravity_comp off.

  home only : ... python scripts/04_move.py --target sim
  home+lift : ... python scripts/04_move.py --target sim --side right --dz 0.10
"""
import argparse

import numpy as np
import _rig
from g1_classical_manip import primitives as P
from g1_classical_manip.ee.hand_base import LEFT, RIGHT


def main():
    ap = argparse.ArgumentParser()
    _rig.add_target_arg(ap)
    ap.add_argument("--side", choices=[LEFT, RIGHT], default=RIGHT)
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
    robot = _rig.connect(args.target, connect_hand=False, **gkw)
    if args.abort is not None:
        robot.executor.abort_thresh = args.abort
    if args.speed is not None:
        robot.executor.time_dilation = args.speed
    print(f"[{args.target}] home:", P.home(robot))

    if abs(args.dz) > 1e-6:
        q = robot.arm.get_current_dual_arm_q()
        cur = robot.planner.fk(args.side, q)
        goal = cur.copy()
        goal.translation = cur.translation + np.array([0.0, 0.0, args.dz])
        print(f"goal {args.side} wrist:", np.round(goal.translation, 3))
        print(f"move {args.side} +{args.dz:.2f} z:", P.move(robot, args.side, goal))
        print("home:", P.home(robot))


if __name__ == "__main__":
    main()
