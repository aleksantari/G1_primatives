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
from g1_primitives import primitives as P
from g1_primitives.ee.hand_base import LEFT, RIGHT


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
        r = P.move(robot, args.side, goal)
        # let the PD converge first: move()/run() returns when the trajectory clock ends,
        # before the arm finishes settling, so the FK error would otherwise read high.
        robot.executor.settle(robot.arm.q_target, tol=0.02, timeout=1.5)
        # achieved-vs-target error: wrist FK after the move vs the commanded goal
        ach = robot.planner.fk(args.side, robot.arm.get_current_dual_arm_q())
        dp_mm = (goal.translation - ach.translation) * 1000.0
        R_err = goal.rotation.T @ ach.rotation
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))))
        print(f"move {args.side} +{args.dz:.2f} z: {r} | reached err: pos "
              f"{np.linalg.norm(dp_mm):.1f} mm {np.round(dp_mm, 1).tolist()}, rot {ang:.1f} deg")
        print("home:", P.home(robot))


if __name__ == "__main__":
    main()
