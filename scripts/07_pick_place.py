#!/usr/bin/env python
"""07 - Pick-and-lift: the first composite task, composed from the action + perception
primitives. Single arm; the script itself is the state machine (no FSM framework).

    home -> open_hand -> detect(block) -> move(approach) -> move(grasp)
         -> close_hand -> move(lift) -> home

The grasp TARGET is the detected block POSITION plus a FIXED wrist orientation (the
detected cube orientation is ignored -- fragile / reachability) and tunable z-offsets.
move() targets the wrist-yaw link and there is NO palm/grasp-frame offset yet, so the
wrist-yaw sits above the block center -- tune --grasp-z / --approach / --lift (and the
GRASP_RPY constant) in sim so the hand actually reaches the block.

Dual-target. The arm moves -- ensure clearance. On real, gravity comp + time_dilation
0.5 apply automatically from config; the operator sets debug mode via the remote first.

  sim : CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
        bash -ic 'use_conda g1_curobo && python scripts/07_pick_place.py --target sim'
  real: bash -ic 'use_conda g1_curobo && python scripts/07_pick_place.py --target real'
"""
import argparse
import sys

import numpy as np
import _rig
from g1_classical_manip import primitives as P
from g1_classical_manip.ee.hand_base import LEFT, RIGHT
from g1_classical_manip.motion.curobo_planner import PlanningError
from g1_classical_manip.spatial.pose import Pose, rpy_to_matrix

# Optional FIXED grasp orientation (pelvis-frame roll/pitch/yaw, radians). None = keep
# the wrist's current FK orientation (guaranteed reachable). Set to a hand-tuned grasp
# once the geometry is dialed in, e.g. (np.pi, 0.0, 0.0) for a palm-down approach.
GRASP_RPY = None


def main():
    ap = argparse.ArgumentParser()
    _rig.add_target_arg(ap)
    ap.add_argument("--side", choices=[LEFT, RIGHT], default=RIGHT)
    ap.add_argument("--object", default="block")
    ap.add_argument("--approach", type=float, default=0.10,
                    help="pre-grasp height ABOVE the grasp pose (m, +z)")
    ap.add_argument("--grasp-z", type=float, default=0.05,
                    help="wrist-yaw z-offset above the block center at grasp (m); tune so "
                         "the fingers reach the block (no palm offset exists yet)")
    ap.add_argument("--lift", type=float, default=0.10,
                    help="lift height after grasp (m, +z)")
    ap.add_argument("--verify", action="store_true",
                    help="verify the grasp on close (hand presets are untuned -> off by default)")
    ap.add_argument("--abort", type=float, default=None,
                    help="override executor tracking-error abort threshold (rad)")
    ap.add_argument("--speed", type=float, default=None,
                    help="trajectory playback time-dilation (<1 = slower; same path/goal)")
    args = ap.parse_args()

    robot = _rig.connect(args.target, connect_hand=True, connect_camera=True,
                         camera_config=_rig.camera_config_for(args.target))
    # sim's PD lags on the fast initial homing move -> looser abort; real uses the config
    # default (planner.yaml 0.20 + time_dilation 0.5) unless overridden.
    if args.abort is not None:
        robot.executor.abort_thresh = args.abort
    elif args.target == "sim":
        robot.executor.abort_thresh = 0.40
    if args.speed is not None:
        robot.executor.time_dilation = args.speed

    side = args.side
    rc = 0
    try:
        # 1. ready: planned home, then open the grasping hand
        r = P.home(robot)
        print("home  :", r)
        if not r.ok:
            raise RuntimeError(f"home failed: {r.info}")
        print("open  :", P.open_hand(robot, side))

        # 2. detect the block (pelvis-frame body-center pose)
        det = P.detect(robot, target=args.object)
        if det is None:
            raise RuntimeError(f"detect('{args.object}') found nothing")
        block_xyz = det.pose.translation.copy()
        print(f"detect: {args.object} @ pelvis {np.round(block_xyz, 3)} m (score {det.score:.1f})")

        # 3. fixed grasp orientation: hand-tuned GRASP_RPY, else the wrist's current FK
        #    orientation (guaranteed reachable). Position comes from the detection.
        q = robot.arm.get_current_dual_arm_q()
        grasp_R = (rpy_to_matrix(*GRASP_RPY) if GRASP_RPY is not None
                   else robot.planner.fk(side, q).rotation.copy())

        grasp = Pose(rotation=grasp_R, translation=block_xyz + np.array([0.0, 0.0, args.grasp_z]))
        approach = grasp.copy()
        approach.translation = grasp.translation + np.array([0.0, 0.0, args.approach])
        lift = grasp.copy()
        lift.translation = grasp.translation + np.array([0.0, 0.0, args.lift])
        print("wrist goals (m):  approach", np.round(approach.translation, 3),
              "| grasp", np.round(grasp.translation, 3),
              "| lift", np.round(lift.translation, 3))

        # 4. approach -> descend -> close -> lift -> home
        r = P.move(robot, side, approach)
        print("approach:", r)
        if not r.ok:
            raise RuntimeError(f"approach move failed: {r.info}")

        r = P.move(robot, side, grasp)
        print("descend :", r)
        if not r.ok:
            raise RuntimeError(f"descend move failed: {r.info}")

        print("close   :", P.close_hand(robot, side, verify=args.verify))

        r = P.move(robot, side, lift)
        print("lift    :", r)
        if not r.ok:
            raise RuntimeError(f"lift move failed: {r.info}")

        print("home  :", P.home(robot))
        print("DONE")
    except Exception as e:                       # noqa: BLE001 - top-level task recovery
        print(f"\nPICK-PLACE ABORTED ({type(e).__name__}): {e}")
        print("recovering -> open hand + home ...")
        try:
            P.open_hand(robot, side)
            P.home(robot)
        except Exception as e2:                  # noqa: BLE001 - best-effort recovery
            print(f"  recovery best-effort failed: {e2}")
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
