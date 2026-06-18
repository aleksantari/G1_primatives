#!/usr/bin/env python
"""07 - Pick-and-lift: the first composite task, composed from the action + perception
primitives. Single arm; the script itself is the state machine (no FSM framework).

    home -> open_hand -> detect(block) -> move(approach) -> move(grasp)
         -> close_hand -> move(lift) -> home

The grasp TARGET is the detected block POSITION plus a FIXED orientation (the detected
cube orientation is ignored -- fragile / reachability) and tunable z-offsets. The true
end-effector is the PALM (grasp center = midpoint of the index/middle finger bases), which
sits the palm offset ahead of the wrist-yaw link that move() targets, so we command the
wrist-yaw backed off by that offset (palm offset, measured from the URDF per side, +
GRASP_OFFSET, the contact margin past the finger bases). Tune GRASP_OFFSET / GRASP_RPY /
--grasp-z / --approach / --lift in sim so the hand reaches the block.

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
from g1_classical_manip.spatial.pose import Pose, rpy_to_matrix

# Optional FIXED grasp orientation (pelvis-frame roll/pitch/yaw, radians). None = keep
# the wrist's current FK orientation (guaranteed reachable). Set to a hand-tuned grasp
# once the geometry is dialed in, e.g. (np.pi, 0.0, 0.0) for a palm-down approach.
GRASP_RPY = None

# Palm offset = wrist_yaw -> palm grasp center, taken as the MIDPOINT of the index_0 &
# middle_0 finger bases, measured from the URDF at q=0. Rotation is identity, so it's a
# pure translation in the WRIST frame (+x forward to the fingers). x & z are common to
# both hands; the lateral y mirrors by side (right -y, left +y). move() targets the
# wrist-yaw, so we back the wrist off by this so the palm center lands on the target.
# (Promote into the planner later so every move() targets the palm.)
PALM_OFFSET_XYZ = np.array([0.1192, -0.0346, 0.0])   # |y|; sign set per side by _palm_offset

# Grasp/contact offset beyond the palm center, along palm +x (toward the fingertips): the
# object sits ~here when grasped, a bit past the finger-base midpoint. Side-agnostic.
GRASP_OFFSET = np.array([0.00, 0.00, 0.02])


def _palm_offset(side: str) -> np.ndarray:
    """wrist_yaw -> palm grasp center for `side` (y mirrors: right -y, left +y)."""
    x, y, z = PALM_OFFSET_XYZ
    return np.array([x, y if side == LEFT else -y, z])


def _wrist_goal(rotation, target_xyz, offset) -> Pose:
    """Wrist-yaw goal Pose (pelvis frame) that lands the palm grasp center (`offset` ahead
    of the wrist-yaw in the wrist frame) at target_xyz with `rotation`:
    wrist = target - R @ offset."""
    off = np.asarray(offset, float)
    wrist_xyz = np.asarray(target_xyz, float) + np.asarray(rotation, float) @ (-off)
    return Pose(rotation=rotation, translation=wrist_xyz)


def _confirm(step: str, auto: bool):
    """Operator gate before a step. Enter -> run; 'q' / Ctrl-D -> abort (raises
    KeyboardInterrupt, which stops cleanly and HOLDS position -- no recovery motion).
    auto=True skips the prompt (the --no-confirm fast path)."""
    if auto:
        return
    try:
        ans = input(f"  >> next: {step} -- Enter to run, 'q' to abort: ").strip().lower()
    except EOFError:
        raise KeyboardInterrupt("stdin closed")
    if ans in ("q", "quit", "n", "no", "abort"):
        raise KeyboardInterrupt(f"operator aborted before '{step}'")


def main():
    ap = argparse.ArgumentParser()
    _rig.add_target_arg(ap)
    ap.add_argument("--side", choices=[LEFT, RIGHT], default=RIGHT)
    ap.add_argument("--object", default="block")
    ap.add_argument("--approach", type=float, default=0.0,
                    help="pre-grasp height ABOVE the grasp pose (m, +z)")
    ap.add_argument("--grasp-z", type=float, default=0.0,
                    help="PALM height above the block CENTER at grasp (m, +z); 0 = palm at "
                         "the center. The wrist-palm gap is handled by PALM_OFFSET.")
    ap.add_argument("--lift", type=float, default=0.10,
                    help="lift height after grasp (m, +z)")
    ap.add_argument("--verify", action="store_true",
                    help="verify the grasp on close (hand presets are untuned -> off by default)")
    ap.add_argument("--abort", type=float, default=None,
                    help="override executor tracking-error abort threshold (rad)")
    ap.add_argument("--speed", type=float, default=None,
                    help="trajectory playback time-dilation (<1 = slower; same path/goal)")
    ap.add_argument("--no-confirm", action="store_true",
                    help="run straight through without the per-step Enter prompt")
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
    auto = args.no_confirm
    rc = 0
    try:
        # 1. ready: planned home, then open the grasping hand
        _confirm("home (planned)", auto)
        r = P.home(robot)
        print("home  :", r)
        if not r.ok:
            raise RuntimeError(f"home failed: {r.info}")

        _confirm("open hand", auto)
        print("open  :", P.open_hand(robot, side))

        # 2. detect the block (pelvis-frame body-center pose)
        _confirm("detect block (place it in view first)", auto)
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

        # grasp-point targets (pelvis frame): grasp point at the block center (+grasp_z),
        # approach above it, lift above it. _wrist_goal backs the wrist-yaw off by the tool
        # offset (palm offset + grasp offset) so the GRASP POINT lands on the target.
        off = _palm_offset(side) + GRASP_OFFSET
        grasp_pt = block_xyz + np.array([0.0, 0.0, args.grasp_z])
        grasp    = _wrist_goal(grasp_R, grasp_pt, off)
        approach = _wrist_goal(grasp_R, grasp_pt + np.array([0.0, 0.0, args.approach]), off)
        lift     = _wrist_goal(grasp_R, grasp_pt + np.array([0.0, 0.0, args.lift]), off)
        print(f"grasp-point target (m): {np.round(grasp_pt, 3)}  "
              f"(tool offset[{side}] {np.round(off, 4).tolist()})")
        print("wrist goals (m): approach", np.round(approach.translation, 3),
              "| grasp", np.round(grasp.translation, 3),
              "| lift", np.round(lift.translation, 3))

        # 4. approach -> descend -> close -> lift -> home (operator-gated each step)
        _confirm("move to APPROACH", auto)
        r = P.move(robot, side, approach)
        print("approach:", r)
        if not r.ok:
            raise RuntimeError(f"approach move failed: {r.info}")

        _confirm("move to GRASP (descend)", auto)
        r = P.move(robot, side, grasp)
        print("descend :", r)
        if not r.ok:
            raise RuntimeError(f"descend move failed: {r.info}")

        _confirm("CLOSE hand", auto)
        print("close   :", P.close_hand(robot, side, verify=args.verify))

        _confirm("move to LIFT", auto)
        r = P.move(robot, side, lift)
        print("lift    :", r)
        if not r.ok:
            raise RuntimeError(f"lift move failed: {r.info}")

        _confirm("home (return)", auto)
        print("home  :", P.home(robot))
        print("DONE")
    except KeyboardInterrupt as e:
        print(f"\nABORTED by operator ({e}) -- holding position; no recovery motion.")
        rc = 1
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
