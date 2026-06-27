#!/usr/bin/env python
"""09 - Pick-and-lift driven by a GraspSource (AprilTag A-B reference OR GraspGenX), with
optional SAM3 segmentation of the target object.

Same shape as 07_pick_place, but the grasp comes from `robot.grasp_source`:

    home -> open -> grasp_source.grasps(object) -> [select 1 reachable candidate]
         -> move(approach) -> move(grasp) -> close -> move(lift) -> home

  --source apriltag    one candidate = detected pose + URDF palm offset (the known-good ref)
  --source graspgenx   ranked 6-DoF grasps from the GraspGenX ZMQ service (needs the server
                       up + the real depth stream; sim has no depth -> fails loudly)
  --segment interactive  refine a SAM3 mask (text/box/points) in a cv2 GUI before the cloud
            auto         one SAM3 call with grasp.yaml default_prompt
            none         no segmentation (whole frame)   [overrides grasp.yaml segment.mode]

ONE candidate is selected up front and its approach/grasp/lift are executed consistently
(avoids switching candidates mid-sequence). `--select` picks which:
  reachable  highest-confidence candidate whose GRASP pose plans (skips unreachable tops)
  top        THE top-confidence grasp -- the one viser draws the gripper mesh on -- with no
             reachability fallback (it just warns if it won't plan; approach/descend then fail
             loudly). Use this to execute exactly the grasp the GUI highlights.
The pre-grasp backs off along the grasp's APPROACH axis (so a 6-DoF GraspGenX grasp is
approached along its own line, not straight down). Operator-gated each step; on any failure it
opens + homes. On real: gravity comp + time_dilation apply from config; operator sets debug
mode via the remote first. Watch the e-stop.

  real: bash -ic 'use_conda g1_curobo && python scripts/09_graspgen.py --target real --source graspgenx --segment interactive'
"""
import argparse
import sys
import time

import numpy as np
import _rig
from g1_classical_manip import primitives as P
from g1_classical_manip.ee.hand_base import LEFT, RIGHT
from g1_classical_manip.motion.curobo_planner import PlanningError


def _shift_z(pose, dz: float):
    """Copy a wrist goal Pose shifted by `dz` along world +z (pelvis frame)."""
    p = pose.copy()
    p.translation = pose.translation + np.array([0.0, 0.0, dz])
    return p


def _approach_pose(grasp_wrist, grasp_pose, dist: float):
    """Pre-grasp: back off `dist` along the grasp APPROACH axis (grasp +Z = into the object)
    when the 6-DoF grasp frame is known (GraspGenX), else straight up (world +z; the AprilTag
    top-down case). So the descend sweeps along the gripper's approach line, not diagonally."""
    axis = grasp_pose.rotation[:, 2] if grasp_pose is not None else np.array([0.0, 0.0, -1.0])
    p = grasp_wrist.copy()
    p.translation = grasp_wrist.translation - dist * np.asarray(axis, float)
    return p


def _finger_contact_check(robot, side, wrist_goal, object_center):
    """SIM verification the viser mesh overlay can't do: FK OUR Dex3 fingertips at `wrist_goal`
    + the power_close preset and report how close our grasp contact lands to the TRUE object
    center (the sim GT cube). Compared against GT -- NOT the grasp pose, which our contact hits
    by construction of the grasp->wrist transform. `object_center` is the pelvis-frame cube
    center. Validates GraspGenX's pick + our derived transform + our real finger geometry."""
    from g1_classical_manip.ee.hand_kinematics import Dex3Kinematics
    from g1_classical_manip.spatial.pose import Pose
    q7 = robot.cfg["hands"]["dex3"]["presets"]["power_close"][side]
    kin = Dex3Kinematics()
    tips = kin.fingertips(side, q7)
    to_pelvis = lambda p: (wrist_goal * Pose(np.eye(3), p)).translation
    c = to_pelvis(kin.contact_point(side, q7))
    fm = 0.5 * (to_pelvis(tips["index"]) + to_pelvis(tips["middle"]))
    d = float(np.linalg.norm(c - object_center))
    th = float(np.linalg.norm(to_pelvis(tips["thumb"]) - object_center))
    fd = float(np.linalg.norm(fm - object_center))
    tag = "OK" if d < 0.03 else "WARN"
    print(f"FK check [{tag}]: contact {np.round(c, 3)} vs cube {np.round(object_center, 3)} "
          f"-> {d * 1000:.0f} mm | thumb {th * 1000:.0f} mm, fingers {fd * 1000:.0f} mm from center")


def _select_candidate(robot, side, candidates, grasp_z):
    """Highest-confidence candidate whose GRASP pose plans from the current config (plan-
    only -- no motion). Returns (candidate, grasp_pose) or (None, None)."""
    q = robot.arm.get_current_dual_arm_q()
    for c in candidates:
        gp = _shift_z(c.wrist_goal, grasp_z)
        try:
            robot.planner.plan_to_pose(q, side, gp)      # reachability probe; moves nothing
        except PlanningError:
            continue
        return c, gp
    return None, None


def main():
    ap = argparse.ArgumentParser()
    _rig.add_target_arg(ap)
    ap.add_argument("--source", choices=["apriltag", "graspgenx", "sim_cloud"], default=None,
                    help="grasp source (overrides grasp.yaml grasp_source; sim_cloud = sim GT cloud)")
    ap.add_argument("--segment", choices=["auto", "interactive", "none"], default=None,
                    help="SAM3 segmentation mode (overrides grasp.yaml segment.mode)")
    ap.add_argument("--visualize", action="store_true",
                    help="show cloud + ranked grasps in a viser GUI (graspgenx source)")
    ap.add_argument("--select", choices=["reachable", "top"], default="reachable",
                    help="reachable = highest-confidence candidate that plans; "
                         "top = THE top-confidence grasp (the viser mesh-overlay best), no fallback")
    ap.add_argument("--side", choices=[LEFT, RIGHT], default=RIGHT)
    ap.add_argument("--object", default="block")
    ap.add_argument("--approach", type=float, default=0.10,
                    help="pre-grasp back-off along the grasp approach axis (m)")
    ap.add_argument("--grasp-z", type=float, default=0.0,
                    help="vertical offset added to the grasp pose (m, +z)")
    ap.add_argument("--lift", type=float, default=0.10, help="lift height after grasp (m, +z)")
    ap.add_argument("--verify", action="store_true", help="verify the grasp on close")
    ap.add_argument("--abort", type=float, default=None,
                    help="override executor tracking-error abort threshold (rad)")
    ap.add_argument("--speed", type=float, default=None,
                    help="trajectory playback time-dilation (<1 = slower)")
    ap.add_argument("--no-confirm", action="store_true", help="skip the per-step prompt")
    ap.add_argument("--close-frac", type=float, default=0.5, help="hand close fraction")
    args = ap.parse_args()

    robot = _rig.connect(args.target, connect_hand=True, connect_camera=True,
                         camera_config=_rig.camera_config_for(args.target))
    if args.source or args.segment is not None or args.visualize:   # overrides of grasp.yaml
        from g1_classical_manip.factory import _build_grasp_source
        if args.source:
            robot.cfg["grasp"]["grasp_source"] = args.source
        if args.segment is not None:
            robot.cfg["grasp"].setdefault("segment", {})["mode"] = (
                None if args.segment == "none" else args.segment)
        if args.visualize:
            robot.cfg["grasp"].setdefault("graspgenx", {}).setdefault(
                "visualize", {})["enabled"] = True
        robot.grasp_source = _build_grasp_source(robot.frames, robot.cfg)
    src_kind = robot.cfg["grasp"].get("grasp_source", "apriltag")
    seg_mode = (robot.cfg["grasp"].get("segment", {}) or {}).get("mode")
    if args.abort is not None:
        robot.executor.abort_thresh = args.abort
    elif args.target == "sim":
        robot.executor.abort_thresh = 0.40
    if args.speed is not None:
        robot.executor.time_dilation = args.speed

    side, auto, rc = args.side, args.no_confirm, 0
    try:
        _rig.confirm("home (planned)", auto)
        r = P.home(robot)
        print("home  :", r)
        if not r.ok:
            raise RuntimeError(f"home failed: {r.info}")

        _rig.confirm("open hand", auto)
        print("open  :", P.open_hand(robot, side))

        _rig.confirm(f"generate grasps via '{src_kind}' (segment={seg_mode}) -- object in view",
                     auto)
        cam = getattr(robot, "camera", None)
        if src_kind == "graspgenx" and cam is not None:   # prime cold/lazy ZED streams so the
            t0 = time.time()                              # first depth capture in grasps() works
            while time.time() - t0 < 6.0:
                if cam.get_rgb_frame() is not None and cam.get_depth_frame() is not None:
                    break
                time.sleep(0.05)
        cands = robot.grasp_source.grasps(robot, side, args.object)
        print(f"grasps: {src_kind} -> {len(cands)} candidate(s) for '{args.object}'")
        if not cands:
            raise RuntimeError(f"grasp source '{src_kind}' produced no candidates "
                               f"(no detection / no depth / no mask / no grasps)")
        if args.select == "top":
            chosen = cands[0]                            # confidence-sorted -> [0] = the viser best
            grasp = _shift_z(chosen.wrist_goal, args.grasp_z)
            try:                                          # probe but DO NOT fall back -- just warn
                robot.planner.plan_to_pose(robot.arm.get_current_dual_arm_q(), side, grasp)
                reach = "plans OK"
            except PlanningError:
                reach = "WILL NOT PLAN -- approach/descend will fail"
            print(f"select=top: top grasp confidence {chosen.confidence:.3f} -- {reach}")
        else:
            chosen, grasp = _select_candidate(robot, side, cands, args.grasp_z)
            if chosen is None:
                raise RuntimeError(f"none of the {len(cands)} candidates plan to a reachable grasp")
        viz = getattr(robot.grasp_source, "viz", None)   # green = the grasp we chose
        if viz is not None and chosen.grasp_pose is not None:
            viz.mark_chosen(chosen.grasp_pose.homogeneous)
        approach = _approach_pose(grasp, chosen.grasp_pose, args.approach)
        lift = _shift_z(grasp, args.lift)
        print(f"chosen: confidence {chosen.confidence:.3f} | grasp wrist "
              f"{np.round(grasp.translation, 3)} m")
        if src_kind == "sim_cloud":      # GT cube -> FK-verify our fingers land on the object
            try:
                ps = getattr(robot.grasp_source, "pose_source", None)
                gt = ps.block_pose(args.object) if ps is not None else None
                if gt is not None:
                    _finger_contact_check(robot, side, grasp, gt.translation)
            except Exception as e:       # noqa: BLE001 - diagnostic only, never blocks the grasp
                print(f"FK check skipped: {e}")

        _rig.confirm("move to APPROACH", auto)
        if not _rig.do_move(robot, side, approach, "approach").ok:
            raise RuntimeError("approach move failed")

        _rig.confirm("move to GRASP (descend)", auto)
        if not _rig.do_move(robot, side, grasp, "descend").ok:
            raise RuntimeError("descend move failed")

        _rig.confirm(f"CLOSE hand to {args.close_frac:.2f}", auto)
        print("close :", P.close_hand(robot, side, verify=args.verify, fraction=args.close_frac))

        _rig.confirm("move to LIFT", auto)
        if not _rig.do_move(robot, side, lift, "lift").ok:
            raise RuntimeError("lift move failed")

        _rig.confirm("home (return)", auto)
        print("home  :", P.home(robot))
        print("DONE")
    except KeyboardInterrupt as e:
        print(f"\nABORTED by operator ({e}) -- holding position; no recovery motion.")
        rc = 1
    except Exception as e:                       # noqa: BLE001 - top-level task recovery
        print(f"\nGRASP-GEN ABORTED ({type(e).__name__}): {e}")
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
