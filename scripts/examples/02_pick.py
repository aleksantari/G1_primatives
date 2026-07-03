#!/usr/bin/env python
"""examples/02 - Pick-and-lift via the Robot facade: grasp source (GraspGenX live depth, or
the sim_cloud GT path in sim) + optional SAM3 segmentation + cuRobo native plan_grasp.

    home -> open -> robot.grasp_candidates(object) -> grasp_motion (goalset: cuRobo picks
    the feasible grasp) -> approach -> grasp -> close -> lift -> home

  --source graspgenx    ranked 6-DoF grasps from the GraspGenX ZMQ service (needs the server
                        up + a head depth stream)
  --source sim_cloud    sim GT: cube cloud from rt/sim_state -> GraspGenX (no camera/SAM3)
  --segment auto|interactive|none    SAM3 mask mode (overrides grasp.yaml segment.mode)
  --collision-world     depth-ESDF world so the approach routes around the table/clutter

Operator-gated each step (--no-confirm to skip); on any failure it opens + homes. On real:
gravity comp + time_dilation apply from config; debug mode is checked/entered on connect.
Watch the e-stop.

  real: bash -ic 'use_conda g1_curobo && python scripts/examples/02_pick.py --target real --segment interactive'
  sim : CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
        bash -ic 'use_conda g1_curobo && python scripts/examples/02_pick.py --target sim --source sim_cloud'
"""
import argparse
import sys

import numpy as np

import g1_primitives as g1
from g1_primitives.api import console
from g1_primitives.api import primitives as P


def _finger_contact_check(robot, side, wrist_goal, object_center):
    """SIM verification the viser mesh overlay can't do: FK OUR Dex3 fingertips at `wrist_goal`
    + the power_close preset and report how close our grasp contact lands to the TRUE object
    center (the sim GT cube). Validates the grasp pick + the derived tool transform + our real
    finger geometry. (Generalized into perception.validation by the validate tool.)"""
    from g1_primitives.ee.hand_kinematics import Dex3Kinematics
    q7 = robot.cfg["hands"]["dex3"]["presets"]["power_close"][side]
    kin = Dex3Kinematics()
    tips = kin.fingertips(side, q7)
    to_pelvis = lambda p: (wrist_goal * g1.Pose(np.eye(3), p)).translation
    c = to_pelvis(kin.contact_point(side, q7))
    d = float(np.linalg.norm(c - object_center))
    tag = "OK" if d < 0.03 else "WARN"
    print(f"FK check [{tag}]: contact {np.round(c, 3)} vs cube {np.round(object_center, 3)} "
          f"-> {d * 1000:.0f} mm")


def main():
    ap = argparse.ArgumentParser()
    console.add_target_arg(ap)
    ap.add_argument("--source", choices=["graspgenx", "sim_cloud"], default=None,
                    help="override grasp.yaml source: (graspgenx = live depth; sim_cloud = sim GT)")
    ap.add_argument("--segment", choices=["auto", "interactive", "none"], default=None,
                    help="SAM3 segmentation mode (overrides grasp.yaml segment.mode)")
    ap.add_argument("--visualize", action="store_true",
                    help="show cloud + ranked grasps (+ the ESDF world) in a viser GUI")
    ap.add_argument("--collision-world", action="store_true",
                    help="build the depth-ESDF collision world so the approach routes AROUND "
                         "the table/clutter (needs a head depth stream: checks/03_depth green)")
    ap.add_argument("--select", choices=["reachable", "first"], default="reachable",
                    help="reachable = feed ALL candidates (cuRobo picks the feasible one); "
                         "first = only the #1 highest-CONFIDENCE grasp (no selection)")
    ap.add_argument("--grasp-only", action="store_true",
                    help="skip approach AND lift: one free-space move straight to the grasp "
                         "pose, then close (grasp-frame sanity check)")
    ap.add_argument("--side", choices=[g1.LEFT, g1.RIGHT], default=g1.RIGHT)
    ap.add_argument("--object", default="block")
    ap.add_argument("--grasp-z", type=float, default=0.0,
                    help="vertical offset added to every grasp pose (m, +z)")
    ap.add_argument("--verify", action="store_true", help="verify the grasp on close")
    ap.add_argument("--close-frac", type=float, default=0.65, help="hand close fraction")
    ap.add_argument("--abort", type=float, default=None,
                    help="override executor tracking-error abort threshold (rad)")
    ap.add_argument("--speed", type=float, default=None,
                    help="trajectory playback time-dilation (<1 = slower)")
    ap.add_argument("--no-confirm", action="store_true", help="skip the per-step prompt")
    ap.add_argument("--diagnose", action="store_true",
                    help="on a plan/exec failure, explain WHY (self/world collision, joint "
                         "limits, singularity); with --visualize, overlay offending spheres")
    ap.add_argument("--diagnose-k", type=int, default=0,
                    help="top candidates --diagnose sweeps on failure (0 = ALL; ~0.8s each)")
    ap.add_argument("--debug-planner", action="store_true",
                    help="turn up cuRobo's own logger (prints its internal failure reasons)")
    args = ap.parse_args()

    robot = g1.connect(args.target)                       # camera on by default
    if args.debug_planner:
        from g1_primitives.motion.diagnostics import set_curobo_log_level
        set_curobo_log_level("debug")
    if args.source:
        robot.set_grasp_source(args.source)
    if args.segment is not None:
        robot.set_segmenter("none" if args.segment == "none" else args.segment)
    if args.visualize:
        robot.set_visualize(True)
    if args.collision_world:
        robot.set_collision_world(True)
        print("collision world: ON (head depth -> cuRobo ESDF; approach is obstacle-aware)")
    robot.set_executor(speed=args.speed,
                       abort_thresh_rad=(args.abort if args.abort is not None
                                         else (0.40 if args.target == "sim" else None)))
    src_kind, side, auto, rc = robot.grasp_source_kind, args.side, args.no_confirm, 0
    seg_mode = (robot.cfg["grasp"].get("segment", {}) or {}).get("mode")

    class _Obs(g1.GraspObserver):
        def on_world_built(self, pts):                    # ESDF overlay on the grasp scene
            viz = getattr(robot.grasp_source, "viz", None)
            if viz is None or pts is None or not (args.visualize and args.collision_world):
                return
            q = robot.arm.get_current_dual_arm_q()
            viz.show_collision_world(
                pts, voxel_size=robot.planner._cw_params()["esdf_voxel_size"],
                wrists={s: robot.planner.fk(s, q).translation for s in (g1.LEFT, g1.RIGHT)})

        def on_selected(self, chosen, report):            # mark + report cuRobo's pick
            viz = getattr(robot.grasp_source, "viz", None)
            if viz is not None and chosen.grasp_pose is not None:
                viz.mark_chosen(chosen.grasp_pose.homogeneous)
            print(f"chosen: confidence {chosen.confidence:.3f} (goalset idx "
                  f"{report.chosen_index}) | grasp wrist "
                  f"{np.round(chosen.wrist_goal.translation, 3)} m")
            if src_kind == "sim_cloud":  # GT cube -> FK-verify our fingers land on the object
                try:
                    ps = getattr(robot.grasp_source, "pose_source", None)
                    gt = ps.block_pose(args.object) if ps is not None else None
                    if gt is not None:
                        _finger_contact_check(robot, side, chosen.wrist_goal, gt.translation)
                except Exception as e:   # noqa: BLE001 - diagnostic only, never blocks the grasp
                    print(f"FK check skipped: {e}")

        def on_phase(self, label, ok, info):
            print(f"{label:8s}: {'ok' if ok else 'FAILED'} ({info})")

    cands = []                                    # so the --diagnose handler can reference it
    try:
        console.confirm("home (planned)", auto)
        r = robot.home()
        print("home  :", r)
        if not r.ok:
            raise RuntimeError(f"home failed: {r.info}")
        console.confirm("open hand", auto)
        print("open  :", robot.open_hand(side))

        console.confirm(f"generate grasps via '{src_kind}' (segment={seg_mode}) -- "
                        f"object in view", auto)
        cands = robot.grasp_candidates(side, args.object)   # warms the camera streams itself
        n_obb = sum(1 for c in cands if c.extra.get("branch_tag") == "obb")
        n_down = sum(1 for c in cands if c.grasp_pose is not None
                     and c.grasp_pose.rotation[2, 2] < -0.7)
        print(f"grasps: {src_kind} -> {len(cands)} candidate(s) for '{args.object}' | "
              f"obb={n_obb} diff={len(cands) - n_obb} | top-down(approach~-Z)={n_down}")
        if not cands:
            raise RuntimeError(f"grasp source '{src_kind}' produced no candidates "
                               f"(no detection / no depth / no mask / no grasps)")

        opts = g1.GraspOptions(
            close_fraction=args.close_frac, verify_close=args.verify,
            approach=not args.grasp_only, lift=not args.grasp_only,
            grasp_z_offset=args.grasp_z,
            max_candidates=(1 if args.select == "first" else None),
            confirm=lambda lbl: console.confirm(
                f"CLOSE hand to {args.close_frac:.2f}" if lbl == "close"
                else f"move to {lbl.upper()}", auto),
            observer=_Obs())
        res = P.grasp_motion(robot, side, cands, opts)
        print(f"grasp_motion: {res.info} | phases: {res.report.phases if res.report else '-'}")
        if not res.ok:
            raise RuntimeError(res.info)

        console.confirm("home (return)", auto)
        print("home  :", robot.home())
        print("DONE")
    except KeyboardInterrupt as e:
        print(f"\nABORTED by operator ({e}) -- holding position; no recovery motion.")
        rc = 1
    except Exception as e:                       # noqa: BLE001 - top-level task recovery
        print(f"\nPICK ABORTED ({type(e).__name__}): {e}")
        if args.diagnose:                        # explain the failure config BEFORE recovery moves it
            from g1_primitives.motion.diagnostics import explain_failure
            explain_failure(robot.planner, side, robot.arm.get_current_dual_arm_q(),
                            candidates=cands, grasp_cfg=robot.cfg["planner"].get("grasp"),
                            viz=getattr(robot.grasp_source, "viz", None),
                            k=args.diagnose_k, collision_world=args.collision_world)
        print("recovering -> open hand + home ...")
        try:
            robot.open_hand(side)
            robot.home()
        except Exception as e2:                  # noqa: BLE001 - best-effort recovery
            print(f"  recovery best-effort failed: {e2}")
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
