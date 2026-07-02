#!/usr/bin/env python
"""09 - Pick-and-lift driven by a GraspGenX GraspSource (live depth on real, or the sim_cloud
ground-truth de-risk path in sim), with optional SAM3 segmentation of the target object.

The AprilTag A-B grasp baseline lives in 07_pick_place (and stays a GraspSource in the codebase);
09 is GraspGenX-only. Same shape as 07_pick_place, but the grasp comes from `robot.grasp_source`,
planned with cuRobo's native plan_grasp (goalset selection + approach/grasp/lift segments):

    home -> open -> grasp_source.grasps(object) -> plan_grasp(goalset of K candidates)
         -> approach -> grasp -> close -> lift -> home   (cuRobo picks the feasible grasp)

  --source graspgenx   ranked 6-DoF grasps from the GraspGenX ZMQ service (needs the server
                       up + the real depth stream; sim has no depth -> fails loudly)
  --source sim_cloud   sim de-risk: GT cube cloud from rt/sim_state -> GraspGenX (no camera/SAM3)
  --segment interactive  refine a SAM3 mask (text/box/points) in a cv2 GUI before the cloud
            auto         one SAM3 call with grasp.yaml default_prompt
            none         no segmentation (whole frame)   [overrides grasp.yaml segment.mode]

By default the K ranked candidates are fed to cuRobo's plan_grasp as ONE goalset; cuRobo returns
the feasible grasp (goalset_index) and plans approach->grasp->lift as native segments, sweeping the
configured (approach, lift) offsets (planner.yaml: grasp.strategies). `--select`:
  reachable  feed ALL candidates -- cuRobo globally picks the feasible one
  first      feed ONLY the #1 (highest-confidence) grasp (the one viser draws the mesh on) -- no
             selection. NB: "first" = top-CONFIDENCE, unrelated to the server's top-DOWN planner
`--legacy` restores the old path (sequential plan_to_pose probe + manual approach/descend/lift
offsets) as an A-B baseline. The approach backs off along the grasp's own APPROACH axis (grasp +Z,
read from the grasp pose in the PELVIS frame -- so legacy is independent of the wrist transform,
unlike the native path which offsets in the wrist/tool frame), the lift goes world +Z up. Operator-
gated each step; on any failure it opens
+ homes. On real: gravity comp + time_dilation apply from config; operator sets debug mode via the
remote first. Watch the e-stop.

  real: bash -ic 'use_conda g1_curobo && python scripts/09_graspgen.py --target real --source graspgenx --segment interactive'
"""
import argparse
import os
import sys
import time

import numpy as np
import _rig
from g1_classical_manip import primitives as P
from g1_classical_manip.ee.hand_base import LEFT, RIGHT
from g1_classical_manip.motion.curobo_planner import PlanningError
from g1_classical_manip.latency import LOG


def _shift_z(pose, dz: float):
    """Copy a wrist goal Pose shifted by `dz` along world +z (pelvis frame)."""
    p = pose.copy()
    p.translation = pose.translation + np.array([0.0, 0.0, dz])
    return p


def _approach_pose(grasp_wrist, grasp_pose, dist: float):
    """Pre-grasp: back off `dist` along the grasp APPROACH axis (grasp +Z = into the object)
    when the 6-DoF grasp frame is known (always, for GraspGenX/sim_cloud), else straight up
    (world +z; defensive fallback). So the descend sweeps along the gripper's approach line."""
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
    ap.add_argument("--source", choices=["graspgenx", "sim_cloud"], default="graspgenx",
                    help="grasp source: graspgenx (live depth, real) | sim_cloud (sim GT de-risk). "
                         "Defaults to graspgenx; pass sim_cloud for the in-sim path. AprilTag is the "
                         "07_pick_place baseline -- not selectable here (09 is GraspGenX-only).")
    ap.add_argument("--segment", choices=["auto", "interactive", "none"], default=None,
                    help="SAM3 segmentation mode (overrides grasp.yaml segment.mode)")
    ap.add_argument("--visualize", action="store_true",
                    help="show cloud + ranked grasps in a viser GUI (graspgenx source)")
    ap.add_argument("--select", choices=["reachable", "first"], default="reachable",
                    help="reachable = feed ALL candidates to plan_grasp (cuRobo picks the feasible); "
                         "first = feed ONLY the #1 highest-confidence grasp (the viser mesh-overlay "
                         "best). 'first' is top-CONFIDENCE, NOT the server's top-DOWN planner")
    ap.add_argument("--legacy", action="store_true",
                    help="old path: sequential plan_to_pose probe + manual approach/descend/lift "
                         "(A-B baseline vs the native plan_grasp goalset solve)")
    ap.add_argument("--grasp-only", action="store_true",
                    help="skip the approach back-off AND the lift: plan ONE free-space move "
                         "straight to the grasp pose, then close. Isolates the grasp frame "
                         "(chosen.wrist_goal) for a sanity check (native path)")
    ap.add_argument("--collision-world", action="store_true",
                    help="build a depth-ESDF collision world (head depth -> cuRobo Mapper) so the "
                         "approach routes AROUND the object/table (planner.yaml grasp.collision_world). "
                         "Needs a head depth stream (08_check_depth --target <t> green)")
    ap.add_argument("--side", choices=[LEFT, RIGHT], default=RIGHT)
    ap.add_argument("--object", default="block")
    ap.add_argument("--approach", type=float, default=0.10,
                    help="pre-grasp back-off along the grasp approach axis (m)")
    ap.add_argument("--grasp-z", type=float, default=0.0,
                    help="vertical offset added to the grasp pose (m, +z)")
    # Tool-frame calibration (vestigial now -- the transform is baked into grasp.yaml; kept for
    # re-tuning). Post-rotate each grasp wrist goal in the WRIST (tool) frame. Under the derived
    # [pi/2,0,pi] map: wrist +X ~ closing axis, wrist +Y = approach axis, wrist +Z = spread axis.
    # roll = about wrist +X (closing); pitch = about wrist +Y (approach); yaw = about wrist +Z
    # (spread). Sweep to overlay the Dex3 on the GraspGenX gripper mesh, then bake the winner into
    # grasp.yaml: wristyaw_grasp_rpy.
    ap.add_argument("--grasp-roll-deg", type=float, default=0.0,
                    help="rotate each grasp wrist goal about wrist +X (closing axis) by N deg")
    ap.add_argument("--grasp-pitch-deg", type=float, default=0.0,
                    help="rotate each grasp wrist goal about wrist +Y (approach axis) by N deg")
    ap.add_argument("--grasp-yaw-deg", type=float, default=0.0,
                    help="rotate each grasp wrist goal about wrist +Z by N deg")
    ap.add_argument("--lift", type=float, default=0.10, help="lift height after grasp (m, +z)")
    ap.add_argument("--verify", action="store_true", help="verify the grasp on close")
    ap.add_argument("--abort", type=float, default=None,
                    help="override executor tracking-error abort threshold (rad)")
    ap.add_argument("--speed", type=float, default=None,
                    help="trajectory playback time-dilation (<1 = slower)")
    ap.add_argument("--no-confirm", action="store_true", help="skip the per-step prompt")
    ap.add_argument("--close-frac", type=float, default=0.65, help="hand close fraction")
    ap.add_argument("--latency", action="store_true",
                    help="record per-component latency (depth/SAM3/GraspGenX/collision-world/"
                         "plan_grasp/exec/hand) and, at the end, print a table + save a "
                         "timeline+bar graph. Isolates real compute from the human GUI/keyboard gates.")
    ap.add_argument("--latency-out", default="latency.png",
                    help="path for the latency graph PNG (a sibling .json of raw spans is also "
                         "written). Default ./latency.png")
    ap.add_argument("--show-spheres", action="store_true",
                    help="overlay the cuRobo collision spheres (green) at the start/home config on "
                         "the grasp viser scene (needs --visualize) -- to SEE why a 'Start or End "
                         "state in collision' fires: a sphere inside the red ESDF voxels = world "
                         "collision; two spheres overlapping = self-collision")
    ap.add_argument("--diagnose", action="store_true",
                    help="on a plan/exec failure, print WHY the config is rejected -- self-collision "
                         "link pairs, ESDF-penetrating spheres, joint-limit margins, and wrist "
                         "singularity (manipulability) -- and (with --visualize) overlay the "
                         "offending spheres in magenta. The tool to decide if finer spheres would help.")
    ap.add_argument("--debug-planner", action="store_true",
                    help="turn up cuRobo's own logger so plan_grasp prints its internal failure "
                         "reasons (graph-planner 'Start or End state in collision', IK 'No grasp in "
                         "goal set was reachable', per-stage trajopt warnings) instead of swallowing them")
    ap.add_argument("--diagnose-k", type=int, default=0,
                    help="how many top candidates --diagnose sweeps for grasp/pre-grasp reachability "
                         "on a failure (0 = ALL; each is ~2 world-free plan solves ~0.8s, so a big "
                         "goalset takes a minute+)")
    args = ap.parse_args()
    if args.latency:
        LOG.enable()

    robot = _rig.connect(args.target, connect_hand=True, connect_camera=True,
                         camera_config=_rig.camera_config_for(args.target))
    if args.debug_planner:
        from g1_classical_manip.motion.curobo_planner import set_curobo_log_level
        set_curobo_log_level("debug")
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
    src_kind = robot.cfg["grasp"].get("grasp_source", "graspgenx")   # --source forces this; never apriltag
    seg_mode = (robot.cfg["grasp"].get("segment", {}) or {}).get("mode")
    if args.abort is not None:
        robot.executor.abort_thresh = args.abort
    elif args.target == "sim":
        robot.executor.abort_thresh = 0.40
    if args.speed is not None:
        robot.executor.time_dilation = args.speed
    if args.grasp_only:        # frame sanity check: ONE move straight to the grasp pose, no
        robot.cfg["planner"].setdefault("grasp", {})["strategies"] = [   # back-off, no lift
            {"approach_offset": 0.0, "plan_approach": False, "plan_lift": False}]
    if args.collision_world:   # depth-ESDF world: approach routes around the object/table
        cw = robot.cfg["planner"].setdefault("grasp", {}).setdefault("collision_world", {})
        cw["enabled"] = True
        robot.planner.set_collision_world(True, cw)     # rebuilds the grasp planner voxel-capable
        print("collision world: ON (head depth -> cuRobo ESDF; approach is obstacle-aware)")

    side, auto, rc = args.side, args.no_confirm, 0
    cands = []                                   # so the --diagnose except handler can reference it
    try:
        _rig.confirm("home (planned)", auto)
        with LOG.span("home:start", LOG.EXEC):
            r = P.home(robot)
        print("home  :", r)
        if not r.ok:
            raise RuntimeError(f"home failed: {r.info}")

        _rig.confirm("open hand", auto)
        with LOG.span("hand:open", LOG.EXEC):
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
        # Top-down verification (GraspGenX protocol v2): obb/diff branch split + how many grasps
        # actually approach from above (approach axis = grasp +Z = grasp_pose.rotation[:,2];
        # down = pelvis -Z, so rotation[2,2] < -0.7 is within ~45deg of straight down).
        n_obb = sum(1 for c in cands if c.extra.get("branch_tag") == "obb")
        n_down = sum(1 for c in cands if c.grasp_pose is not None
                     and c.grasp_pose.rotation[2, 2] < -0.7)
        print(f"grasps: {src_kind} -> {len(cands)} candidate(s) for '{args.object}' | "
              f"obb={n_obb} diff={len(cands) - n_obb} | top-down(approach≈-Z)={n_down}")
        if not cands:
            raise RuntimeError(f"grasp source '{src_kind}' produced no candidates "
                               f"(no detection / no depth / no mask / no grasps)")
        if args.show_spheres and args.visualize:   # cuRobo collision spheres @ the start/home config
            viz = getattr(robot.grasp_source, "viz", None)
            if viz is not None:
                c, r = robot.planner.collision_spheres(robot.arm.get_current_dual_arm_q())
                viz.show_collision_spheres(c, r)
        if args.grasp_roll_deg or args.grasp_pitch_deg or args.grasp_yaw_deg:
            from dataclasses import replace                    # post-rotate each wrist goal in
            from g1_classical_manip.spatial.pose import Pose, rpy_to_matrix   # the WRIST frame:
            rot = Pose(rpy_to_matrix(np.deg2rad(args.grasp_roll_deg),         # roll/pitch/yaw =
                                     np.deg2rad(args.grasp_pitch_deg),        # about wrist X/Y/Z
                                     np.deg2rad(args.grasp_yaw_deg)), [0, 0, 0])
            cands = [replace(c, wrist_goal=c.wrist_goal * rot) for c in cands]
            print(f"applied grasp rotation rpy(deg)=[{args.grasp_roll_deg:+.0f}, "
                  f"{args.grasp_pitch_deg:+.0f}, {args.grasp_yaw_deg:+.0f}] in the wrist frame "
                  f"(sweep to match the Dex3 to the GraspGenX gripper mesh)")
        def _report_choice(chosen, grasp_wrist, note=""):
            """Mark the chosen grasp in viser, print it, and (sim_cloud) FK-check our fingertips
            against the GT cube. Shared by the native + legacy paths."""
            viz = getattr(robot.grasp_source, "viz", None)   # green = the grasp we chose
            if viz is not None and chosen.grasp_pose is not None:
                viz.mark_chosen(chosen.grasp_pose.homogeneous)
            print(f"chosen: confidence {chosen.confidence:.3f}{note} | grasp wrist "
                  f"{np.round(grasp_wrist.translation, 3)} m")
            if src_kind == "sim_cloud":  # GT cube -> FK-verify our fingers land on the object
                try:
                    ps = getattr(robot.grasp_source, "pose_source", None)
                    gt = ps.block_pose(args.object) if ps is not None else None
                    if gt is not None:
                        _finger_contact_check(robot, side, grasp_wrist, gt.translation)
                except Exception as e:   # noqa: BLE001 - diagnostic only, never blocks the grasp
                    print(f"FK check skipped: {e}")

        def _do_close():
            _rig.confirm(f"CLOSE hand to {args.close_frac:.2f}", auto)
            with LOG.span("hand:close", LOG.EXEC):
                print("close :", P.close_hand(robot, side, verify=args.verify, fraction=args.close_frac))

        def _show_world(points):
            """Overlay the depth-ESDF collision world on the SAME viser scene as the grasps (red
            voxels + blue wrist@q markers), so the operator sees the obstacles the approach routes
            around -- and that the robot self-filter took the arm out. Fired after the world is
            built (before planning), so it shows even if planning then fails."""
            viz = getattr(robot.grasp_source, "viz", None)
            if viz is None or points is None:
                return
            q = robot.arm.get_current_dual_arm_q()
            wrists = {s: robot.planner.fk(s, q).translation for s in (LEFT, RIGHT)}
            viz.show_collision_world(points, voxel_size=robot.planner._cw_params()["esdf_voxel_size"],
                                     wrists=wrists)

        if args.legacy:                  # --- old sequential probe + manual offsets (A-B baseline) ---
            if args.select == "first":
                chosen = cands[0]                        # confidence-sorted -> [0] = the viser best
                grasp = _shift_z(chosen.wrist_goal, args.grasp_z)
                try:                                      # probe but DO NOT fall back -- just warn
                    robot.planner.plan_to_pose(robot.arm.get_current_dual_arm_q(), side, grasp)
                    reach = "plans OK"
                except PlanningError:
                    reach = "WILL NOT PLAN -- approach/descend will fail"
                print(f"select=first: top grasp confidence {chosen.confidence:.3f} -- {reach}")
            else:
                chosen, grasp = _select_candidate(robot, side, cands, args.grasp_z)
                if chosen is None:
                    raise RuntimeError(f"none of the {len(cands)} candidates plan to a reachable grasp")
            _report_choice(chosen, grasp)
            approach = _approach_pose(grasp, chosen.grasp_pose, args.approach)
            lift = _shift_z(grasp, args.lift)

            _rig.confirm("move to APPROACH", auto)
            if not _rig.do_move(robot, side, approach, "approach").ok:
                raise RuntimeError("approach move failed")
            _rig.confirm("move to GRASP (descend)", auto)
            if not _rig.do_move(robot, side, grasp, "descend").ok:
                raise RuntimeError("descend move failed")
            _do_close()
            _rig.confirm("move to LIFT", auto)
            if not _rig.do_move(robot, side, lift, "lift").ok:
                raise RuntimeError("lift move failed")
        else:                            # --- native cuRobo plan_grasp goalset path ---
            from dataclasses import replace
            feed = cands[:1] if args.select == "first" else cands
            if args.grasp_z:             # optional vertical pre-shift of the grasp goals
                feed = [replace(c, wrist_goal=_shift_z(c.wrist_goal, args.grasp_z)) for c in feed]
            res = P.grasp_motion(
                robot, side, feed, close_cb=_do_close,
                confirm_cb=lambda lbl: _rig.confirm(f"move to {lbl.upper()}", auto),
                on_selected=lambda chosen, out: _report_choice(
                    chosen, chosen.wrist_goal, note=f" (goalset idx {out.chosen_index})"),
                on_world_built=(_show_world if (args.visualize and args.collision_world) else None))
            print(f"grasp_motion: {res.info}")
            if not res.ok:
                raise RuntimeError(res.info)

        _rig.confirm("home (return)", auto)
        with LOG.span("home:return", LOG.EXEC):
            print("home  :", P.home(robot))
        print("DONE")
    except KeyboardInterrupt as e:
        print(f"\nABORTED by operator ({e}) -- holding position; no recovery motion.")
        rc = 1
    except Exception as e:                       # noqa: BLE001 - top-level task recovery
        print(f"\nGRASP-GEN ABORTED ({type(e).__name__}): {e}")
        if args.diagnose:                        # explain the failure config BEFORE recovery moves it
            try:
                q_fail = robot.arm.get_current_dual_arm_q()
                wpts = robot.planner.collision_world_points()
                viz = getattr(robot.grasp_source, "viz", None)
                print("--- diagnose: START config (current) ---")
                out = robot.planner.diagnose(q_fail, side, world_points=wpts)
                if viz is not None and len(out["offending_centers"]):   # magenta = start offenders
                    viz.show_collision_spheres(out["offending_centers"], out["offending_radii"],
                                               color=[255, 0, 255], name="diag_offenders")
                if args.collision_world:         # TRUTH-TEST: the planner's OWN ESDF gate at start --
                    wc = robot.planner.world_check(side, q_fail)   # the real 'Start or End' predicate
                    if viz is not None and len(wc["offending_centers"]):   # red = gate-failing spheres
                        viz.show_collision_spheres(wc["offending_centers"], wc["offending_radii"],
                                                   color=[255, 60, 60], name="diag_world_gate")
                if cands:                        # END: sweep the top-K candidates' grasp + pre-grasp
                    gp = robot.cfg["planner"].get("grasp") or {}
                    strat = (gp.get("strategies") or [{}])[0]
                    dist = abs(float(strat.get("approach_offset", -0.10)))
                    grasp_goals = [c.wrist_goal for c in cands]
                    pregrasp_goals = [_approach_pose(c.wrist_goal, c.grasp_pose, dist) for c in cands]
                    print(f"--- diagnose: END configs (top candidates, pre-grasp back-off {dist:.2f}m) ---")
                    robot.planner.diagnose_candidates(
                        side, grasp_goals, pregrasp_goals, world_points=wpts,
                        k=args.diagnose_k, start_q_repo14=q_fail)
            except Exception as de:              # noqa: BLE001 - diagnostics must never mask the abort
                print(f"diagnose failed: {de}")
        print("recovering -> open hand + home ...")
        try:
            P.open_hand(robot, side)
            P.home(robot)
        except Exception as e2:                  # noqa: BLE001 - best-effort recovery
            print(f"  recovery best-effort failed: {e2}")
        rc = 1
    if args.latency:                             # emit even on abort -- partial timings are useful
        print(LOG.summary())
        try:
            png = LOG.plot(args.latency_out, title="09_graspgen latency")
            jsn = LOG.dump_json(os.path.splitext(args.latency_out)[0] + ".json")
            if png:
                print(f"latency graph -> {os.path.abspath(png)}")
            print(f"latency spans -> {os.path.abspath(jsn)}")
        except Exception as e:                   # noqa: BLE001 - reporting must never crash the run
            print(f"latency graph failed: {e}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
