#!/usr/bin/env python
"""12 - Inspect the depth-ESDF collision world IN ISOLATION (no planning, no motion).

Builds the cuRobo Mapper world from ONE head depth frame and reports/visualises it, so we can
answer the two questions that decide whether plan_grasp can use it:

  1. Is the geometry placed correctly in the pelvis frame? -> the ESDF occupied voxels should
     overlap the raw deproject cloud (our known-good convention). If they're rotated/offset, the
     Mapper camera-pose convention is wrong.
  2. Is anything in the world that should NOT be (the ROBOT'S OWN ARM)? The head camera looking
     at the table also sees the forearms/hands; if those get fused, the grasp planner starts the
     arm inside a baked-in copy of itself -> "Goalset planning returned None". The wrist-FK probe
     + the viser overlay reveal this.

Three sources for the depth frame + the robot pose:
  * LIVE (default): connects DDS read-only via _rig.connect (sim -> domain 1/lo, home_on_connect=
    False -> NOTHING moves) so the self-filter runs at the LIVE measured arm+finger q -- exactly
    like the real grasp path (primitives._update_collision_world passes the current q). The filter
    masks the robot at the config IN the depth frame, so it MUST match where the arm actually is;
    if no live state arrives it falls back to home.
  * --no-dds: camera/ZMQ depth only (no DDS); self-filter at home.
  * --frame <capture.npz>: OFFLINE replay of an 11_capture_frame capture -- NO robot/camera/DDS at
    all (planner-only on the GPU). Uses the frame's depth + intrinsics + T_pelvis_camera, and the
    arm/hand q recorded at capture time if present (older captures lack them -> self-filter at home).
World params come from planner.yaml grasp.collision_world (edit there to tune grid_center/extent/voxel).

  sim:   bash -ic 'use_conda g1_curobo && python scripts/12_check_world.py --target sim --visualize'
         (DDS uses domain 1/lo directly like 01_check_dds -- no CYCLONEDDS_URI needed. Depth over
          ZMQ :55556; the sim must be up for both the depth stream AND the live arm q.)
  demo:  bash -ic 'use_conda g1_curobo && python scripts/12_check_world.py --frame captures/scene1.npz --visualize'
         (no robot -- inspect a captured world offline; the 11 -> 12 demo chain)
"""
import argparse
import sys
import time

import numpy as np
import _rig
from g1_classical_manip.factory import make_robot
from g1_classical_manip.ee.hand_base import LEFT, RIGHT
from g1_classical_manip.motion.planner_base import DOF
from g1_classical_manip.motion.collision_world import EsdfMapper
from g1_classical_manip.perception.depth import deproject_depth


def _aabb(p):
    return (np.round(p.min(0), 3).tolist(), np.round(p.max(0), 3).tolist()) if len(p) else (None, None)


def _load_capture(path):
    """An 11_capture_frame .npz -> (depth, K{fx,fy,cx,cy}, T_pelvis_camera Pose, q14|None,
    hand_q{LEFT,RIGHT}|None). q14 / hand_q are present only if the capture recorded them (older
    captures predate that -> None -> the self-filter falls back to home)."""
    from g1_classical_manip.spatial.pose import Pose
    z = np.load(path)
    depth = np.asarray(z["depth"], np.float32)
    K = {k: float(z[k]) for k in ("fx", "fy", "cx", "cy")}
    T_pc = Pose.from_homogeneous(np.asarray(z["T_pelvis_camera"], float))
    q14 = np.asarray(z["q14"], float) if "q14" in z and np.size(z["q14"]) == DOF else None
    hand_q = None
    if "hand_q_left" in z and "hand_q_right" in z and np.size(z["hand_q_left"]) == 7:
        hand_q = {LEFT: np.asarray(z["hand_q_left"], float),
                  RIGHT: np.asarray(z["hand_q_right"], float)}
    return depth, K, T_pc, q14, hand_q


def _live_q(robot):
    """Live arm q (wait briefly for lowstate) + finger q for the self-filter. (None, None) if absent."""
    arm = getattr(robot, "arm", None)
    q14 = None
    if arm is not None:
        for _ in range(20):
            q = arm.get_current_dual_arm_q()
            if np.any(q):                                # nonzero -> real state arrived
                q14 = q
                break
            time.sleep(0.05)
    hand = getattr(robot, "hand", None)
    hand_q = {s: hand.get_q(s) for s in (LEFT, RIGHT)} if hasattr(hand, "get_q") else None
    return q14, hand_q


def main():
    ap = _rig.add_target_arg(argparse.ArgumentParser())
    ap.add_argument("--visualize", action="store_true", help="viser overlay (occupied vs cloud vs wrists)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-self-filter", action="store_true",
                    help="skip the robot self-filter -> show the raw world WITH the arm baked in")
    ap.add_argument("--probe", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="report ESDF + raw-cloud coverage near a pelvis-frame point (e.g. the cube "
                         "[0.3 -0.1 0.1]) -> is the object actually in the world, or was it erased?")
    ap.add_argument("--margin", type=float, default=None,
                    help="override robot_mask_margin (m) for the self-filter -- sweep to find the "
                         "value that removes the arm WITHOUT erasing a nearby object")
    ap.add_argument("--esdf-voxel", type=float, default=None,
                    help="override esdf_voxel_size (m) -- finer = a crisper object (tsdf set to half)")
    ap.add_argument("--no-dds", action="store_true",
                    help="don't connect DDS (offline: camera/ZMQ depth only); self-filter at home "
                         "instead of the live measured arm q")
    ap.add_argument("--frame", default=None,
                    help="OFFLINE: inspect an 11_capture_frame .npz (no robot/camera/DDS). Uses the "
                         "frame's depth + intrinsics + T_pelvis_camera + recorded q (else home)")
    ap.add_argument("--show-spheres", action="store_true",
                    help="overlay the cuRobo collision spheres (green) at the self-filter config on "
                         "the viser scene (needs --visualize) -- a sphere INSIDE the red ESDF voxels "
                         "= that config collides with the world (start-in-collision); two spheres "
                         "overlapping = self-collision")
    ap.add_argument("--diagnose", action="store_true",
                    help="print WHY the self-filter config is in collision -- self-collision link "
                         "pairs, which spheres penetrate the ESDF (+ mm), joint-limit margins, and "
                         "wrist singularity -- against the built world, and overlay the offending "
                         "spheres in magenta (needs --visualize). Numeric backing for --show-spheres.")
    ap.add_argument("--side", choices=[LEFT, RIGHT], default=RIGHT,
                    help="arm for the --diagnose joint-limit + singularity report (default right); "
                         "self/world-collision are whole-robot and side-independent")
    args = ap.parse_args()

    # --- source the depth + camera pose + robot config: a captured .npz (offline) OR live ---
    q14 = hand_q = None
    if args.frame:                                                   # OFFLINE replay -- no robot at all
        robot = make_robot(connect_dds=False, connect_camera=False,  # planner only (GPU for the ESDF)
                           camera_config=_rig.camera_config_for(args.target))
        depth, K, T_pc, q14, hand_q = _load_capture(args.frame)
        print(f"frame: {args.frame}  depth {depth.shape}  (offline replay, no robot)")
    else:
        if args.no_dds:                                              # camera/ZMQ only, no DDS
            robot = make_robot(connect_dds=False, connect_camera=True,
                               camera_config=_rig.camera_config_for(args.target))
        else:                                                        # live q; home_on_connect=False -> no motion
            robot = _rig.connect(args.target, home_on_connect=False, connect_camera=True,
                                 camera_config=_rig.camera_config_for(args.target))
        cam = robot.camera
        if not cam.has_depth:
            print(f"[{args.target}] no head depth stream -- run 08_check_depth first."); return
        depth = None                                                 # prime the conflate socket
        for _ in range(40):
            depth = cam.get_depth_frame()
            if depth is not None:
                break
            time.sleep(0.05)
        if depth is None:
            print("no depth frame received (is the sim publishing depth on :55556?)"); return
        K = robot.cfg["camera"]["intrinsics"]
        T_pc = robot.frames.T_pelvis_camera(None)                    # head cam optical pose in pelvis
        q14, hand_q = _live_q(robot)                                 # live arm + finger q for the self-filter
    finite = np.isfinite(depth)
    print(f"depth: {depth.shape} finite={100*finite.mean():.0f}% "
          f"range[{np.nanmin(depth[finite])/1000:.2f},{np.nanmax(depth[finite])/1000:.2f}]m")

    p = robot.planner._cw_params()                                  # same params plan_grasp would use
    print(f"world: grid_center={p['grid_center']} extent_m={p['extent_m']} "
          f"esdf_voxel={p['esdf_voxel_size']} depth=[{p['depth_min_m']},{p['depth_max_m']}]m")

    # raw deproject (known-good convention), CROPPED to the workspace box for an apples-to-apples
    # placement check vs the box-bounded ESDF (the full cloud spans the whole frustum, so its raw
    # AABB is meaningless here).
    cloud = deproject_depth(depth, K, T_pc, z_max_m=p["depth_max_m"]).points
    gc, ex = np.array(p["grid_center"]), np.array(p["extent_m"])
    lo, hi = gc - ex / 2, gc + ex / 2
    in_box = np.all((cloud >= lo) & (cloud <= hi), axis=1)
    cloud_box = cloud[in_box]
    print(f"raw cloud : {len(cloud):6d} pts ({len(cloud_box)} in box)  box AABB(pelvis) {_aabb(cloud_box)}")

    # --- build the ESDF world exactly as the planner does (with the robot self-filter) ---
    # The self-filter masks the robot (arm + fingers) at the config IN the depth frame. Use the
    # captured/live q if we have it; else fall back to home (an older capture, or DDS quiet in sim).
    # The arms-only planning model locks the fingers open, so live finger q is what lets the filter
    # mask a bent thumb at its real pose.
    home = np.deg2rad(robot.cfg["robot"]["home_q14_deg"])
    if q14 is not None and np.size(q14) == DOF:
        q_arm = np.asarray(q14, float).reshape(DOF)
        src = "frame (recorded)" if args.frame else "live (measured)"
    else:
        q_arm, src = home, "home (fallback -- no recorded/live arm state)"
    print(f"self-filter arm q [{src}]: {np.round(q_arm, 3)}")
    if hand_q is not None:
        print(f"self-filter hand q: L {np.round(hand_q[LEFT], 2)}  R {np.round(hand_q[RIGHT], 2)}")
    rf = None if args.no_self_filter else \
        robot.planner.robot_depth_filter(q_arm, hand_q=hand_q, margin=args.margin)
    if args.margin is not None:
        print(f"self-filter margin override: {args.margin} m")
    ev = float(args.esdf_voxel) if args.esdf_voxel else p["esdf_voxel_size"]
    tv = ev / 2 if args.esdf_voxel else p["tsdf_voxel_size"]
    if args.esdf_voxel:
        print(f"esdf voxel override: {ev} m (tsdf {tv} m)")
    mapper = EsdfMapper(grid_center=p["grid_center"], extent_m=p["extent_m"],
                        esdf_voxel_size=ev, tsdf_voxel_size=tv,
                        image_hw=depth.shape, depth_min_m=p["depth_min_m"], depth_max_m=p["depth_max_m"])
    t0 = time.time()
    mapper.esdf_from_depth(depth, K, T_pc, robot_filter=rf)         # first call JIT-compiles kernels
    occ = mapper.occupied_points()
    sf = "off" if args.no_self_filter else "on"
    print(f"ESDF      : {len(occ):6d} occupied voxels  AABB(pelvis) {_aabb(occ)}  "
          f"(self_filter={sf}, {time.time()-t0:.1f}s)")
    if len(occ) and len(cloud_box):
        dc = np.linalg.norm(np.array(_aabb(occ)[0]) - np.array(_aabb(cloud_box)[0])) \
            + np.linalg.norm(np.array(_aabb(occ)[1]) - np.array(_aabb(cloud_box)[1]))
        tag = "OK (matches the in-box cloud)" if dc < 0.15 else \
            "MISMATCH -> camera-pose convention may be wrong (or the self-filter removed a lot)"
        print(f"  occupied vs in-box-cloud AABB delta {dc*1000:.0f} mm -> {tag}")

    # --- self-view probe: is the robot's own arm STILL in the world? (should be clear once filtered) ---
    wrists = {}
    for side in (LEFT, RIGHT):
        wp = robot.planner.fk(side, q_arm).translation
        wrists[side] = wp
        if len(occ):
            dists = np.linalg.norm(occ - wp, axis=1)
            near = occ[dists < 0.15]                      # occupied voxels near the wrist
            # the ARM reaches UP to the shoulder (z~0.3); the table/block are low (z<~0.13). So
            # TALL occupied geometry near the wrist = un-removed arm; low = just table/block.
            zhi = float(near[:, 2].max()) if len(near) else -9.9
            arm_remnant = zhi > 0.15
            flag = "  <-- ARM REMNANT (tall geom near wrist)" if arm_remnant else "  (arm clear; near = table/block)"
            print(f"self-view {side:5s}: wrist@q {np.round(wp,3)} nearest occ {dists.min()*1000:.0f} mm, "
                  f"tallest-near z={zhi:.3f}{flag}")

    # --- object probe: is the cube actually in the ESDF (vs erased by the self-filter / occlusion)? ---
    if args.probe is not None:
        pt = np.array(args.probe, float)
        d_occ = float(np.linalg.norm(occ - pt, axis=1).min()) if len(occ) else 9.9
        n_occ = int((np.linalg.norm(occ - pt, axis=1) < 0.05).sum()) if len(occ) else 0
        d_raw = float(np.linalg.norm(cloud - pt, axis=1).min()) if len(cloud) else 9.9
        n_raw = int((np.linalg.norm(cloud - pt, axis=1) < 0.05).sum()) if len(cloud) else 0
        verdict = ("IN THE WORLD" if n_occ > 0 else
                   ("ERASED -- raw cloud has it but the ESDF doesn't (self-filter occlusion/over-mask)"
                    if n_raw > 0 else "NOT SEEN -- absent from BOTH (out of box / occluded / no depth)"))
        print(f"probe {np.round(pt,3)}: ESDF nearest {d_occ*1000:.0f} mm, {n_occ} within 5cm | "
              f"raw cloud nearest {d_raw*1000:.0f} mm, {n_raw} within 5cm -> {verdict}")

    # --- TRUTH-TEST (--diagnose): load the SAME depth into the grasp planner and query ITS checker.
    # This is the exact predicate that fails plan_grasp ('Start or End state in collision'), incl.
    # cuRobo's ESDF semantics -- unlike the geometric occupied-voxel-centre checks above. Any CLI
    # overrides (--esdf-voxel/--margin/--no-self-filter) are folded in so both worlds match.
    wc = None
    if args.diagnose:
        cw = dict(p, enabled=True, esdf_voxel_size=ev, tsdf_voxel_size=tv,
                  self_filter=(not args.no_self_filter))
        if args.margin is not None:
            cw["robot_mask_margin"] = float(args.margin)
        robot.planner.set_collision_world(True, cw)
        if robot.planner.update_grasp_world(args.side, depth, K, T_pc, q_arm, hand_q=hand_q):
            print(f"--- truth-test: the grasp planner's OWN ESDF gate at the self-filter q [{src}] ---")
            wc = robot.planner.world_check(args.side, q_arm)
        else:
            print("truth-test skipped (planner world update failed)")

    if args.visualize:
        try:
            import viser
        except Exception as e:                                       # noqa: BLE001
            print(f"viser unavailable: {e}"); sys.exit(0)
        srv = viser.ViserServer(host="127.0.0.1", port=args.port)
        if len(cloud):
            srv.scene.add_point_cloud("/raw_cloud", points=cloud.astype(np.float32),
                                      colors=np.tile([150, 150, 150], (len(cloud), 1)).astype(np.uint8),
                                      point_size=0.004)
        if len(occ):
            srv.scene.add_point_cloud("/esdf_occupied", points=occ.astype(np.float32),
                                      colors=np.tile([255, 40, 40], (len(occ), 1)).astype(np.uint8),
                                      point_size=float(p["esdf_voxel_size"]) * 0.9)
        for side, wp in wrists.items():
            srv.scene.add_icosphere(f"/wrist_{side}", radius=0.04, color=(40, 40, 255),
                                    position=tuple(float(x) for x in wp))
        if args.probe is not None:
            srv.scene.add_icosphere("/probe", radius=0.03, color=(0, 220, 0),
                                    position=tuple(float(x) for x in args.probe))
        n_sph = 0
        if args.show_spheres:               # the cuRobo collision spheres at the self-filter config
            from g1_classical_manip.viz import viser_primitives as vp
            c, r = robot.planner.collision_spheres(q_arm)
            n_sph = vp.add_collision_spheres(srv, c, r, name="/collision_spheres")
            print(f"collision spheres: {n_sph} overlaid (green) at the self-filter q "
                  f"[{src}] -- any green sphere inside the RED voxels = in collision with the world")
        if args.diagnose:                   # numeric backing: WHY q_arm is (self/world) in collision
            from g1_classical_manip.viz import viser_primitives as vp
            # geometric predicate (legacy, side-by-side with the truth-test above)
            out = robot.planner.diagnose(q_arm, args.side, world_points=(occ if len(occ) else None),
                                         voxel_size=float(p["esdf_voxel_size"]))
            if len(out["offending_centers"]):
                vp.add_collision_spheres(srv, out["offending_centers"], out["offending_radii"],
                                         color=(255, 0, 255), name="/diag_offenders")
                print(f"diagnose: {len(out['offending_centers'])} offending sphere(s) overlaid (magenta)")
            if wc is not None and len(wc["offending_centers"]):   # the REAL gate's offenders (orange)
                vp.add_collision_spheres(srv, wc["offending_centers"], wc["offending_radii"],
                                         color=(255, 140, 0), name="/world_gate_offenders")
                print(f"world_check: {len(wc['offending_centers'])} gate-failing sphere(s) overlaid "
                      f"(orange) -- these are what cuRobo itself rejects")
        print(f"viser: http://localhost:{args.port}  (gray=raw cloud, RED=ESDF occupied, blue=wrist@q"
              + (", GREEN=collision spheres@q" if n_sph else "") + ")")
        print("Ctrl-C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
