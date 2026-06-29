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

Camera-only: make_robot(connect_dds=False) -> no DDS, no homing, nothing moves. World params come
from planner.yaml grasp.collision_world (edit there to tune grid_center / extent_m / voxel).

  sim:  bash -ic 'use_conda g1_curobo && python scripts/12_check_world.py --target sim --visualize'
"""
import argparse
import sys
import time

import numpy as np
import _rig
from g1_classical_manip.factory import make_robot
from g1_classical_manip.ee.hand_base import LEFT, RIGHT
from g1_classical_manip.motion.collision_world import EsdfMapper
from g1_classical_manip.perception.depth import deproject_depth


def _aabb(p):
    return (np.round(p.min(0), 3).tolist(), np.round(p.max(0), 3).tolist()) if len(p) else (None, None)


def main():
    ap = _rig.add_target_arg(argparse.ArgumentParser())
    ap.add_argument("--visualize", action="store_true", help="viser overlay (occupied vs cloud vs wrists)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-self-filter", action="store_true",
                    help="skip the robot self-filter -> show the raw world WITH the arm baked in")
    ap.add_argument("--probe", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="report ESDF + raw-cloud coverage near a pelvis-frame point (e.g. the cube "
                         "[0.3 -0.1 0.1]) -> is the object actually in the world, or was it erased?")
    args = ap.parse_args()

    robot = make_robot(connect_dds=False, connect_camera=True,        # camera only, no motion
                       camera_config=_rig.camera_config_for(args.target))
    cam = robot.camera
    if not cam.has_depth:
        print(f"[{args.target}] no head depth stream -- run 08_check_depth first."); return

    depth = None                                                     # prime the conflate socket
    for _ in range(40):
        depth = cam.get_depth_frame()
        if depth is not None:
            break
        time.sleep(0.05)
    if depth is None:
        print("no depth frame received (is the sim publishing depth on :55556?)"); return
    finite = np.isfinite(depth)
    print(f"depth: {depth.shape} finite={100*finite.mean():.0f}% "
          f"range[{np.nanmin(depth[finite])/1000:.2f},{np.nanmax(depth[finite])/1000:.2f}]m")

    K = robot.cfg["camera"]["intrinsics"]
    T_pc = robot.frames.T_pelvis_camera(None)                        # head cam optical pose in pelvis
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
    home = np.deg2rad(robot.cfg["robot"]["home_q14_deg"])           # arm config in the depth (no motion)
    rf = None if args.no_self_filter else robot.planner.robot_depth_filter(home)
    mapper = EsdfMapper(grid_center=p["grid_center"], extent_m=p["extent_m"],
                        esdf_voxel_size=p["esdf_voxel_size"], tsdf_voxel_size=p["tsdf_voxel_size"],
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
        wp = robot.planner.fk(side, home).translation
        wrists[side] = wp
        if len(occ):
            d = float(np.linalg.norm(occ - wp, axis=1).min())
            flag = "  <-- ARM STILL IN THE WORLD" if d < 0.10 else "  (clear)"
            print(f"self-view {side:5s}: wrist@home {np.round(wp,3)} nearest occupied {d*1000:.0f} mm{flag}")

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
        print(f"viser: http://localhost:{args.port}  (gray=raw cloud, RED=ESDF occupied, blue=wrist@home)")
        print("Ctrl-C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
