#!/usr/bin/env python
"""tools/validate_perception - SIM ground-truth gate for the FULL perception stack
(NO arm motion).

Runs the real pipeline -- head depth -> SAM3 mask -> deproject -> GraspGenX -- against
the Isaac sim, then scores what it perceived against the rt/sim_state ground truth:

  cloud     perceived object cloud vs an analytic GT cube at the live block pose
            (centroid mm, chamfer mean/p95, inlier fraction, AABB delta)
  mask IoU  the SAM3 mask vs the GT cube projected into the image -- separates a
            SEGMENTATION failure from an EXTRINSICS failure (good IoU + bad cloud
            error points at T_pelvis_camera)
  grasps    where GraspGenX put the object (grasp origin + fingertip depth * approach)
            vs the GT center, and an FK check of OUR fingertips at the top candidate

Exit codes make it a regression gate: 0 = pass, 1 = a --max-* threshold failed,
2 = setup failure (no GT / no frames / empty cloud). Needs the sim up (depth PUB
:55556 + rt/sim_state) and the SAM3 (:5557) + GraspGenX (:5556) servers unless
--segment none / --no-grasps.

  bash -ic 'use_conda g1_curobo && python scripts/tools/validate_perception.py --segment auto --json'
"""
import argparse
import json
import sys
import time

import numpy as np

import g1_primitives as g1
from g1_primitives import PointCloud
from g1_primitives.api import console
from g1_primitives.grasp.sim_cloud_source import sample_cube
from g1_primitives.perception import validation as V


def _gt_cloud(robot, gt_pose, edge_m, n_points):
    """Analytic GT cube cloud at the live block pose, culled to the faces the head
    camera can see (matches the partial view the depth pipeline perceives)."""
    pts, normals = sample_cube(edge_m, n_points)
    R, t = gt_pose.rotation, gt_pose.translation
    pts_pelvis = pts @ R.T + t
    cam = np.asarray(robot.frames.T_pelvis_camera(None).translation, float)
    facing = np.einsum("ij,ij->i", normals @ R.T, cam[None, :] - pts_pelvis) > 0.0
    if facing.any():
        pts_pelvis = pts_pelvis[facing]
    return PointCloud(pts_pelvis.astype(np.float32), frame="pelvis")


def _run_once(robot, args, edge_m):
    gt = robot.detect(args.object)
    if gt is None:
        print(f"no GT pose for '{args.object}' from rt/sim_state -- is the sim up?")
        return None, 2
    gt_cloud = _gt_cloud(robot, gt.pose, edge_m, args.n_points)

    cands = []
    try:
        cands = robot.grasp_candidates(args.side, args.object)
    except Exception as e:                     # noqa: BLE001 - GraspGenX down is OK for --no-grasps
        if not args.no_grasps:
            print(f"grasp source failed: {type(e).__name__}: {e}")
            return None, 2
        print(f"(grasp source failed after the cloud was built -- fine for --no-grasps: {e})")
    snap = robot.grasp_source.last_snapshot
    if snap is None or snap.cloud is None or snap.cloud.is_empty():
        print("no perceived cloud (no depth frame / empty mask) -- "
              "check checks/03_depth + tools/segment first.")
        return None, 2

    out = {"object": args.object, "gt_center": np.round(gt.pose.translation, 4).tolist()}
    ce = V.cloud_error(snap.cloud, gt_cloud)
    out["cloud"] = ce.as_dict()

    if snap.mask is not None:                  # segmentation vs extrinsics separation
        gt_mask = V.project_mask(gt_cloud, robot.cfg["camera"]["intrinsics"],
                                 robot.frames.T_pelvis_camera(None),
                                 snap.mask.shape, dilate_px=2)
        out["mask_iou"] = round(V.mask_iou(np.asarray(snap.mask, bool), gt_mask), 3)

    if cands and not args.no_grasps:
        out["grasps"] = V.candidate_set_error(cands, gt.pose.translation)
        q7 = robot.cfg["hands"]["dex3"]["presets"]["power_close"][args.side]
        out["fingertips_top1"] = V.fingertip_contact_error(
            args.side, cands[0].wrist_goal, gt.pose.translation, q7)

    if args.visualize:
        _show(robot, gt_cloud, snap, args)
    return out, 0


def _show(robot, gt_cloud, snap, args):
    """GT (green) + perceived (its own color / gray) in one viser scene."""
    try:
        import viser
    except Exception as e:                     # noqa: BLE001
        print(f"viser unavailable: {e}")
        return
    if not hasattr(_show, "_srv"):
        _show._srv = viser.ViserServer(host="127.0.0.1", port=args.port)
    srv = _show._srv
    p = snap.cloud.points
    colors = snap.cloud.colors if snap.cloud.colors is not None else \
        np.tile([150, 150, 150], (len(p), 1)).astype(np.uint8)
    srv.scene.add_point_cloud("/perceived", points=p.astype(np.float32),
                              colors=colors, point_size=0.003)
    g = gt_cloud.points
    srv.scene.add_point_cloud("/ground_truth", points=g.astype(np.float32),
                              colors=np.tile([0, 220, 0], (len(g), 1)).astype(np.uint8),
                              point_size=0.003)
    print(f"viser: http://localhost:{args.port}  (GREEN = GT cube, colored/gray = perceived)")


def main():
    ap = console.add_target_arg(argparse.ArgumentParser())
    ap.add_argument("--object", default="block")
    ap.add_argument("--side", choices=[g1.LEFT, g1.RIGHT], default=g1.RIGHT)
    ap.add_argument("--segment", choices=["auto", "interactive", "none"], default="auto",
                    help="SAM3 mode for the perceived pipeline (auto = headless default)")
    ap.add_argument("--no-grasps", action="store_true",
                    help="cloud-only: skip the GraspGenX metrics (server may be down; the "
                         "perceived cloud is captured before the infer call)")
    ap.add_argument("--n-points", type=int, default=2000, help="GT cube sample points")
    ap.add_argument("--edge", type=float, default=None,
                    help="GT cube edge (m); default grasp.yaml sim_cloud.object_size_m")
    ap.add_argument("--max-centroid-mm", type=float, default=None,
                    help="FAIL (exit 1) if the cloud centroid error exceeds this")
    ap.add_argument("--max-chamfer-mm", type=float, default=None,
                    help="FAIL (exit 1) if the mean chamfer distance exceeds this")
    ap.add_argument("--visualize", action="store_true", help="GT vs perceived in viser")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--loop", action="store_true", help="Enter re-runs on a fresh frame")
    ap.add_argument("--json", action="store_true", help="print metrics as one JSON object")
    args = ap.parse_args()
    if args.target != "sim":
        print("validate_perception is SIM-only (the GT side is rt/sim_state).")
        return 2

    # Full perception stack under test: DDS (GT pose) + camera, NO homing/motion.
    robot = g1.connect("sim", home_on_connect=False)
    robot.set_grasp_source("graspgenx")                     # the real pipeline, in sim
    robot.set_segmenter("none" if args.segment == "none" else args.segment)
    edge_m = args.edge if args.edge is not None else float(
        (robot.cfg["grasp"].get("sim_cloud") or {}).get("object_size_m", 0.06))

    rc = 0
    while True:
        out, rc = _run_once(robot, args, edge_m)
        if out is not None:
            if args.json:
                print(json.dumps(out))
            else:
                ce = out["cloud"]
                print(f"cloud : centroid {ce['centroid_mm']:.1f} mm {ce['centroid_delta_mm']} | "
                      f"chamfer mean {ce['chamfer_mean_mm']:.1f} / p95 {ce['chamfer_p95_mm']:.1f} mm | "
                      f"inliers {100 * ce['inlier_frac']:.0f}% | bbox d {ce['bbox_delta_mm']} mm "
                      f"({ce['n_perceived']} vs GT {ce['n_gt']} pts)")
                if "mask_iou" in out:
                    print(f"mask  : IoU vs projected GT = {out['mask_iou']:.3f}")
                if "grasps" in out:
                    gr = out["grasps"]
                    print(f"grasps: {gr['n']} cands | implied-object err top1 {gr['top1_mm']} mm, "
                          f"best {gr['best_mm']} mm, mean {gr['mean_mm']} mm")
                if "fingertips_top1" in out:
                    ft = out["fingertips_top1"]
                    print(f"FK    : top-1 contact {ft['contact_mm']} mm from GT center "
                          f"(thumb {ft['thumb_mm']}, fingers {ft['fingers_mm']})")
            # threshold gates -> the sim regression exit code
            ce = out["cloud"]
            if args.max_centroid_mm is not None and ce["centroid_mm"] > args.max_centroid_mm:
                print(f"GATE FAIL: centroid {ce['centroid_mm']:.1f} > {args.max_centroid_mm} mm")
                rc = 1
            if args.max_chamfer_mm is not None and ce["chamfer_mean_mm"] > args.max_chamfer_mm:
                print(f"GATE FAIL: chamfer {ce['chamfer_mean_mm']:.1f} > {args.max_chamfer_mm} mm")
                rc = 1
            if rc == 0:
                print("PASS" if (args.max_centroid_mm or args.max_chamfer_mm) else "done (no gates set)")
        if not args.loop:
            break
        try:
            if input("  >> Enter to re-run on a fresh frame, 'q' to quit: ").strip().lower() in (
                    "q", "quit"):
                break
        except (EOFError, KeyboardInterrupt):
            break
        time.sleep(0.1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
