#!/usr/bin/env python
"""10 - Standalone GraspGenX grasp inspection in viser (no robot, no motion).

Loads a saved pelvis-frame point cloud, sends it to the GraspGenX ZMQ server, and
shows the cloud + ranked grasps + gripper-mesh overlay in viser -- the client-side
equivalent of GraspGenX's scripts/demo_object_pc.py. Use it to iterate on grasps
(gripper, num_grasps, topk, threshold) without running a pick. Reuses the thin
ZMQ client and the viz module; imports nothing from the graspgenx package.

    bash -ic 'use_conda g1_curobo && python scripts/10_graspgen_viz.py --pcd captures/cloud.npy'

The cloud must already be in the frame you want grasps in (pelvis, meters) -- the
server centers/uncenters internally and returns grasps in that same frame.
"""
import argparse
import os

import numpy as np
import trimesh

from g1_classical_manip.grasp.graspgenx_client import GraspGenXClient
from g1_classical_manip.viz import load_gripper_geom
from g1_classical_manip.viz.grasp_viz import GraspViz

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cloud(path: str) -> np.ndarray:
    """Load an (N,3) cloud from .npy / .npz / .xyz / .ply / .obj."""
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "npy":
        xyz = np.load(path)
    elif ext == "npz":
        d = np.load(path)
        xyz = d[d.files[0]]
    elif ext in ("xyz", "txt"):
        xyz = np.loadtxt(path)
    elif ext in ("ply", "obj", "pcd"):
        loaded = trimesh.load(path)
        xyz = np.asarray(getattr(loaded, "vertices", loaded))
    else:
        raise ValueError(f"unsupported cloud format: .{ext}")
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"expected (N, 3+); got {xyz.shape}")
    return xyz[:, :3]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcd", required=True,
                    help="pelvis-frame (N,3) cloud: .npy/.npz/.xyz/.ply/.obj")
    ap.add_argument("--host", default="127.0.0.1", help="GraspGenX server host")
    ap.add_argument("--port", type=int, default=5556, help="GraspGenX server port")
    ap.add_argument("--gripper_name", default="unitree_g1")
    ap.add_argument("--gripper_asset_dir", default=os.path.join(_REPO_ROOT, "assets", "grippers"),
                    help="<dir>/<gripper_name>/{config.json, *_mesh.obj} (repo-root default)")
    ap.add_argument("--num_grasps", type=int, default=200)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--grasp_threshold", type=float, default=-1.0)
    ap.add_argument("--viser_port", type=int, default=8080)
    ap.add_argument("--no-mesh", action="store_true",
                    help="skip the gripper-mesh overlay (markers only)")
    args = ap.parse_args()

    cloud = load_cloud(args.pcd)
    print(f"loaded {len(cloud)} points from {args.pcd}")

    with GraspGenXClient(host=args.host, port=args.port) as client:
        grasps, conf = client.infer(
            cloud, gripper_name=args.gripper_name, num_grasps=args.num_grasps,
            grasp_threshold=args.grasp_threshold, topk_num_grasps=args.topk)
    print(f"server returned {len(grasps)} grasps")

    asset_dir = args.gripper_asset_dir
    if not os.path.isabs(asset_dir):                # anchor a relative override on the repo root
        asset_dir = os.path.join(_REPO_ROOT, asset_dir)
    geom = load_gripper_geom(asset_dir, args.gripper_name)
    viz = GraspViz(geom, port=args.viser_port, show_mesh=not args.no_mesh,
                   threshold_tuner=True)
    viz.show_candidates(cloud, grasps, conf)
    viz.spin()


if __name__ == "__main__":
    main()
