#!/usr/bin/env python
"""10 - SAM3 segmentation tool (NO motion). Get a head-cam frame (live ZED), a captured
.npz, or a static image, refine a SAM3 mask (interactive cv2 GUI, or one auto call), and
report the masked single-object point cloud. For developing segmentation + cloud offline.

  --frame PATH.npz    replay a captured frame (11_capture_frame): rgb+depth+intrinsics+
                      extrinsics -> full segment -> cloud, NO robot (the offline loop)
  --image PATH        segment a static image -- SAM3 only, NO robot / depth / cloud
  --mode interactive  cv2 GUI: text/box/points, cycle candidates, accept (needs a display)
  --mode auto         one SAM3 call with --text / grasp.yaml default_prompt
  --save              write <base>_cloud.ply (downsampled, COLORED object cloud) + _mask.npy
                      + _overlay.png next to the frame -> feed 10_graspgen_viz --pcd

Live mode needs the real ZED depth stream; all modes need a running SAM3 server
(python -m sam3.serving --port 5557).

  frame: bash -ic 'use_conda g1_curobo && python scripts/10_segment.py --frame captures/scene1.npz --mode interactive'
  image: bash -ic 'use_conda g1_curobo && python scripts/10_segment.py --image ~/repos/sam3/assets/images/groceries.jpg --mode auto --text "a bottle"'
  live : bash -ic 'use_conda g1_curobo && python scripts/10_segment.py --target real --mode interactive'
"""
import argparse
import os
import time

import numpy as np
import _rig
from g1_primitives.factory import make_robot, load_configs
from g1_primitives.spatial.pose import Pose
from g1_primitives.perception.depth import deproject_depth
from g1_primitives.perception.segment import Sam3Segmenter, SegmentationAborted
from g1_primitives.perception.sam3_client import Sam3Client

try:
    import cv2
except Exception:
    cv2 = None


def _show_or_save(bgr, name: str, path: str):
    """Show a BGR image; fall back to writing it when there's no display."""
    if cv2 is None:
        return
    try:
        cv2.imshow(name, bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        cv2.imwrite(path, bgr)
        print(f"no display -> saved {path}")


def _overlay(bgr, mask) -> np.ndarray:
    """Translucent red tint over the masked-in pixels."""
    tint = np.zeros_like(bgr)
    tint[np.asarray(mask, bool)] = (0, 0, 255)
    return cv2.addWeighted(bgr, 1.0, tint, 0.5, 0)


def _grab_pair(cam, timeout_s: float = 6.0):
    """Poll until BOTH the color and depth streams have warmed up, then grab a
    same-instant (rgb, depth) pair. The SUB sockets start cold each run (and depth
    subscribes lazily on the first call), so the very first frame is None -- retry
    instead of bailing. Returns (rgb, depth); either may still be None at timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if cam.get_rgb_frame() is not None and cam.get_depth_frame() is not None:
            break
        time.sleep(0.05)
    return cam.get_rgb_frame(), cam.get_depth_frame()   # both warm -> back-to-back


def _load_frame(path):
    """Load a captured .npz (11_capture_frame) -> (rgb, depth, intrinsics, T_pelvis_camera)."""
    z = np.load(path)
    rgb = np.asarray(z["rgb"], np.uint8)
    depth = np.asarray(z["depth"], np.float32)
    intr = {k: float(z[k]) for k in ("fx", "fy", "cx", "cy")}
    T_pc = Pose.from_homogeneous(np.asarray(z["T_pelvis_camera"], float))
    return rgb, depth, intr, T_pc


def _out_base(args) -> str:
    """Output stem for --save artifacts: next to the source (frame/image), or --save-dir."""
    if args.frame:
        src = args.frame
    elif args.image:
        src = args.image
    else:
        src = os.path.join("captures", f"segment_{time.strftime('%Y%m%d_%H%M%S')}")
    stem = os.path.splitext(src)[0]
    if args.save_dir:
        stem = os.path.join(args.save_dir, os.path.basename(stem))
    return stem


def main():
    ap = _rig.add_target_arg(argparse.ArgumentParser())
    ap.add_argument("--image", default=None,
                    help="segment a static image file instead of the live camera (SAM3 only)")
    ap.add_argument("--frame", default=None,
                    help="replay a captured .npz (11_capture_frame): rgb+depth+cloud, no robot")
    ap.add_argument("--mode", choices=["interactive", "auto"], default="interactive")
    ap.add_argument("--text", default=None, help="text prompt (auto, or seeds interactive)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--save", action="store_true",
                    help="save the mask + segmented cloud + overlay (feeds 10_graspgen_viz)")
    ap.add_argument("--save-dir", default=None,
                    help="dir for --save artifacts (default: next to the source frame)")
    args = ap.parse_args()

    # SAM3 client config (no GPU/robot needed); grasp.yaml: segment.{host,port,...}.
    cfg = load_configs(camera_file=_rig.camera_config_for(args.target))
    seg_cfg = cfg["grasp"].get("segment", {}) or {}
    prompt = {"text": args.text} if args.text else (seg_cfg.get("default_prompt") or {"text": "object"})

    def _client():
        return Sam3Client(host=seg_cfg.get("host", "127.0.0.1"),
                          port=int(seg_cfg.get("port", 5557)),
                          timeout_ms=int(seg_cfg.get("timeout_ms", 60000)))

    # --- source the frame: captured .npz (offline) | static image | live ZED ---
    robot, depth, intr, T_pc = None, None, None, None
    if args.frame:
        rgb, depth, intr, T_pc = _load_frame(args.frame)
        print(f"frame: {args.frame}  rgb {rgb.shape}  depth {depth.shape}  (offline replay)")
    elif args.image:
        if cv2 is None:
            print("cv2 is required to read --image.")
            return
        bgr = cv2.imread(args.image)
        if bgr is None:
            print(f"could not read image: {args.image}")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        print(f"image: {args.image}  rgb {rgb.shape}")
    else:
        robot = make_robot(connect_dds=False, connect_camera=True,
                           camera_config=_rig.camera_config_for(args.target))
        if not robot.camera.has_depth:
            print(f"[{args.target}] no head depth stream -- need the real ZED (or use --image).")
            return
        rgb, depth = _grab_pair(robot.camera)            # warm up both streams first
        if rgb is None or depth is None:
            missing = " + ".join(s for s, v in (("color", rgb), ("depth", depth)) if v is None)
            print(f"no {missing} frame after warmup -- is that stream publishing? "
                  f"(depth = 08_check_depth, color = 02_check_image; both must show frames)")
            return
        intr = cfg["camera"]["intrinsics"]
        T_pc = robot.frames.T_pelvis_camera(None)        # head cam is q-independent
        print(f"[{args.target}] frame: rgb {rgb.shape}  depth {depth.shape}")

    # --- segment ---
    try:
        if args.mode == "interactive":
            from g1_primitives.perception.segment_gui import refine_mask
            with _client() as c:
                mask = refine_mask(rgb, c, prompt, top_k=args.top_k)
        else:
            mask = Sam3Segmenter(_client, prompt, top_k=1).mask(rgb)
    except SegmentationAborted:
        print("segmentation aborted by operator.")
        return
    except (RuntimeError, TimeoutError) as e:        # no display / SAM3 down / SAM3 error
        print(f"segmentation failed: {e}")
        return

    n_mask = "(none / whole frame)" if mask is None else f"{int(np.count_nonzero(mask))} px"
    print(f"mask: {n_mask}")

    # mask overlay on the RGB (works for both modes; shows or saves headless)
    overlay = (_overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask)
               if cv2 is not None and mask is not None else None)
    if overlay is not None:
        _show_or_save(overlay, "segmentation (any key to exit)", "/tmp/segmented_mask.png")

    # cloud only when we have depth (live ZED or a captured --frame): the mask is applied
    # pre-deproject, with the captured/live intrinsics + T_pelvis_camera.
    obj = None
    if depth is not None and intr is not None and T_pc is not None:
        full = deproject_depth(depth, intr, T_pc)
        obj = deproject_depth(depth, intr, T_pc, mask=mask, rgb=rgb)   # rgb -> per-point color
        print(f"full-scene cloud      : {full.n} points")
        print(f"segmented object cloud: {obj.n} points")

    # --- save artifacts for the offline GraspGenX demo (10_graspgen_viz --pcd) ---
    if args.save:
        base = _out_base(args)
        os.makedirs(os.path.dirname(os.path.abspath(base)) or ".", exist_ok=True)
        saved = []
        if mask is not None:
            np.save(f"{base}_mask.npy", np.asarray(mask, bool))
            saved.append(f"{base}_mask.npy")
        if overlay is not None:
            cv2.imwrite(f"{base}_overlay.png", overlay)
            saved.append(f"{base}_overlay.png")
        if obj is not None and not obj.is_empty():
            voxel_m = (cfg["grasp"].get("graspgenx", {}) or {}).get("voxel_m")
            cloud = obj.voxel_downsampled(voxel_m) if voxel_m else obj   # match the live source
            import trimesh                                               # colored .ply carrier
            ply = f"{base}_cloud.ply"
            trimesh.PointCloud(cloud.points.astype(np.float64), colors=cloud.colors).export(ply)
            tag = " colored" if cloud.colors is not None else ""
            saved.append(f"{ply} ({cloud.n} pts{tag})")
        if saved:
            print("saved: " + "  ".join(saved))
            if obj is not None and not obj.is_empty():
                print(f"  next: python scripts/10_graspgen_viz.py --pcd {base}_cloud.ply")
        else:
            print("--save: nothing to write (no mask, and no depth for a cloud).")


if __name__ == "__main__":
    main()
