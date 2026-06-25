#!/usr/bin/env python
"""10 - SAM3 segmentation tool (NO motion). Get a head-cam frame (live ZED) OR a static
image file, refine a SAM3 mask (interactive cv2 GUI, or one auto call), and report the
result. For developing segmentation before the grasp run.

  --image PATH        segment a static image file -- SAM3 only, NO robot / depth / cloud
                      (the tightest loop for iterating on prompts; pairs with --mode auto)
  --mode interactive  cv2 GUI: text/box/points, cycle candidates, accept (needs a display)
  --mode auto         one SAM3 call with --text / grasp.yaml default_prompt

Live mode needs the real ZED depth stream; both modes need a running SAM3 server
(python -m sam3.serving --port 5557).

  image: bash -ic 'use_conda g1_curobo && python scripts/10_segment.py --image ~/repos/sam3/assets/images/groceries.jpg --mode auto --text "a bottle"'
  live : bash -ic 'use_conda g1_curobo && python scripts/10_segment.py --target real --mode interactive'
"""
import argparse

import numpy as np
import _rig
from g1_classical_manip.factory import make_robot, load_configs
from g1_classical_manip.perception.depth import deproject_depth
from g1_classical_manip.perception.segment import Sam3Segmenter, SegmentationAborted
from g1_classical_manip.perception.sam3_client import Sam3Client

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


def main():
    ap = _rig.add_target_arg(argparse.ArgumentParser())
    ap.add_argument("--image", default=None,
                    help="segment a static image file instead of the live camera (SAM3 only)")
    ap.add_argument("--mode", choices=["interactive", "auto"], default="interactive")
    ap.add_argument("--text", default=None, help="text prompt (auto, or seeds interactive)")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    # SAM3 client config (no GPU/robot needed); grasp.yaml: segment.{host,port,...}.
    cfg = load_configs(camera_file=_rig.camera_config_for(args.target))
    seg_cfg = cfg["grasp"].get("segment", {}) or {}
    prompt = {"text": args.text} if args.text else (seg_cfg.get("default_prompt") or {"text": "object"})

    def _client():
        return Sam3Client(host=seg_cfg.get("host", "127.0.0.1"),
                          port=int(seg_cfg.get("port", 5557)),
                          timeout_ms=int(seg_cfg.get("timeout_ms", 60000)))

    # --- source the frame: static image (no robot) OR the live ZED (rgb+depth) ---
    robot, depth = None, None
    if args.image:
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
        rgb, depth = robot.camera.get_rgb_frame(), robot.camera.get_depth_frame()
        if rgb is None or depth is None:
            print("no rgb/depth frame yet -- is the camera streaming?")
            return
        print(f"[{args.target}] frame: rgb {rgb.shape}  depth {depth.shape}")

    # --- segment ---
    try:
        if args.mode == "interactive":
            from g1_classical_manip.perception.segment_gui import refine_mask
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

    # mask overlay on the RGB (works for both modes; saves headless)
    if cv2 is not None and mask is not None:
        _show_or_save(_overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask),
                      "segmentation (any key to exit)", "/tmp/segmented_mask.png")

    # cloud counts only when we have depth (live ZED): mask applied pre-deproject
    if depth is not None:
        intr = cfg["camera"]["intrinsics"]
        T_pc = robot.frames.T_pelvis_camera(None)    # head cam is q-independent
        full = deproject_depth(depth, intr, T_pc)
        obj = deproject_depth(depth, intr, T_pc, mask=mask)
        print(f"full-scene cloud      : {full.n} points")
        print(f"segmented object cloud: {obj.n} points")


if __name__ == "__main__":
    main()
