#!/usr/bin/env python
"""10 - SAM3 segmentation tool (camera-only, NO motion). Capture a head-cam frame, refine a
SAM3 mask (interactive cv2 GUI, or one auto call), apply it to the aligned depth, and report
the resulting single-object point cloud. For developing segmentation before the grasp run.

  --mode interactive  cv2 GUI: text/box/points, cycle candidates, accept (needs a display)
  --mode auto         one SAM3 call with --text / grasp.yaml default_prompt

Needs the real ZED depth stream + a running SAM3 server (python -m sam3.serving --port 5557).

  real: bash -ic 'use_conda g1_curobo && python scripts/10_segment.py --target real --mode interactive'
"""
import argparse

import numpy as np
import _rig
from g1_classical_manip.factory import make_robot
from g1_classical_manip.perception.depth import deproject_depth
from g1_classical_manip.perception.segment import Sam3Segmenter, SegmentationAborted
from g1_classical_manip.perception.sam3_client import Sam3Client

try:
    import cv2
except Exception:
    cv2 = None


def main():
    ap = _rig.add_target_arg(argparse.ArgumentParser())
    ap.add_argument("--mode", choices=["interactive", "auto"], default="interactive")
    ap.add_argument("--text", default=None, help="text prompt (auto, or seeds interactive)")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    robot = make_robot(connect_dds=False, connect_camera=True,
                       camera_config=_rig.camera_config_for(args.target))
    cam = robot.camera
    if not cam.has_depth:
        print(f"[{args.target}] no head depth stream on this target -- need the real ZED. "
              "Nothing to segment.")
        return

    seg_cfg = robot.cfg["grasp"].get("segment", {}) or {}
    prompt = {"text": args.text} if args.text else (seg_cfg.get("default_prompt") or {"text": "object"})

    def _client():
        return Sam3Client(host=seg_cfg.get("host", "127.0.0.1"),
                          port=int(seg_cfg.get("port", 5557)),
                          timeout_ms=int(seg_cfg.get("timeout_ms", 60000)))

    rgb = cam.get_rgb_frame()                         # same-instant pair
    depth = cam.get_depth_frame()
    if rgb is None or depth is None:
        print("no rgb/depth frame yet -- is the camera streaming?")
        return
    print(f"[{args.target}] frame: rgb {rgb.shape}  depth {depth.shape}")

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
    except RuntimeError as e:                         # no display / SAM3 error
        print(f"segmentation failed: {e}")
        return

    intr = robot.cfg["camera"]["intrinsics"]
    T_pc = robot.frames.T_pelvis_camera(None)         # head cam is q-independent
    full = deproject_depth(depth, intr, T_pc)
    obj = deproject_depth(depth, intr, T_pc, mask=mask)
    n_mask = "(none / whole frame)" if mask is None else f"{int(np.count_nonzero(mask))} px"
    print(f"mask: {n_mask}")
    print(f"full-scene cloud      : {full.n} points")
    print(f"segmented object cloud: {obj.n} points")

    # interactive => a display exists => show the colorized masked depth that feeds the cloud
    if args.mode == "interactive" and mask is not None and cv2 is not None:
        masked = np.where(mask, depth, np.nan).astype(np.float32)
        finite = np.isfinite(masked)
        out = np.zeros((*masked.shape, 3), np.uint8)
        if finite.any():
            lo, hi = np.percentile(masked[finite], [5, 95])
            norm = np.clip((masked - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            norm[~finite] = 0.0
            out = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            out[~finite] = 0
        try:
            cv2.imshow("segmented depth (any key to exit)", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            cv2.imwrite("/tmp/segmented_depth.png", out)
            print("no display -> saved /tmp/segmented_depth.png")


if __name__ == "__main__":
    main()
