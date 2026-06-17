#!/usr/bin/env python
"""06 - Perception check: live head-cam feed with AprilTag overlay, then the fused
detect() pose cross-checked against the ground-truth detector.

Camera-only (no control DDS, nothing moves). Sim now (head-cam ZMQ); the real-robot
image client is not wired yet. 'q' quits the live window early.

  CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
  bash -ic 'use_conda g1_curobo && python scripts/06_detect.py --object block --secs 15'
"""
import argparse
import time

import numpy as np
import _rig
from g1_classical_manip.factory import make_robot
from g1_classical_manip.perception.ground_truth import GroundTruthDetector

try:
    import cv2
except Exception:
    cv2 = None


def _overlay(bgr, raw, accepted):
    if cv2 is None:
        return bgr
    for d in raw:
        c = d.corners.astype(int).reshape(-1, 1, 2)
        cv2.polylines(bgr, [c], True, (0, 255, 0), 2)
        cx, cy = d.center.astype(int)
        cv2.putText(bgr, f"id{d.tag_id}", (cx - 12, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
    cv2.putText(bgr, f"accepted: {accepted}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)
    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="block")
    ap.add_argument("--secs", type=float, default=15.0)
    args = ap.parse_args()

    robot = make_robot(connect_dds=False, connect_camera=True)   # camera only -> no motion
    det, cam = robot.detector, robot.camera
    print("camera shape:", cam.shape(), "| detector:", type(det).__name__)
    view = _rig.Viewer("detect", save_path="/tmp/detect_feed.png")

    accepted, t0 = 0, time.time()
    while time.time() - t0 < args.secs:
        bgr = cam.get_bgr_frame()
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if cv2 is not None else bgr
        raw = []
        if hasattr(det, "_ensure_detector"):          # AprilTagDetector only
            try:
                raw = det._ensure_detector().detect(gray, estimate_tag_pose=False)
            except Exception:
                raw = []
        accepted += len(det.detect(gray))             # feed the fused/median filter
        if not view.show(_overlay(bgr, raw, accepted)):
            break
    view.close()

    pose = det.block_pose(args.object)
    if pose is None:
        print(f"no fused pose for '{args.object}' (no tag seen?).")
        return
    print(f"\n{args.object} @ pelvis: {np.round(pose.translation, 4)} m")

    gt_cfg = (robot.cfg["perception"].get("ground_truth", {}) or {}).get("block", {})
    if gt_cfg:
        gt = GroundTruthDetector.from_config(robot.frames, gt_cfg).block_pose(args.object)
        if gt is not None:
            dt = pose.translation - gt.translation
            R_err = pose.rotation.T @ gt.rotation
            ang = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
            print(f"ground truth     : {np.round(gt.translation, 4)} m")
            print(f"delta: {np.round(dt * 1000, 1)} mm (norm {np.linalg.norm(dt) * 1000:.1f}), "
                  f"rotation {ang:.1f} deg")


if __name__ == "__main__":
    main()
