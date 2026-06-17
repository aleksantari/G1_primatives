#!/usr/bin/env python
"""Live head-camera AprilTag detection check against the running Isaac sim.

Pulls head frames over the teleimager ZMQ stream, runs detect(robot,'block'),
prints the pelvis-frame block pose, and cross-checks it against the ground_truth
detector. The camera->block distance ratio between the two is the tag_size
calibration factor (AprilTag depth scales linearly with the assumed tag_size).

Prereqs: sim up (head cam ZMQ on :55555), intrinsics set in camera.yaml. Run:
  bash -ic 'use_conda g1_curobo && python scripts/detect_check.py'
"""
import argparse

import numpy as np

from g1_classical_manip.factory import make_robot
from g1_classical_manip import primitives as P
from g1_classical_manip.perception.ground_truth import GroundTruthDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="block")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--save", default="/tmp/detect_frame.png")
    args = ap.parse_args()

    robot = make_robot(connect_dds=False, connect_camera=True)
    print("camera shape (h,w):", robot.camera.shape())

    try:
        import cv2
        bgr = robot.camera.get_bgr_frame()
        if bgr is not None and args.save:
            cv2.imwrite(args.save, bgr)
            print("saved frame:", args.save)
    except Exception as e:
        print("frame save skipped:", e)

    det = P.detect(robot, args.target, frames=args.frames)
    cam = robot.frames.T_pelvis_camera()

    if det is None:
        print("\ndetect() -> None. Raw detector debug:")
        gray = robot.camera.get_gray_frame()
        if gray is None:
            print("  no frame received from the ZMQ stream")
            return
        raw = robot.detector._ensure_detector().detect(
            gray, estimate_tag_pose=True,
            camera_params=robot.detector.camera_params, tag_size=robot.detector.tag_size)
        print(f"  raw tags seen: {[(d.tag_id, round(float(d.decision_margin), 1)) for d in raw]}")
        print(f"  configured ids: {robot.detector.id_names}, "
              f"min_margin: {robot.detector.min_decision_margin}")
        return

    print(f"\nAprilTag {args.target} @ pelvis: {np.round(det.pose.translation, 4)} m "
          f"(tag_id {det.tag_id}, score {det.score:.0f})")

    gt_block = (robot.cfg["perception"].get("ground_truth", {}) or {}).get("block", {})
    gt_pose = GroundTruthDetector.from_config(robot.frames, gt_block).block_pose(args.target)
    print(f"GroundTruth {args.target} @ pelvis: {np.round(gt_pose.translation, 4)} m")

    # translation delta
    dt = det.pose.translation - gt_pose.translation
    print(f"translation delta (apriltag - gt): {np.round(dt * 1000, 1)} mm  |  "
          f"norm {np.linalg.norm(dt) * 1000:.1f} mm")

    # rotation delta: geodesic angle between the two orientations
    R_err = det.pose.rotation.T @ gt_pose.rotation
    ang = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))
    print(f"apriltag quat wxyz: {np.round(det.pose.quaternion_wxyz(), 3)}  |  "
          f"gt quat wxyz: {np.round(gt_pose.quaternion_wxyz(), 3)}")
    print(f"rotation delta (apriltag vs gt): {ang:.1f} deg")

    d_at = np.linalg.norm(det.pose.translation - cam.translation)
    d_gt = np.linalg.norm(gt_pose.translation - cam.translation)
    print(f"camera->block distance: apriltag {d_at * 1000:.1f} mm, gt {d_gt * 1000:.1f} mm")
    if d_at > 1e-6:
        print(f"suggested tag_size = {robot.detector.tag_size * d_gt / d_at:.4f} m "
              f"(current {robot.detector.tag_size:.4f}; depth scale gap {d_at / d_gt:.3f})")


if __name__ == "__main__":
    main()
