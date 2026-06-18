#!/usr/bin/env python
"""06 - Perception check: live head-cam feed with AprilTag overlay (tag quad, numbered
corners, and the 3D pose axis triad X=red/Y=green/Z=blue at the tag CENTER), a throttled
readout of the camera-frame tag rpy + the resulting pelvis-frame block rpy, then the fused
detect() pose cross-checked against the ground-truth detector. The axis triad needs the
camera intrinsics filled (camera_*.yaml); without them only the 2D quad/corners are drawn.

Camera-only (no control DDS, nothing moves). 'q' quits the live window early.
  --target sim   Isaac mono head cam (camera_sim.yaml)
  --target real  ZED stereo, one eye (camera_real.yaml)

  bash -ic 'use_conda g1_curobo && python scripts/06_detect.py --target sim --secs 15'
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


def _rpy_deg(R):
    """Rotation matrix -> (roll, pitch, yaw) degrees, matching spatial.pose's
    rpy_to_matrix convention (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    R = np.asarray(R, float).reshape(3, 3)
    pitch = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.round(np.degrees([roll, pitch, yaw]), 1)


def _overlay(bgr, raw, accepted, det):
    if cv2 is None:
        return bgr
    # camera matrix for the 3D axis triad (needs filled intrinsics)
    K = dist = axis_len = None
    cp = getattr(det, "camera_params", None)
    if cp and cp[0] > 0:
        fx, fy, cx, cy = cp
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        dist = np.zeros(5)
        axis_len = getattr(det, "tag_size", 0.05) * 0.5
    for d in raw:
        c = d.corners.astype(int).reshape(-1, 1, 2)
        cv2.polylines(bgr, [c], True, (0, 255, 0), 2)
        # numbered corners (0..3): shows the corner order and that the pose origin is the
        # tag CENTER, not a corner.
        for i, (px, py) in enumerate(d.corners.astype(int)):
            cv2.circle(bgr, (int(px), int(py)), 3, (0, 0, 255), -1)
            cv2.putText(bgr, str(i), (int(px) + 4, int(py) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cx, cy = d.center.astype(int)
        cv2.putText(bgr, f"id{d.tag_id}", (cx - 12, cy - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        # 3D pose axes at the tag center: X red, Y green, Z blue (z points INTO the tag
        # surface -- the apriltag convention tag_to_block then flips 180deg about x).
        if K is not None and getattr(d, "pose_R", None) is not None:
            rvec, _ = cv2.Rodrigues(np.asarray(d.pose_R, float))
            tvec = np.asarray(d.pose_t, float).reshape(3, 1)
            cv2.drawFrameAxes(bgr, K, dist, rvec, tvec, axis_len)
    cv2.putText(bgr, f"accepted: {accepted}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)
    return bgr


def main():
    ap = _rig.add_target_arg(argparse.ArgumentParser())
    ap.add_argument("--object", default="block")
    ap.add_argument("--secs", type=float, default=15.0)
    args = ap.parse_args()

    robot = make_robot(connect_dds=False, connect_camera=True,   # camera only -> no motion
                       camera_config=_rig.camera_config_for(args.target))
    det, cam = robot.detector, robot.camera
    print("camera shape:", cam.shape(), "| detector:", type(det).__name__)
    view = _rig.Viewer("detect", save_path="/tmp/detect_feed.png")

    has_K = bool(getattr(det, "camera_params", None)) and det.camera_params[0] > 0
    if hasattr(det, "_ensure_detector") and not has_K:
        print("NOTE: camera intrinsics unset -> no 3D pose axes (fill camera_*.yaml).")
    accepted, t0, last_print = 0, time.time(), 0.0
    while time.time() - t0 < args.secs:
        bgr = cam.get_bgr_frame()
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if cv2 is not None else bgr
        raw = []
        if hasattr(det, "_ensure_detector"):          # AprilTagDetector only
            try:                                       # pose for the axis triad needs K
                raw = det._ensure_detector().detect(
                    gray, estimate_tag_pose=has_K,
                    camera_params=det.camera_params, tag_size=det.tag_size)
            except Exception:
                raw = []
        accepted += len(det.detect(gray))             # feed the fused/median filter
        now = time.time()
        if raw and now - last_print > 1.0:             # throttled full-pose readout
            last_print = now
            for d in raw:
                if getattr(d, "pose_R", None) is not None:
                    print(f"tag{d.tag_id} cam-frame: t="
                          f"{np.round(np.asarray(d.pose_t).reshape(3), 3)} m  "
                          f"rpy={_rpy_deg(d.pose_R)} deg")
            bp = det.block_pose(args.object)
            if bp is not None:
                print(f"  -> {args.object}@pelvis: t={np.round(bp.translation, 3)} m  "
                      f"rpy={_rpy_deg(bp.rotation)} deg")
        if not view.show(_overlay(bgr, raw, accepted, det)):
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
