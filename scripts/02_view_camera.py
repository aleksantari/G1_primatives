#!/usr/bin/env python
"""[HARDWARE] Show head-camera frames with AprilTag detections overlaid.

    python scripts/02_view_camera.py --host 192.168.123.164
Press q to quit. Needs the image server (teleimager) running on PC2.
"""
import argparse

from g1_classical_manip.factory import make_robot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.123.164")
    args = ap.parse_args()
    import cv2

    # camera backend is config-driven (configs/cameras.yaml); force the rig on
    # even though we don't connect DDS (perception-only viewer).
    robot = make_robot(connect_dds=False, build_perception=True,
                       connect_camera=True, image_host=args.host)
    det = robot.perception.detector

    while True:
        rgb = robot.cameras.get_rgb_frame("head")
        if rgb is None:
            continue
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for d in det.detect(gray, estimate_tag_pose=False):
            pts = d.corners.astype(int)
            cv2.polylines(bgr, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            c = d.center.astype(int)
            cv2.putText(bgr, str(d.tag_id), tuple(c), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
        cv2.imshow("head + apriltag", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
