#!/usr/bin/env python
"""[HARDWARE] Show head-camera frames with AprilTag detections overlaid.

    python scripts/02_view_camera.py --host 192.168.123.164
Press q to quit. Needs the image server (teleimager) running on PC2.
"""
import argparse

import numpy as np

from g1_classical_manip.factory import make_robot
from g1_classical_manip.image_server.image_client import HeadCamera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.123.164")
    ap.add_argument("--backend", default="teleimager")
    args = ap.parse_args()
    import cv2

    robot = make_robot(connect_dds=False, build_perception=True)
    det = robot.perception.detector
    cam = HeadCamera(host=args.host, backend=args.backend)

    while True:
        gray = cam.get_gray_frame()
        if gray is None:
            continue
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
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
