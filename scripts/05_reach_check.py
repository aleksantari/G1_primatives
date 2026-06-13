#!/usr/bin/env python
"""[HARDWARE] Perceive the block, move the wrist to the perceived HOVER pose, and
STOP (no grasp). Use to calibrate the table plane / grasp offsets safely.

    python scripts/05_reach_check.py --host 192.168.123.164 --tag 0
"""
import argparse
import time

import numpy as np

from g1_classical_manip.factory import make_robot
from g1_classical_manip.image_server.image_client import HeadCamera
from g1_classical_manip.tasks import primitives as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.123.164")
    ap.add_argument("--tag", type=int, default=0)
    ap.add_argument("--side", default="left")
    ap.add_argument("--backend", default="teleimager")
    args = ap.parse_args()

    robot = make_robot(connect_dds=True, build_perception=True)
    cam = HeadCamera(host=args.host, backend=args.backend)

    print("homing..."); P.move_to_home(robot)
    print("perceiving...")
    for _ in range(2 * robot.perception.median_frames):
        g = cam.get_gray_frame()
        if g is not None:
            robot.perception.detect(g, q14=robot.arm.get_current_dual_arm_q())
        time.sleep(0.03)
    block = robot.perception.block_pose(args.tag)
    if block is None:
        print("no stable block detection; aborting."); return
    print("block @ pelvis:", np.round(block.translation, 3))

    hover = robot.cfg["pick_place"]["grasp"]["hover_offset_z"]
    traj = P.plan_reach(robot, block, args.side, clearance=hover)
    print(f"moving to hover: {traj.q.shape[0]} steps, {traj.duration:.1f}s, "
          f"time-scale {traj.meta['max_time_scale']:.1f}x -- STOPPING at hover.")
    res = robot.executor.run(traj)
    print("execution:", res)
    print("Inspect alignment, then re-home manually if satisfied.")


if __name__ == "__main__":
    main()
