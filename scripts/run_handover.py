#!/usr/bin/env python
"""[HARDWARE] Run pick -> arm-to-arm handover -> place-with-receiver (Phase-6).

    python scripts/run_handover.py --host 192.168.123.164
giver/receiver, handover pose pair, and handshake timings come from
configs/task_handover.yaml. By construction the giver never opens unless the
receiver grasp is verified (no drops).
"""
import argparse
import threading
import time

from g1_classical_manip.factory import make_robot
from g1_classical_manip.tasks.pick_place_handover import run_pick_handover_place
from g1_classical_manip.utils.rerun_viz import RerunLogger


def perception_thread(robot, stop):
    while not stop.is_set():
        frames = robot.cameras.get_rgb_frames() if robot.cameras else {}
        if frames:
            q = robot.arm.get_current_dual_arm_q() if robot.connected else None
            robot.perception.process(frames, q14=q)
        time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.123.164")
    ap.add_argument("--no-rerun", action="store_true")
    args = ap.parse_args()

    # camera backend(s) are config-driven (configs/cameras.yaml)
    robot = make_robot(connect_dds=True, build_perception=True, image_host=args.host)
    stop = threading.Event()
    threading.Thread(target=perception_thread, args=(robot, stop), daemon=True).start()

    rr = RerunLogger(enabled=not args.no_rerun)
    if robot.executor is not None:
        robot.executor.rr = rr
    try:
        end = run_pick_handover_place(robot, on_transition=rr.log_transition)
        print("FSM finished in state:", end)
    finally:
        stop.set()


if __name__ == "__main__":
    main()
