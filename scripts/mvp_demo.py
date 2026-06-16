#!/usr/bin/env python
"""MVP demo: the 3 cuRobo-native primitives against the Isaac sim.

    move(RIGHT, goal)  ->  close_hand(RIGHT)  ->  open_hand(RIGHT)

Prereqs: sim up on loopback (domain 1 / lo). Run:
  CYCLONEDDS_HOME=/opt/cyclonedds \
  CYCLONEDDS_URI=file://$HOME/repos/G1_classical_manip/configs/cyclonedds_loopback.xml \
  bash -ic 'use_conda g1_curobo && python scripts/mvp_demo.py'
"""
import time

import numpy as np

from g1_classical_manip.factory import make_robot
from g1_classical_manip import primitives as P


def settle(robot, q, tol=0.05, timeout=5.0):
    """Hold joint target q until the (sim PD) measured pose converges within tol."""
    t0 = time.time()
    err = np.inf
    while time.time() - t0 < timeout:
        robot.executor.hold(q)
        err = float(np.max(np.abs(robot.arm.get_current_dual_arm_q() - q)))
        if err < tol:
            return err
        time.sleep(0.02)
    return err


def main():
    robot = make_robot(connect_dds=True, connect_hand=True,
                       dds_domain=1, dds_interface="lo", mode="sim")
    robot.executor.abort_thresh = 0.40   # sim tolerance

    side = P.RIGHT

    # The sim resets to ~all-zeros (arms straight down) which cuRobo flags as a
    # self-collision start. Un-tuck to cuRobo's collision-free 'ready' config and
    # SETTLE there (hold until the sim PD converges) so Cartesian planning has a
    # valid, collision-free, well-tracked start.
    q_ready = robot.planner.default_q()
    err = settle(robot, q_ready)
    print(f"settled at ready: max|meas-ready| = {np.rad2deg(err):.2f} deg")

    cur = robot.planner.fk(side, q_ready)            # right-wrist pose at the ready config
    goal = cur.copy()
    goal.translation = cur.translation + np.array([0.0, 0.0, 0.10])   # +10 cm z (validated-safe)

    print("goal wrist pos:", np.round(goal.translation, 3))
    print("move :", P.move(robot, side, goal))
    print("close:", P.close_hand(robot, side))
    print("open :", P.open_hand(robot, side))
    print("DONE")


if __name__ == "__main__":
    main()
