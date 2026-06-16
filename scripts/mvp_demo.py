#!/usr/bin/env python
"""MVP demo: the cuRobo-native action primitives against the Isaac sim.

    home()  ->  move(RIGHT, goal)  ->  close_hand(RIGHT)  ->  open_hand(RIGHT)  ->  home()

Prereqs: sim up on loopback (domain 1 / lo). Run:
  CYCLONEDDS_HOME=/opt/cyclonedds \
  CYCLONEDDS_URI=file://$HOME/repos/G1_classical_manip/configs/cyclonedds_loopback.xml \
  bash -ic 'use_conda g1_curobo && python scripts/mvp_demo.py'
"""
import numpy as np

from g1_classical_manip.factory import make_robot
from g1_classical_manip import primitives as P


def main():
    robot = make_robot(connect_dds=True, connect_hand=True,
                       dds_domain=1, dds_interface="lo", mode="sim")
    robot.executor.abort_thresh = 0.40   # sim tolerance

    side = P.RIGHT
    print("home :", P.home(robot))                       # go to the launch/ready pose

    q = robot.arm.get_current_dual_arm_q()
    cur = robot.planner.fk(side, q)                      # right-wrist pose at home
    goal = cur.copy()
    goal.translation = cur.translation + np.array([0.0, 0.0, 0.10])   # lift +10 cm z
    print("goal wrist pos:", np.round(goal.translation, 3))

    print("move :", P.move(robot, side, goal))
    print("close:", P.close_hand(robot, side))
    print("open :", P.open_hand(robot, side))
    print("home :", P.home(robot))                       # return to home
    print("DONE")


if __name__ == "__main__":
    main()
