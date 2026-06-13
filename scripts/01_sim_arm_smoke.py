#!/usr/bin/env python
"""[HARDWARE/SIM] Go home, then trace a 10 cm wrist circle with the left arm via
the full plan -> retime -> execute stack. Phase-1 sim acceptance.

    python scripts/01_sim_arm_smoke.py            # uses configs/robot.yaml dds
Requires a DDS peer (unitree_mujoco ROBOT=g1 domain 1 iface lo, or hardware).
The right arm holds; only the left wrist circles. STOP-safe: returns home.
"""
import argparse

import numpy as np
import pinocchio as pin

from g1_classical_manip.factory import make_robot
from g1_classical_manip.motion.planner_base import Goal, CartesianWaypoint, World
from g1_classical_manip.tasks import primitives as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()

    robot = make_robot(connect_dds=True, build_perception=False)
    print("homing...")
    P.move_to_home(robot)
    q0 = robot.arm.get_current_dual_arm_q()
    TL, _ = robot.ik.fk(q0)

    # circle in the y-z plane around the current left EE position
    wps = []
    for i in range(args.steps + 1):
        th = 2 * np.pi * i / args.steps
        d = np.array([0.0, args.radius * np.cos(th), args.radius * np.sin(th)])
        M = TL.copy(); M.translation = TL.translation + d
        wps.append(CartesianWaypoint(left=M, label=f"c{i}"))

    traj = P.plan_cartesian(robot, q0, wps, world=World())
    print(f"executing circle: {traj.q.shape[0]} steps, {traj.duration:.1f}s, "
          f"max|qd| {traj.meta['max_qd']:.2f}")
    res = robot.executor.run(traj)
    print("execution:", res)
    P.move_to_home(robot)


if __name__ == "__main__":
    main()
