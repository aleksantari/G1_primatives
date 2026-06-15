#!/usr/bin/env python
"""[HARDWARE/SIM] Go home, then run a left-arm pick cycle (hover -> descend -> lift
-> return) via the full plan -> retime -> execute stack. The right arm holds.

    python scripts/01_sim_arm_smoke.py --isaac    # unitree_sim_isaaclab on loopback (domain 1/lo)
    python scripts/01_sim_arm_smoke.py            # hardware: uses configs/robot.yaml dds

STOP-safe: returns home. Reports the executor's max joint tracking error.
"""
import argparse

import numpy as np

from g1_classical_manip.factory import make_robot
from g1_classical_manip.motion.planner_base import CartesianWaypoint, World
from g1_classical_manip.tasks import primitives as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.05)
    ap.add_argument("--domain", type=int, default=None, help="DDS domain (sim=1)")
    ap.add_argument("--interface", default=None, help="DDS interface (sim='lo')")
    ap.add_argument("--isaac", action="store_true",
                    help="unitree_sim_isaaclab on loopback: domain 1 / lo, with Dex3 hands")
    args = ap.parse_args()

    domain = 1 if args.isaac else args.domain
    iface = "lo" if args.isaac else args.interface
    mode = "sim" if args.isaac else None
    robot = make_robot(connect_dds=True, build_perception=False,
                       connect_hand=args.isaac,
                       dds_domain=domain, dds_interface=iface, mode=mode)
    if args.isaac:
        # Isaac's implicit-PD actuators track well (pick cycle ~0.12 rad) but lag
        # transiently on the fast initial homing move; 0.40 rad clears that without
        # masking a real divergence.
        robot.executor.abort_thresh = 0.40
    print("homing...")
    hr = P.move_to_home(robot)
    print("  home:", hr.info)
    q0 = robot.arm.get_current_dual_arm_q()
    TL, _ = robot.ik.fk(q0)

    def shift(dxyz):
        M = TL.copy(); M.translation = TL.translation + np.array(dxyz, float); return M

    # Well-conditioned left-arm pick cycle relative to the home EE pose
    # (the same motion offline_plan_check validates). Right arm holds.
    r = args.radius
    wps = [
        CartesianWaypoint(left=shift([0.05, 0.0, 2 * r]), label="hover"),
        CartesianWaypoint(left=shift([0.05, 0.0, 0.0]), label="descend"),
        CartesianWaypoint(left=shift([0.05, 0.0, 2 * r]), label="lift"),
        CartesianWaypoint(left=shift([0.0, 0.0, 0.0]), label="return"),
    ]
    traj = P.plan_cartesian(robot, q0, wps, world=World())
    print(f"executing pick cycle: {traj.q.shape[0]} steps, {traj.duration:.1f}s, "
          f"max|qd| {traj.meta['max_qd']:.2f}, time-scale {traj.meta['max_time_scale']:.1f}x")
    res = robot.executor.run(traj)
    print("execution:", res)
    P.move_to_home(robot)


if __name__ == "__main__":
    main()
