#!/usr/bin/env python
"""[HARDWARE/SIM] Go home, then trace a 10 cm wrist circle with the left arm via
the full plan -> retime -> execute stack. Phase-1 sim acceptance.

    python scripts/01_sim_arm_smoke.py            # uses configs/robot.yaml dds
Requires a DDS peer (unitree_mujoco ROBOT=g1 domain 1 iface lo, or hardware).
The right arm holds; only the left wrist circles. STOP-safe: returns home.
"""
import argparse

import numpy as np

from g1_classical_manip.factory import make_robot
from g1_classical_manip.motion.planner_base import CartesianWaypoint, World
from g1_classical_manip.tasks import primitives as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--domain", type=int, default=None, help="DDS domain (sim=1)")
    ap.add_argument("--interface", default=None, help="DDS interface (sim='lo')")
    ap.add_argument("--sim", action="store_true", help="unitree_mujoco: --domain 1 --interface lo, no hands")
    ap.add_argument("--isaac", action="store_true", help="unitree_sim_isaaclab: domain 1 / lo, WITH Dex3 hands")
    args = ap.parse_args()

    sim_dds = args.sim or args.isaac
    domain = 1 if sim_dds else args.domain
    iface = "lo" if sim_dds else args.interface
    mode = "sim" if sim_dds else None
    # Isaac has Dex3 hands (with --enable_dex3_dds) and applies arm position targets
    # via its own actuators; unitree_mujoco has no hands and torque-clips the arm.
    connect_hand = bool(args.isaac)
    robot = make_robot(connect_dds=True, build_perception=False, connect_hand=connect_hand,
                       dds_domain=domain, dds_interface=iface, mode=mode)
    if args.sim:
        # unitree_mujoco's arm is torque-controlled with our kp/kd, so it tracks
        # with more lag than the real robot's high-bandwidth position loop. Slow
        # the trajectory and loosen the abort threshold for the sim smoke.
        rt = robot.cfg["planner"]["retimer"]
        rt["max_velocity"], rt["max_acceleration"], rt["max_jerk"] = 1.0, 2.0, 15.0
        robot.executor.abort_thresh = 0.5
    elif args.isaac:
        # Isaac's implicit-PD actuators track well (pick cycle ~0.12 rad), but lag
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
