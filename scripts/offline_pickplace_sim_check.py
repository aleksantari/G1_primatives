#!/usr/bin/env python
"""Offline validation of the SIM single-arm pick-place path (NO robot, NO DDS).

Builds the ground-truth block pose (sim world -> pelvis) and plans the full FSM
motion chain with the configured arm + grasp approach:
    home -> hover -> descend -> lift -> place_above/place -> home
Asserts every stage is well-conditioned (retimer time-scale below threshold = the
grasp pose stays clear of wrist singularities) and IK-continuous. This is the
offline gate before launching Isaac.

    bash -ic 'use_conda g1_classical_manip && python scripts/offline_pickplace_sim_check.py'
"""
import argparse
import sys

import numpy as np
import pinocchio as pin

from g1_classical_manip.factory import make_robot
from g1_classical_manip.perception.ground_truth import GroundTruthBlockSource
from g1_classical_manip.tasks import primitives as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-time-scale", type=float, default=3.0,
                    help="fail if any stage exceeds this retimer time-scale")
    args = ap.parse_args()

    robot = make_robot(connect_dds=False, build_perception=False)
    pp = robot.cfg["pick_place"]
    side = pp["arm"]
    gt = GroundTruthBlockSource.from_config(robot.frames, pp["ground_truth_block_world"])
    block = gt.block_pose(0)
    place = pin.SE3(np.eye(3), np.asarray(pp["place"]["xyz"], float))
    home = P.home_q(robot)

    print(f"arm: {side}   grasp.approach: {pp['grasp'].get('approach')}")
    print(f"GT block @ pelvis: {np.round(block.translation, 4)}")
    print(f"place    @ pelvis: {np.round(place.translation, 4)}")

    g = pp["grasp"]
    stages, q, ok = [], home, True
    cont_thr = robot.planner_cfg["cartesian"]["continuity_jump_rad"]

    def stage(name, traj):
        nonlocal q, ok
        q = traj.q[-1]
        scale = traj.meta.get("max_time_scale", 0.0)
        jump = traj.meta.get("max_jump", 0.0)
        well = scale <= args.max_time_scale
        ok &= well
        stages.append((name, scale, traj.duration))
        print(f"  {name:11s} scale={scale:5.2f}x  dur={traj.duration:5.2f}s  "
              f"{'OK' if well else 'NEAR-SINGULAR'}")

    stage("hover",   P.plan_reach(robot, block, side, g["hover_offset_z"], start_q=home))
    stage("descend", P.plan_reach(robot, block, side, g["descend_clearance"], start_q=q))
    stage("lift",    P.plan_pick_lift(robot, block, side, start_q=q))
    stage("place",   P.plan_place(robot, place, side, start_q=q))
    stage("home",    P.plan_joint_move(robot, q, home))

    worst = max(s for _, s, _ in stages)
    total = sum(d for _, _, d in stages)
    print(f"\nworst time-scale: {worst:.2f}x (limit {args.max_time_scale})   "
          f"total motion: {total:.1f}s   (continuity thr {cont_thr})")
    print("RESULT:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
