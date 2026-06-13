#!/usr/bin/env python
"""Offline validation of the motion stack (NO robot, NO DDS).

Plans the Phase-2 acceptance motion home -> hover -> descend -> lift -> home in
the pelvis frame, retimes it, and asserts:
  * geometric continuity (no IK branch flips),
  * velocity / acceleration within the configured limits (the retimer is the
    speed authority -- the controller's velocity clip must never be the limiter),
  * non-degenerate, monotonic timing.

Optionally writes a Rerun .rrd of planned q(t). Run:
    bash -ic 'use_conda g1_classical_manip && python scripts/offline_plan_check.py'
"""
import argparse
import sys

import numpy as np
import pinocchio as pin

from g1_classical_manip.factory import make_robot
from g1_classical_manip.motion.planner_base import Goal, CartesianWaypoint, World, Box
from g1_classical_manip.tasks import primitives as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="write planned q to a .rrd")
    ap.add_argument("--rrd", default="offline_plan_check.rrd")
    args = ap.parse_args()

    robot = make_robot(connect_dds=False, build_perception=False)
    ik = robot.ik
    home = P.home_q(robot)
    TL, TR = ik.fk(home)

    def shift(T, dxyz):
        M = T.copy(); M.translation = T.translation + np.array(dxyz, float); return M

    # Gentle, well-conditioned pick cycle relative to the home EE pose (a real
    # grasp pose comes from perception; this demo only exercises the motion stack).
    g = robot.cfg["pick_place"]["grasp"]
    hov = g["hover_offset_z"]
    wps = [
        CartesianWaypoint(left=shift(TL, [0.05, 0.0, hov]), label="hover"),
        CartesianWaypoint(left=shift(TL, [0.05, 0.0, 0.0]), label="descend"),
        CartesianWaypoint(left=shift(TL, [0.05, 0.0, hov]), label="lift"),
        CartesianWaypoint(left=TL, label="home"),
    ]
    box = robot.cfg["pick_place"]["workspace_box"]
    world = World(workspace_box=Box(np.array(box["min"], float), np.array(box["max"], float)))

    path = robot.planner.plan(home, Goal(wps), world)
    traj = P.retime_from_config(path, robot.planner_cfg)

    rt = robot.planner_cfg["retimer"]
    vmax, amax = rt["max_velocity"], rt["max_acceleration"]
    jump = path.max_consecutive_jump()
    cont_thr = robot.planner_cfg["cartesian"]["continuity_jump_rad"]
    dt = np.diff(traj.t)

    print(f"path samples         : {path.n}")
    print(f"max consecutive jump : {jump:.4f} rad  (continuity thr {cont_thr})")
    print(f"trajectory steps     : {traj.q.shape[0]}   duration {traj.duration:.2f} s")
    print(f"max |qd|             : {traj.meta['max_qd']:.3f}  (limit {vmax})")
    print(f"max |qdd|            : {traj.meta['max_qdd']:.3f}  (limit {amax})")
    print(f"time-scale (singular): {traj.meta['max_time_scale']:.2f}x")

    ok = True
    ok &= jump <= cont_thr;                       print("  [continuity]", "PASS" if jump <= cont_thr else "FAIL")
    ok &= traj.meta["max_qd"] <= vmax * 1.05;     print("  [velocity]  ", "PASS" if traj.meta["max_qd"] <= vmax*1.05 else "FAIL")
    ok &= traj.meta["max_qdd"] <= amax * 1.20;    print("  [accel]     ", "PASS" if traj.meta["max_qdd"] <= amax*1.20 else "FAIL")
    ok &= bool(np.all(dt > 0));                   print("  [monotonic] ", "PASS" if np.all(dt > 0) else "FAIL")

    TLf, _ = ik.fk(path.q[-1])
    home_err = np.linalg.norm(TLf.translation - TL.translation)
    ok &= home_err < 5e-3;                         print(f"  [returns home] {'PASS' if home_err<5e-3 else 'FAIL'} ({home_err*1000:.1f} mm)")

    if args.rerun:
        try:
            import rerun as rr
            rr.init("g1_offline_plan_check")
            rr.save(args.rrd)
            for i in range(traj.q.shape[0]):
                rr.set_time_seconds("t", float(traj.t[i]))
                for j in range(14):
                    rr.log(f"q/{j:02d}", rr.Scalar(float(traj.q[i, j])))
            print(f"wrote {args.rrd}")
        except Exception as e:
            print(f"(rerun logging skipped: {e})")

    print("\nRESULT:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
