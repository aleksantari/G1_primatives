#!/usr/bin/env python
"""[SIM] Single-arm pick-and-place in unitree_sim_isaaclab using the simulator's
GROUND-TRUTH block pose (no perception).

Launch the Isaac `pick_place_redblock_g1_29dof_dex3` task first on loopback DDS
(see SIM_NOTES.md), then:
    python scripts/run_pick_place_sim.py

The red block + robot-base world poses are hardcoded in configs (mirroring the
Isaac scene); GroundTruthBlockSource maps the block world pose into the pelvis
frame via the robot's fixed-base world pose. Arm / grasp approach / place target
come from configs/task_pick_place.yaml (arm: right, grasp.approach: home_aligned).

--assume-grasp closes the hand without grasp verification (Dex3 press sensors may
read 0 in sim) -- a fallback if stall/tau verification is flaky.
"""
import argparse

from g1_classical_manip.factory import make_robot
from g1_classical_manip.perception.ground_truth import GroundTruthBlockSource
from g1_classical_manip.tasks.pick_place_handover import run_pick_place
from g1_classical_manip.utils.rerun_viz import RerunLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=int, default=1)
    ap.add_argument("--interface", default="lo")
    ap.add_argument("--abort-thresh", type=float, default=0.5,
                    help="executor tracking-error abort (rad); sim PD lags looser "
                         "than hardware on fast moves")
    ap.add_argument("--assume-grasp", action="store_true",
                    help="close the hand without grasp verification (sim fallback)")
    ap.add_argument("--no-rerun", action="store_true")
    args = ap.parse_args()

    robot = make_robot(connect_dds=True, build_perception=False, connect_hand=True,
                       connect_camera=False, dds_domain=args.domain,
                       dds_interface=args.interface, mode="sim")

    # Ground-truth block pose (sim world -> pelvis) in place of perception.
    robot.perception = GroundTruthBlockSource.from_config(
        robot.frames, robot.cfg["pick_place"]["ground_truth_block_world"])

    if robot.executor is not None:
        robot.executor.abort_thresh = args.abort_thresh   # sim PD looser than hardware

    if args.assume_grasp:
        _close = robot.hand.close
        robot.hand.close = lambda side, verify=True: (_close(side, verify=False), True)[1]

    rr = RerunLogger(enabled=not args.no_rerun)
    if robot.executor is not None:
        robot.executor.rr = rr

    def on_transition(state, ctx):
        info = getattr(ctx.last, "info", "") or ""
        print(f"  -> {state}" + (f"   [{info}]" if info else ""), flush=True)
        rr.log_transition(state, ctx)

    print("running pick-place (ground-truth block)...", flush=True)
    end = run_pick_place(robot, on_transition=on_transition)
    print("FSM finished in state:", end, flush=True)


if __name__ == "__main__":
    main()
