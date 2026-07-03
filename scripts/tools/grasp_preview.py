#!/usr/bin/env python
"""tools/grasp_preview - Online SAM3-segment + GraspGenX preview (NO robot motion).

Combines tools/segment (SAM3) and tools/graspgen_viz (GraspGenX in viser) into one LIVE loop
against the head camera (sim or real): grab an RGB-D frame -> SAM3 mask (interactive cv2 GUI /
auto / none) -> deproject the masked depth to a pelvis cloud -> GraspGenX -> cloud + ranked
grasps + gripper mesh in viser. It is exactly the perception+grasp-generation half of
examples/02_pick with NO planner motion, NO executor, NO DDS -- so you can iterate on SAM3
prompts and GraspGenX params online without ever driving the arm. It reuses the SAME grasp
source as the pick example, so what you see here is what the pick will plan.

Loop: after each frame it prints the grasp summary and waits -- Enter re-runs on a fresh frame,
'q' quits. `--once` does a single shot. `--latency` profiles each frame (depth/SAM3/GraspGenX)
and saves a graph (SAM3 timed at its ZMQ round-trip, so the cv2-GUI time is NOT counted).

Needs a running SAM3 server (:5557), GraspGenX server (:5556), and a head DEPTH stream (sim ZMQ
:55556 / real ZED). viser serves the scene at :8080. No DDS -> no CYCLONEDDS_URI needed.

  sim : bash -ic 'use_conda g1_curobo && python scripts/tools/grasp_preview.py --target sim'
  real: bash -ic 'use_conda g1_curobo && python scripts/tools/grasp_preview.py --target real --segment interactive'
"""
import argparse
import os
import sys

from g1_primitives import Robot, LEFT, RIGHT
from g1_primitives.api import console
from g1_primitives.latency import LOG


def main():
    ap = console.add_target_arg(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--segment", choices=["interactive", "auto", "none"], default="interactive",
                    help="SAM3 mask: interactive cv2 GUI | one auto call | none (whole frame)")
    ap.add_argument("--side", choices=[LEFT, RIGHT], default=RIGHT,
                    help="hand the grasps are mapped for (left is mirrored) -- viz only, no motion")
    ap.add_argument("--object", default="block", help="target label / SAM3 prompt seed")
    ap.add_argument("--planner", default=None, choices=["diffusion", "graspmoe", "topdown"],
                    help="GraspGenX planner (default: grasp.yaml)")
    ap.add_argument("--num-grasps", dest="num_grasps", type=int, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--grasp-threshold", dest="grasp_threshold", type=float, default=None)
    ap.add_argument("--viser-port", dest="viser_port", type=int, default=8080)
    ap.add_argument("--once", action="store_true", help="single shot (no re-run loop)")
    ap.add_argument("--latency", action="store_true",
                    help="profile each frame (depth/SAM3/GraspGenX) + save a graph per frame")
    ap.add_argument("--latency-out", default="latency_preview.png",
                    help="latency graph path; the frame index is appended (default ./latency_preview.png)")
    args = ap.parse_args()

    # motion-free robot: camera + frames only (no DDS, no arm/hand controllers).
    robot = Robot.offline(args.target, camera=True)
    if robot.camera is None or not robot.camera.has_depth:
        print(f"[{args.target}] no head depth stream -- need the sim depth PUB (:55556) or the real "
              f"ZED. Check checks/03_depth.py --target {args.target} first.")
        return 2

    # force the graspgenx source + segment mode + viser viz, with optional param overrides.
    gx_overrides = {k: v for k, v in (("planner", args.planner),
                                      ("num_grasps", args.num_grasps),
                                      ("topk", args.topk),
                                      ("grasp_threshold", args.grasp_threshold)) if v is not None}
    robot.set_segmenter("none" if args.segment == "none" else args.segment)
    robot.set_visualize(True, port=args.viser_port)
    robot.set_grasp_source("graspgenx", **gx_overrides)
    gx = robot.cfg["grasp"]["graspgenx"]

    print(f"[{args.target}] warming camera ...")
    if not robot.wait_for_frames(rgb=True, depth=True):
        print("no rgb+depth after warmup -- is the sim/ZED publishing? "
              "(checks/02_camera / checks/03_depth)")
        return 2
    print(f"segment={args.segment}  planner={gx.get('planner', 'default')}  viser=:{args.viser_port}\n"
          f"servers required: SAM3 :5557, GraspGenX :5556. Ctrl-C to quit.")

    n = 0
    while True:
        n += 1
        if args.latency:
            LOG.enable()                       # fresh timings for this frame
        try:
            cands = robot.grasp_source.grasps(robot, args.side, args.object)
        except KeyboardInterrupt:
            print("\nquit."); break
        except Exception as e:                 # noqa: BLE001 - a preview must survive a bad frame
            print(f"[{n}] grasp source error: {type(e).__name__}: {e}")
            cands = []

        n_obb = sum(1 for c in cands if c.extra.get("branch_tag") == "obb")
        n_down = sum(1 for c in cands if c.grasp_pose is not None
                     and c.grasp_pose.rotation[2, 2] < -0.7)
        print(f"[{n}] grasps -> {len(cands)} | obb={n_obb} diff={len(cands) - n_obb} | "
              f"top-down(approach~-Z)={n_down}"
              + (f" | best conf {cands[0].confidence:.3f}" if cands else " (none)"))

        if args.latency:
            print(LOG.summary())
            stem, ext = os.path.splitext(args.latency_out)
            try:
                png = LOG.plot(f"{stem}_{n}{ext}", title=f"grasp_preview frame {n}")
                if png:
                    print(f"latency graph -> {os.path.abspath(png)}")
            except Exception as e:             # noqa: BLE001 - reporting never crashes the preview
                print(f"latency graph failed: {e}")

        if args.once:
            break
        try:
            if input("  >> Enter to re-run on a fresh frame, 'q' to quit: ").strip().lower() in (
                    "q", "quit"):
                break
        except (EOFError, KeyboardInterrupt):
            break
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
