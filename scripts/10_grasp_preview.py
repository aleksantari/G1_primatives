#!/usr/bin/env python
"""10 - Online SAM3-segment + GraspGenX preview (NO robot motion).

Combines 10_segment (SAM3) and 10_graspgen_viz (GraspGenX in viser) into one LIVE loop against
the head camera (sim or real): grab an RGB-D frame -> SAM3 mask (interactive cv2 GUI / auto / none)
-> deproject the masked depth to a pelvis cloud -> GraspGenX -> cloud + ranked grasps + gripper
mesh in viser. It is exactly the perception+grasp-generation half of 09_graspgen with NO planner,
NO executor, NO DDS motion -- so you can iterate on SAM3 prompts and GraspGenX params online, on
sim or the real robot, without ever driving the arm. It reuses the SAME grasp source as 09, so what
you see here is what 09 will plan.

Loop: after each frame it prints the grasp summary and waits -- Enter re-runs on a fresh frame, 'q'
quits. `--once` does a single shot. `--latency` profiles each frame (depth/SAM3/GraspGenX) and saves
a graph (SAM3 timed at its ZMQ round-trip, so the cv2-GUI time is NOT counted as inference).

Needs a running SAM3 server (:5557), GraspGenX server (:5556), and a head DEPTH stream (sim ZMQ
:55556 / real ZED). viser serves the scene at :8080. No DDS -> no CYCLONEDDS_URI needed.

  sim : bash -ic 'use_conda g1_curobo && python scripts/10_grasp_preview.py --target sim'
  real: bash -ic 'use_conda g1_curobo && python scripts/10_grasp_preview.py --target real --segment interactive'
"""
import argparse
import os
import sys
import time

import _rig
from g1_classical_manip.factory import make_robot, _build_grasp_source
from g1_classical_manip.ee.hand_base import LEFT, RIGHT
from g1_classical_manip.latency import LOG


def _warm_camera(cam, timeout_s: float = 6.0) -> bool:
    """Poll until BOTH color and depth streams have warmed up (the SUBs start cold)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if cam.get_rgb_frame() is not None and cam.get_depth_frame() is not None:
            return True
        time.sleep(0.05)
    return False


def main():
    ap = _rig.add_target_arg(argparse.ArgumentParser(
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
    robot = make_robot(connect_dds=False, connect_camera=True,
                       camera_config=_rig.camera_config_for(args.target))
    if robot.camera is None or not robot.camera.has_depth:
        print(f"[{args.target}] no head depth stream -- need the sim depth PUB (:55556) or the real "
              f"ZED. Check 08_check_depth --target {args.target} first.")
        return 2

    # force the graspgenx source + segment mode + viser viz, with optional param overrides.
    g = robot.cfg["grasp"]
    g["grasp_source"] = "graspgenx"
    g.setdefault("segment", {})["mode"] = None if args.segment == "none" else args.segment
    gx = g.setdefault("graspgenx", {})
    gx.setdefault("visualize", {})["enabled"] = True
    gx["visualize"]["port"] = args.viser_port
    for key, val in (("planner", args.planner), ("num_grasps", args.num_grasps),
                     ("topk", args.topk), ("grasp_threshold", args.grasp_threshold)):
        if val is not None:
            gx[key] = val
    robot.grasp_source = _build_grasp_source(robot.frames, robot.cfg)

    print(f"[{args.target}] warming camera ...")
    if not _warm_camera(robot.camera):
        print("no rgb+depth after warmup -- is the sim/ZED publishing? (02_check_image / 08_check_depth)")
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
              f"top-down(approach≈-Z)={n_down}"
              + (f" | best conf {cands[0].confidence:.3f}" if cands else " (none)"))

        if args.latency:
            print(LOG.summary())
            stem, ext = os.path.splitext(args.latency_out)
            try:
                png = LOG.plot(f"{stem}_{n}{ext}", title=f"10_grasp_preview frame {n}")
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
