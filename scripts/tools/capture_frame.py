#!/usr/bin/env python
"""tools/capture_frame - Capture a head-cam RGB+depth frame for OFFLINE pipeline testing
(NO motion).

Grabs one same-instant (rgb, depth) pair and saves it -- together with the camera intrinsics,
the `T_pelvis_camera` extrinsic, and a READ-ONLY snapshot of the arm + hand q at capture time
-- to a self-contained `.npz`, so the deproject -> segment -> cloud (-> GraspGenX) pipeline AND
the depth-ESDF collision world can be re-run with no robot. Replay with
`tools/segment.py --frame <file>.npz` (grasp pipeline) or `tools/check_world.py --frame
<file>.npz` (collision world; the recorded q lets the self-filter mask the robot offline).
Also writes `_rgb.png` / `_depth.png` previews next to the npz.

  bash -ic 'use_conda g1_curobo && python scripts/tools/capture_frame.py --target real --out captures/scene1.npz'
"""
import argparse
import os
import sys
import time

import numpy as np

from g1_primitives import Robot
from g1_primitives.api import console

try:
    import cv2
except Exception:
    cv2 = None


def _grab_state(target, timeout_s: float = 2.0):
    """Best-effort READ-ONLY DDS snapshot of the arm + hand q at capture time (raw subscribers
    like checks/01_dds -- NO controllers, nothing commanded, nothing moves), so the offline
    collision-world self-filter (tools/check_world --frame) can mask the robot at the right pose.
    Returns (q14, hand_q_left, hand_q_right) in repo / Dex3 get_q order; any may be None if the
    bus is quiet. Never fails the capture -- a frame without q just self-filters at home offline."""
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, HandState_
        domain, iface = console.dds_for(target)
        ChannelFactoryInitialize(domain, iface) if iface else ChannelFactoryInitialize(domain)
        low = ChannelSubscriber("rt/lowstate", LowState_); low.Init()
        lh = ChannelSubscriber("rt/dex3/left/state", HandState_); lh.Init()
        rh = ChannelSubscriber("rt/dex3/right/state", HandState_); rh.Init()
        arm_idx = list(range(15, 29))                  # G1_29 arm motors: left 7 (15-21) + right 7 (22-28)
        q14 = lq = rq = None
        t0 = time.time()
        while time.time() - t0 < timeout_s and (q14 is None or lq is None or rq is None):
            lm = low.Read()
            if lm is not None and q14 is None:
                q14 = np.array([lm.motor_state[i].q for i in arm_idx], float)
            ls, rs = lh.Read(), rh.Read()
            if ls is not None and lq is None:
                lq = np.array([ls.motor_state[i].q for i in range(7)], float)
            if rs is not None and rq is None:
                rq = np.array([rs.motor_state[i].q for i in range(7)], float)
            if q14 is None or lq is None or rq is None:
                time.sleep(0.05)
        return q14, lq, rq
    except Exception as e:                             # noqa: BLE001 - best-effort; capture saves anyway
        print(f"  (arm/hand q not captured: {e})")
        return None, None, None


def main():
    ap = console.add_target_arg(argparse.ArgumentParser())
    ap.add_argument("--out", default=None,
                    help="output .npz (default captures/capture_<timestamp>.npz)")
    args = ap.parse_args()

    robot = Robot.offline(args.target, camera=True)
    if not robot.camera.has_depth:
        print(f"[{args.target}] no head depth stream -- nothing to capture.")
        return
    if not robot.wait_for_frames(rgb=True, depth=True):   # warm up both cold streams
        print("no color+depth frame after warmup -- is that stream publishing?")
        return
    rgb, depth = robot.rgb(), robot.depth()               # both warm -> back-to-back pair
    depth = np.asarray(depth, np.float32)

    intr = robot.cfg["camera"]["intrinsics"]
    T_pc = robot.frames.T_pelvis_camera(None).homogeneous     # head cam is q-independent

    # READ-ONLY snapshot of the robot pose at capture time, so tools/check_world --frame can run
    # the collision-world self-filter offline (mask the robot's own arm/fingers). Best-effort.
    q14, hq_l, hq_r = _grab_state(args.target)
    empty = np.array([])

    out = args.out or os.path.join("captures", f"capture_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out,
             rgb=np.ascontiguousarray(rgb, np.uint8),
             depth=depth,                                      # (H,W) float32 mm, NaN/inf = invalid
             fx=float(intr["fx"]), fy=float(intr["fy"]),
             cx=float(intr["cx"]), cy=float(intr["cy"]),
             width=int(intr.get("width", rgb.shape[1])),
             height=int(intr.get("height", rgb.shape[0])),
             T_pelvis_camera=np.asarray(T_pc, float),          # (4,4) optical->pelvis
             q14=(q14 if q14 is not None else empty),          # arm q (repo order) at capture, for
             hand_q_left=(hq_l if hq_l is not None else empty),   # the offline self-filter; empty if
             hand_q_right=(hq_r if hq_r is not None else empty),  # the bus was quiet
             target=str(args.target))

    finite = np.isfinite(depth)
    v = depth[finite]
    q_status = (f"arm+hand q recorded (arm {np.round(np.rad2deg(q14), 0).astype(int).tolist()} deg)"
                if q14 is not None else "no arm/hand q (offline self-filter -> home)")
    print(f"saved {out}")
    print(f"  rgb {rgb.shape}  depth {depth.shape}  finite {100 * finite.mean():.1f}%  "
          f"depth[min/med/max] {v.min():.0f}/{np.median(v):.0f}/{v.max():.0f} mm")
    print(f"  pose: {q_status}")
    print(f"  replay: python scripts/tools/segment.py --frame {out} --mode interactive")
    print(f"          python scripts/tools/check_world.py --frame {out} --visualize")

    # eyeball previews next to the npz
    if cv2 is not None:
        base = os.path.splitext(out)[0]
        cv2.imwrite(base + "_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        d = np.where(finite, depth, np.nan)
        lo, hi = (np.percentile(v, [5, 95]) if finite.any() else (0.0, 1.0))
        norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        norm[~finite] = 0.0
        col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        col[~finite] = 0
        cv2.imwrite(base + "_depth.png", col)
        print(f"  previews: {base}_rgb.png  {base}_depth.png")


if __name__ == "__main__":
    main()
    # The work is done + saved above. cuRobo/torch (CUDA) + the ZMQ camera daemon threads
    # abort during interpreter teardown ("terminate called ... / Aborted") AFTER all of it.
    # Hard-exit to skip the messy atexit/destructor teardown (exit 0, no core dump).
    sys.stdout.flush()
    os._exit(0)
