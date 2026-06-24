#!/usr/bin/env python
"""08 - Head-camera DEPTH feed check. Pulls depth frames from the head camera, prints
per-frame stats (shape, % finite, min/median/max mm), and shows a colorized live window
('q' to quit; no display -> saves to /tmp/head_depth.png).

Camera-only (no control DDS, nothing moves). Head depth is a dedicated raw-float32 ZMQ
stream (720x1280, single left eye, MILLIMETERS, NaN/inf = invalid) -- see
docs/depth_integration_handoff.md.

  --target real  ZED head depth (configs/camera_real.yaml; port from the server cam_config)
  --target sim   no depth stream -> clean exit (configs/camera_sim.yaml is color-only)

  bash -ic 'use_conda g1_curobo && python scripts/08_check_depth.py --target real'
"""
import argparse
import time

import numpy as np

import _rig
from g1_classical_manip.factory import make_robot

try:
    import cv2
except Exception:
    cv2 = None


def _stats(depth: np.ndarray) -> str:
    """One-line summary of a float32 depth map (mm, NaN/inf = invalid)."""
    finite = np.isfinite(depth)
    pct = 100.0 * float(finite.mean())
    if not finite.any():
        return f"shape={tuple(depth.shape)} finite=0.0% (all invalid)"
    v = depth[finite]
    return (f"shape={tuple(depth.shape)} finite={pct:4.1f}% "
            f"min={v.min():7.1f} med={float(np.median(v)):7.1f} max={v.max():7.1f} mm")


def _colorize(depth: np.ndarray):
    """float32 depth (mm, NaN/inf = invalid) -> BGR uint8 via a 5-95% percentile JET map;
    invalid pixels render black. None if cv2 is unavailable (stats-only mode)."""
    if cv2 is None:
        return None
    finite = np.isfinite(depth)
    out = np.zeros((depth.shape[0], depth.shape[1], 3), np.uint8)
    if finite.any():
        lo, hi = np.percentile(depth[finite], [5, 95])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        norm[~finite] = 0.0
        out = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        out[~finite] = 0
    return out


def main():
    ap = _rig.add_target_arg(argparse.ArgumentParser())
    ap.add_argument("--secs", type=float, default=30.0)
    ap.add_argument("--every", type=float, default=1.0, help="seconds between stat prints")
    args = ap.parse_args()

    robot = make_robot(connect_dds=False, connect_camera=True,   # camera only -> no motion
                       camera_config=_rig.camera_config_for(args.target))
    if not robot.camera.has_depth:
        print(f"[{args.target}] no head depth stream on this target -- only the real ZED "
              "publishes depth (sim is color-only). Nothing to do.")
        return

    view = _rig.Viewer("head depth", save_path="/tmp/head_depth.png")
    print(f"[{args.target}] reading head depth (mm). 'q' to quit.")

    t0, t_last, n, got = time.time(), 0.0, 0, 0
    while time.time() - t0 < args.secs:
        depth = robot.camera.get_depth_frame()
        if depth is None:
            time.sleep(0.005)                          # port not ready yet; don't busy-spin
            continue
        n += 1
        got += 1
        now = time.time()
        if now - t_last >= args.every:
            print(f"  frame {n:5d}: {_stats(depth)}")
            t_last = now
        bgr = _colorize(depth)
        if bgr is not None and not view.show(bgr):
            break
    view.close()
    if got == 0:
        print("no depth frames received (port silent?). Re-run the probe in "
              "docs/depth_integration_handoff.md against the target robot.")
    else:
        print(f"received {got} depth frames.")


if __name__ == "__main__":
    main()
