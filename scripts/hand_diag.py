#!/usr/bin/env python
"""Diagnose the dex3 hand command->state loop against the Isaac sim.

Answers two questions directly:
  1. Is hand STATE streaming?  (we read q/dq/tau/press for both hands)
  2. Do our COMMANDS update that state? (command close, dwell, watch q move)

Unlike mvp_demo, this dwells between close and open so motion is actually
observable (mvp_demo runs both with verify=False -> instant, no time to move).

The sim's apply path gates on BOTH hands having a command (action_provider_dds
:234). Our Dex3Controller publishes both continuously, but --both forces an
explicit close on both hands to rule that gate out.

Run (sim up, loopback):
  CYCLONEDDS_HOME=/opt/cyclonedds \
  CYCLONEDDS_URI=file://$HOME/repos/G1_classical_manip/configs/cyclonedds_loopback.xml \
  bash -ic 'use_conda g1_curobo && python scripts/hand_diag.py --side right'
"""
import argparse
import time

import numpy as np

from g1_classical_manip.factory import make_robot
from g1_classical_manip.ee.hand_base import LEFT, RIGHT


def snap(hand, side):
    st = hand.get_state(side)
    return st["q"].copy(), st


def watch(hand, side, secs, label):
    print(f"  [{label}] streaming {side} q for {secs:.1f}s:")
    t0 = time.time()
    last = None
    while time.time() - t0 < secs:
        st = hand.get_state(side)
        print(f"    t={time.time()-t0:4.1f}  q={np.round(st['q'],3)}  "
              f"tau={np.round(st['tau'],2)}  press={np.round(st['press'],1)}")
        last = st["q"].copy()
        time.sleep(0.25)
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=[LEFT, RIGHT], default=RIGHT)
    ap.add_argument("--both", action="store_true",
                    help="command BOTH hands (rules out the sim's both-hands apply gate)")
    ap.add_argument("--dwell", type=float, default=2.5, help="seconds to hold each pose")
    args = ap.parse_args()

    try:
        robot = make_robot(connect_dds=True, connect_hand=True,
                           dds_domain=1, dds_interface="lo", mode="sim")
    except Exception as e:
        print(f"make_robot failed (is the sim up on domain 1 / lo with --enable_dex3_dds?): {e}")
        return

    hand = robot.hand
    side = args.side

    # 1) state streaming?
    lq, _ = snap(hand, LEFT)
    rq, _ = snap(hand, RIGHT)
    print(f"STATE OK -> left q={np.round(lq,3)}  right q={np.round(rq,3)}")

    q_open = hand._preset("open", side)                 # per-hand aware
    q_close = hand._preset(hand.close_preset, side)
    print(f"open preset  ({side}) = {np.round(q_open,2)}")
    print(f"close preset ({side}) = {np.round(q_close,2)}  ({hand.close_preset})")

    q0, _ = snap(hand, side)

    # 2) command CLOSE and watch
    print(f"\n--> commanding CLOSE ({'both hands' if args.both else side + ' only'})")
    if args.both:
        hand.command(LEFT, hand._preset(hand.close_preset, LEFT))
        hand.command(RIGHT, hand._preset(hand.close_preset, RIGHT))
    else:
        hand.command(side, q_close)
    q_after_close = watch(hand, side, args.dwell, "after CLOSE")

    # 3) command OPEN and watch
    print(f"\n--> commanding OPEN ({'both hands' if args.both else side + ' only'})")
    if args.both:
        hand.command(LEFT, hand._preset("open", LEFT))
        hand.command(RIGHT, hand._preset("open", RIGHT))
    else:
        hand.command(side, q_open)
    q_after_open = watch(hand, side, args.dwell, "after OPEN")

    # verdict
    moved_close = float(np.max(np.abs(q_after_close - q0)))
    moved_open = float(np.max(np.abs(q_after_open - q_after_close)))
    print(f"\nVERDICT ({side}):")
    print(f"  max |dq| start->close = {moved_close:.3f} rad")
    print(f"  max |dq| close->open  = {moved_open:.3f} rad")
    if moved_close < 0.02 and moved_open < 0.02:
        print("  => COMMANDS DO NOT MOVE THE HAND. Topic publishes but sim isn't "
              "applying it (check --enable_dex3_dds, or the both-hands apply gate).")
    else:
        print("  => COMMANDS MOVE THE HAND. The loop works; mvp_demo just lacked "
              "dwell between close/open (verify=False returns instantly).")


if __name__ == "__main__":
    main()
