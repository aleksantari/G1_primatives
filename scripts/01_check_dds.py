#!/usr/bin/env python
"""01 - READ-ONLY DDS state check: arms (rt/lowstate) + dex3 hands (rt/dex3/*/state).

No controllers, no publishers -> NOTHING is commanded, NOTHING moves. The safe first
contact with a real robot: confirms live state streams before any motion script.

  sim : CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
        bash -ic 'use_conda g1_curobo && python scripts/01_check_dds.py --target sim'
  real: bash -ic 'use_conda g1_curobo && python scripts/01_check_dds.py --target real'
"""
import argparse
import time

import numpy as np
import _rig

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, HandState_

ARM_IDX = list(range(15, 29))   # G1_29 arm motor indices: left 7 (15-21) + right 7 (22-28)


def _hand_q(sub):
    msg = sub.Read()
    return None if msg is None else np.array([msg.motor_state[i].q for i in range(7)])


def main():
    ap = argparse.ArgumentParser()
    _rig.add_target_arg(ap)
    ap.add_argument("--secs", type=float, default=10.0)
    args = ap.parse_args()

    domain, iface = _rig.dds_for(args.target)
    if iface:
        ChannelFactoryInitialize(domain, iface)
    else:
        ChannelFactoryInitialize(domain)

    low = ChannelSubscriber("rt/lowstate", LowState_); low.Init()
    lh = ChannelSubscriber("rt/dex3/left/state", HandState_); lh.Init()
    rh = ChannelSubscriber("rt/dex3/right/state", HandState_); rh.Init()
    print(f"[{args.target}] subscribed READ-ONLY (no commands). arm + dex3 state @1Hz, "
          f"Ctrl-C to stop.")

    t0 = time.time()
    while time.time() - t0 < args.secs:
        lm = low.Read()
        if lm is not None:
            q = np.rad2deg([lm.motor_state[i].q for i in ARM_IDX])
            print(f"arm deg  L {np.round(q[:7], 1)}  R {np.round(q[7:], 1)}")
        else:
            print("arm: (no rt/lowstate yet)")
        lq, rq = _hand_q(lh), _hand_q(rh)
        if lq is not None or rq is not None:
            print(f"hand q   L {np.round(lq, 2) if lq is not None else '--'}  "
                  f"R {np.round(rq, 2) if rq is not None else '--'}")
        time.sleep(1.0)
    print("done -- no commands were sent.")


if __name__ == "__main__":
    main()
