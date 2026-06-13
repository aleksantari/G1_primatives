#!/usr/bin/env python
"""[HARDWARE] Print live dual-arm joint q at 1 Hz -- DDS connectivity check.

    python scripts/00_dds_echo.py --domain 0 --interface enp5s0
unitree_mujoco: --domain 1 --interface lo
"""
import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--interface", default="")
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from g1_classical_manip.robot_control.robot_arm import G1_29_ArmController

    if args.interface:
        ChannelFactoryInitialize(args.domain, args.interface)
    else:
        ChannelFactoryInitialize(args.domain)

    arm = G1_29_ArmController(simulation_mode=True)  # read-only here; don't command
    print("subscribed. printing dual-arm q at 1 Hz (Ctrl-C to stop)")
    try:
        while True:
            q = arm.get_current_dual_arm_q()
            print("L:", np.round(np.rad2deg(q[:7]), 1), " R:", np.round(np.rad2deg(q[7:]), 1))
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
