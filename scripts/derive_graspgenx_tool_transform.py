#!/usr/bin/env python
"""Provenance: DERIVE the GraspGenX grasp->wrist_yaw transform from kinematics.

NOT run at import -- this is the recipe behind the constants in `configs/grasp.yaml`
(`graspgenx.wristyaw_grasp_rpy` + `graspgenx.palm_offset_xyz`). Re-run it if the Dex3 URDF
is recalibrated or the GraspGenX gripper convention changes, then paste the printed values.

These are DERIVED, not the AprilTag top-down palm offset reused (07/apriltag) -- that value
was reverse-engineered for a different grasp and has no meaning for GraspGenX's frame.

What GraspGenX returns (verified from ~/repos/GraspGenX + the unitree_g1 mesh):
  * a pose in the GRASP frame: origin at the gripper base, +Z = approach (into the object),
    +X = the jaw-closing/opposition axis (thumb vs the two fingers), fingertips at +Z=`depth`
    (config.json `fingertip=[0,0,depth]`), object at the sweep-volume centre [0,0,depth].

Our hand (Dex3, from the URDF): fingers reach along wrist_yaw +X (approach), the thumb opposes
the index+middle across wrist_yaw +Y (closing), index/middle are split along wrist_yaw ±Z. So
the grasp->wrist axis map is fixed:
    grasp +Z(approach) -> wrist +X     grasp +X(closing) -> wrist +Y     grasp +Y -> wrist +Z
which is exactly rpy = [pi/2, 0, pi/2]  (Rz(pi/2)Ry(0)Rx(pi/2), the repo's rpy convention).

The translation is the grasp-frame origin in wrist_yaw coords. We anchor it FUNCTIONALLY:
our power_close contact midpoint `c` (FK) must sit at the object = grasp origin + depth*approach,
so  t = c - R @ [0,0,depth]  (= c - depth*x_hat, since approach maps to wrist +X).

  bash -ic 'use_conda g1_curobo && python scripts/derive_graspgenx_tool_transform.py'
"""
import json
import os

import numpy as np
import yaml

from g1_classical_manip.spatial.pose import rpy_to_matrix
from g1_classical_manip.ee.hand_base import LEFT, RIGHT
from g1_classical_manip.ee.hand_kinematics import Dex3Kinematics

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RPY = [np.pi / 2, 0.0, np.pi / 2]          # grasp(+Z approach,+X closing) -> wrist_yaw axes


def main():
    with open(os.path.join(_REPO, "assets/grippers/unitree_g1/config.json")) as f:
        depth = float(json.load(f)["fingertip"][-1])     # GraspGenX base->fingertip (+Z), ~0.07
    with open(os.path.join(_REPO, "configs/hands.yaml")) as f:
        close = yaml.safe_load(f)["dex3"]["presets"]["power_close"]

    kin = Dex3Kinematics()
    R = rpy_to_matrix(*RPY)
    approach_offset = R @ np.array([0.0, 0.0, depth])     # = depth * (R col2) = [depth,0,0]
    np.set_printoptions(precision=4, suppress=True, sign=" ")

    print(f"GraspGenX fingertip depth (base->object, +Z) : {depth:.4f} m")
    print(f"R_wristyaw_grasp  (rpy {np.round(RPY,4).tolist()}):\n{R}")
    print(f"  grasp +X(close)->wrist {R[:,0]}   +Y->wrist {R[:,1]}   +Z(approach)->wrist {R[:,2]}\n")

    for side in (RIGHT, LEFT):
        q7 = np.asarray(close[side], float)
        tips = kin.fingertips(side, q7)
        c = kin.contact_point(side, q7)                   # object centre in wrist_yaw frame
        t = c - approach_offset                           # grasp-frame origin in wrist_yaw frame
        fingers_mid = 0.5 * (tips["index"] + tips["middle"])
        span = np.linalg.norm(tips["thumb"] - fingers_mid)
        print(f"[{side:5s}] power_close contact midpoint c = {c}  (wrist_yaw frame, m)")
        print(f"         thumb {tips['thumb']}  idx {tips['index']}  mid {tips['middle']}")
        print(f"         thumb<->fingers span {span*1000:5.1f} mm   approach_offset {approach_offset}")
        print(f"   => palm_offset_xyz (t) = [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
        if side == RIGHT:
            print(f"\n   configs/grasp.yaml  graspgenx:  (RIGHT-hand reference; LEFT mirrored in code)")
            print(f"     wristyaw_grasp_rpy: [{RPY[0]:.7f}, {RPY[1]:.1f}, {RPY[2]:.7f}]")
            print(f"     palm_offset_xyz:    [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]\n")


if __name__ == "__main__":
    main()
