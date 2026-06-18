#!/usr/bin/env python
"""Regenerate the cuRobo robot config `configs/curobo/g1_dex3_curobo.yml` from the
in-repo URDF. This is the PROVENANCE/recipe for that config -- it is NOT run at
import or runtime; run it by hand only when the URDF or the arms-only reduction
changes.

    bash -ic 'use_conda g1_curobo && python configs/curobo/build_g1_dex3.py'

Pipeline (consolidated from the original two scratch scripts in
~/repos/curobo_g1_smoke/: build_config.py + patch_config.py):

  1. cuRobo `RobotBuilder` reads the URDF and AUTO-generates the kinematics, the
     collision spheres (fit_collision_spheres -- slow, one-time), and the
     self-collision-ignore matrix (compute_collision_matrix).
  2. We hand-set `tool_frames` (the two wrist-yaw links) and inject the 27
     `lock_joints` (12 legs + waist_yaw + 14 Dex3 hand joints, all at 0) so cuRobo
     plans the 14 arm joints only (left 7 + right 7).
  3. PATCH the self-collision-ignore matrix: RobotBuilder's auto-matrix only ignored
     3 of the 6 adjacent torso<->shoulder pairs, which flags a FALSE "start in
     collision" at the home pose (all-zeros). We add all 6 bidirectionally.

Caveats:
  * The CURRENT committed config is hardware-validated (MVP runs on the real G1).
    Re-running overwrites it -- only regenerate intentionally, then re-validate.
  * RobotBuilder bakes ABSOLUTE paths into the yml (`asset_root_path`, urdf path).
    Fine on this machine; non-portable if the repo moves. Re-run here to refresh them.
  * The mode_16 URDF already has a fixed base (no floating_base_joint) and waist
    roll/pitch fixed; we additionally lock the legs + waist_yaw + hands.
"""
import os

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
ASSET = os.path.join(REPO, "assets", "g1")
URDF = os.path.join(ASSET, "g1_29dof_mode_16_dex3.urdf")
OUT = os.path.join(_HERE, "g1_dex3_curobo.yml")

# wrist-yaw links as tool frames (the repo's L_ee/R_ee = wrist_yaw + [0.05,0,0];
# the 5 cm offset is a later-integration detail, not part of the cuRobo config).
TOOL_FRAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]

# 27 non-arm joints to lock at the suspended (upright) posture -> arms-only active.
LOCK = {}
for s in ("left", "right"):
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"):
        LOCK[f"{s}_{j}_joint"] = 0.0
LOCK["waist_yaw_joint"] = 0.0
for s in ("left", "right"):
    for j in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"):
        LOCK[f"{s}_hand_{j}_joint"] = 0.0


def build():
    """Step 1+2: RobotBuilder from the URDF, then inject tool_frames + lock_joints."""
    from curobo.robot_builder import RobotBuilder

    b = RobotBuilder(URDF, ASSET, tool_frames=TOOL_FRAMES)
    print(f"base_link={b._base_link}  tool_frames={b.tool_frames}")
    print("fitting collision spheres (slow, one-time)...")
    b.fit_collision_spheres()
    print("computing self-collision matrix...")
    b.compute_collision_matrix()
    cfg = b.build()
    b.save(cfg, OUT)

    d = yaml.safe_load(open(OUT))
    d["kinematics"]["lock_joints"] = LOCK   # active joints = cspace joints not locked
    with open(OUT, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
    print(f"saved {OUT} with {len(LOCK)} locked joints")


def patch():
    """Step 3: torso_link <-> all 6 shoulder links ignored bidirectionally (they are
    adjacent; the auto-matrix missed 3, causing a false start-in-collision at home)."""
    d = yaml.safe_load(open(OUT))
    ig = d["kinematics"]["self_collision_ignore"]
    shoulders = [f"{s}_shoulder_{j}_link" for s in ("left", "right")
                 for j in ("pitch", "roll", "yaw")]

    def add(a, c):
        ig.setdefault(a, [])
        if c not in ig[a]:
            ig[a].append(c)

    for sh in shoulders:
        add("torso_link", sh)
        add(sh, "torso_link")
    with open(OUT, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
    print("torso_link ignores:", sorted(ig["torso_link"]))


if __name__ == "__main__":
    build()
    patch()
    print("DONE_BUILD")
