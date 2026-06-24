# g1_classical_manip

A **cuRobo-native motion library** for the Unitree G1 (29-DoF, Dex3-1 hands), exposing a
small set of **action + perception primitives** meant to be **composed by an LLM agent** as a
**baseline for pick-and-place** tasks. This is the **beta** primitive surface — to be expanded.
It runs **off-board** on an RTX 5090 workstation that talks to the robot's PC2 (or the Isaac
sim) over CycloneDDS. The robot is **suspended on a back-plate mount**: only the 14 arm
joints and 2×7 hand joints are ever commanded.

The authoritative design is **`G1_CLASSICAL_MANIP_PLAN.md`**; this README is the quickstart.
**`HARDWARE_TODO.md`** lists what's still gated on the physical robot/camera. **`SIM_NOTES.md`**
covers the Isaac sim.

> **History:** v1 was a classical Cartesian-waypoint + pinocchio-IK + FSM pick-place pipeline.
> That was an experiment; the goal is now a clean primitive surface. v1's Cartesian planner,
> dual-arm IK, Ruckig retimer, and `tasks/` FSM have been removed — cuRobo is the single
> source of kinematics and planning. Perception is kept on disk but dormant.

## Motion stack
```
goal Pose ─► CuroboArmPlanner ─► JointTrajectory ─► Executor ─► DDS ─► (sim | robot)
            plan_pose / plan_cspace   (t,q,qd,qdd,    streams @ control_hz,
            (cuRobo native timing)     cuRobo dt)      aborts-to-hold on tracking error
```
cuRobo emits a time-parameterized, dynamically-feasible trajectory directly — no separate
retiming step. The executor holds the only handle to `G1_29_ArmController`.

## Primitives (`g1_classical_manip/primitives.py`)
```python
home(robot)                       # both arms to the launch pose (forearms forward), settle
move(robot, side, goal_pose)      # one wrist to goal_pose (pelvis frame); other arm holds
open_hand(robot, side)            # blocks until the fingers finish moving
close_hand(robot, side, verify=, fraction=)  # fraction<1 closes partway; verify=True = held?
detect(robot, target="block")     # head-cam AprilTag → object pose (pelvis frame)
```
Action verbs return `Result(ok, info)`; `detect` returns a `Detection` whose `.pose` feeds
straight into `move`. Tasks are composed from these.

## Layout
```
g1_classical_manip/
  spatial/         pose.py        — numpy SE(3) Pose (the pin.SE3 replacement), pelvis frame
  motion/          curobo_planner — the only planner (plan_to_pose / plan_joint, fk)
                   executor       — streams JointTrajectory @ control_hz, abort-to-hold
                   planner_base   — JointPath / JointTrajectory containers
  ee/              hand_base, dex3, dex1 — grasp presets + verification
  robot_control/   robot_arm (G1_29_ArmController), robot_hand_unitree (threaded Dex3/Dex1)
  primitives.py    home / move / open_hand / close_hand / detect
  factory.py       make_robot() — cuRobo planner + DDS controllers + executor + perception
  image_server/    HeadCamera — head-cam frames (zmq | teleimager | unitree_lerobot)
  perception/      transforms (frame math, cuRobo FK) · base (Detector seam) ·
                   apriltag_block · ground_truth   (Detector = apriltag | ground_truth)
configs/           robot, planner, hands, camera, perception (+ curobo/, cyclonedds_loopback.xml)
scripts/           01_check_dds → 08_check_depth bring-up ladder (--target sim|real) + hand_diag.py
tests/             test_pose, test_grasp, test_detect   (pure-math, no robot)
```

## Environment
Runs in the **`g1_curobo`** conda env (Python 3.11, numpy 2, torch 2.9.1+cu128, cuRobo V2
from source, cyclonedds, unitree_sdk2py). **No pinocchio.** Build/install is not plain
`pip install` — see the header of `requirements-curobo.txt`. Always use the lane wrapper:

```bash
bash -ic 'use_conda g1_curobo && pytest tests/'                 # 18 pure-math tests
bash -ic 'use_conda g1_curobo && python -c "import g1_classical_manip.factory"'  # planner build
```

## Run against the Isaac sim
Two terminals share one DDS bus on loopback (full notes + gotchas in `SIM_NOTES.md`).

**Terminal 1 — Isaac sim** (`launch_sim.sh` activates the `unitree` env itself; keep it running):
```bash
cd ~/repos/unitree_sim_isaaclab
UNITREE_DDS_IFACE=lo \
CYCLONEDDS_URI=file://$HOME/repos/G1_classical_manip/configs/cyclonedds_loopback.xml \
TASK=Isaac-PickPlace-RedBlock-G129-Dex3-Joint \
./launch_sim.sh                 # opens the Isaac viewer; add --headless for none
```
Wait until it prints `[DDSManager] DDS system initialized` and is stepping.

**Terminal 2 — control stack** (the `g1_curobo` env):
```bash
cd ~/repos/G1_classical_manip
export CYCLONEDDS_HOME=/opt/cyclonedds
export CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml   # MUST match the sim
use_conda g1_curobo

# incremental bring-up ladder (each `--target sim|real`; sim shown), run in order:
python scripts/01_check_dds.py  --target sim     # READ-ONLY arm + hand state (no motion)
python scripts/02_check_image.py --target sim    # head-cam feed (sim mono / real ZED eye)
python scripts/03_hands.py       --target sim --side right   # close/open hand primitives
python scripts/04_move.py        --target sim                # home (add --dz 0.1 for a lift)
python scripts/05_mvp_demo.py    --target sim    # home → move → close → open → home
python scripts/06_detect.py      --target sim    # AprilTag feed: 3D pose axes + rpy vs ground truth
python scripts/07_pick_place.py  --target sim    # pick+lift: home→open→detect→grasp→lift→home
python scripts/08_check_depth.py  --target real   # head-cam DEPTH feed (real only): mm stats + colorized view
```
Run them in order — `01`/`02` are read-only/no-motion (safe first contact), `03`–`05` and `07`
command the arms/hands, `06`/`08` are camera-only (`08` is the head **depth** feed — real only;
sim is color-only and exits cleanly). `02`/`06`/`07` work on both targets: `--target real`
uses the ZED head via `camera_real.yaml`. The camera scripts
don't use the `CYCLONEDDS_*` exports. `scripts/hand_diag.py` remains a low-level hand
command→state diagnostic.

**Reset the block (sim only)** — re-place the red block at its spawn pose between pick
attempts. Runs in the **`unitree`** (sim) env, on the same loopback bus:
```bash
UNITREE_DDS_IFACE=lo CYCLONEDDS_URI=file://$HOME/repos/G1_classical_manip/configs/cyclonedds_loopback.xml \
  bash -ic 'use_conda unitree && python reset_pose_test.py'
```

## Status
cuRobo-native MVP is **sim-validated**: `home → move → close_hand → open_hand → home` runs
end-to-end on `unitree_sim_isaaclab` with zero executor aborts; `home`/`move` are
collision-aware via cuRobo. `detect()` is **sim-validated** too — the head-cam AprilTag
(ID 14) block pose lands within ~3.5 cm of ground truth. Hand presets are untuned
placeholders. **On hardware (2026-06-18): the MVP `home → move → close_hand → open_hand → home`
runs end-to-end on the physical G1.** The arms power-on folded, so a direct un-planned launch home
(PD, velocity-capped) brings them to the launch pose, then the collision-aware planned `home` lands
within ~3° and `move` tracks clean. What makes it work on real: gravity comp **on** (hardware-
validated, off in sim), trajectory **`time_dilation 0.5`** (the velocity clip throttles PD torque,
so the full-speed plan can't be tracked — play it back slower), **`arm_velocity_limit ≥ ~12`**, and
debug mode set by the operator via the physical remote (we don't call `MotionSwitcher`). See
`docs/gravity_comp.md`. Still to do on the robot: `detect` validation + ZED intrinsics/mount,
hand-preset tuning, and the velocity-clip torque root-fix (time-dilation is a workaround). Rerun
logging is still open. `scripts/07_pick_place.py` is the first **composite task** — a single-arm
pick+lift (home→open→detect→approach→grasp→close→lift→home) with a URDF-measured palm/grasp
offset, the LLM-composable baseline to expand (place/handover, dual-arm, multi-object) — see
`HARDWARE_TODO.md` and the PLAN's roadmap.

The head-camera **depth** stream is now consumed (`HeadCamera.get_depth_frame()` — the real
ZED's dedicated raw-float32 720×1280 mm stream; validated on the G1 at ~89% finite / ~30 fps via
`scripts/08_check_depth.py`), the first step toward a point-cloud → grasp-pose pipeline. The wire
spec is in `docs/depth_integration_handoff.md`.
