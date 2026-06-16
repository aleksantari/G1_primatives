# g1_classical_manip

A **cuRobo-native motion library** for the Unitree G1 (29-DoF, Dex3-1 hands), exposing a
small set of **action primitives** meant to be called as tools (eventually by an LLM agent).
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
close_hand(robot, side, verify=)  # verify=True returns whether an object is held
```
Each returns `Result(ok, info)`. Tasks are composed from these.

## Layout
```
g1_classical_manip/
  spatial/         pose.py        — numpy SE(3) Pose (the pin.SE3 replacement), pelvis frame
  motion/          curobo_planner — the only planner (plan_to_pose / plan_joint, fk)
                   executor       — streams JointTrajectory @ control_hz, abort-to-hold
                   planner_base   — JointPath / JointTrajectory containers
  ee/              hand_base, dex3, dex1 — grasp presets + verification
  robot_control/   robot_arm (G1_29_ArmController), robot_hand_unitree (threaded Dex3/Dex1)
  primitives.py    home / move / open_hand / close_hand
  factory.py       make_robot() — cuRobo planner + DDS controllers + executor
  perception/      DORMANT (pinocchio-bound; reworked cuRobo-native later)
configs/           robot, planner, hands (+ curobo/g1_dex3_curobo.yml, cyclonedds_loopback.xml)
scripts/           mvp_demo.py, hand_diag.py   (others are legacy — see CLAUDE.md)
tests/             test_pose, test_grasp       (pure-math, no robot)
```

## Environment
Runs in the **`g1_curobo`** conda env (Python 3.11, numpy 2, torch 2.9.1+cu128, cuRobo V2
from source, cyclonedds, unitree_sdk2py). **No pinocchio.** Build/install is not plain
`pip install` — see the header of `requirements-curobo.txt`. Always use the lane wrapper:

```bash
bash -ic 'use_conda g1_curobo && pytest tests/'                 # 12 pure-math tests
bash -ic 'use_conda g1_curobo && python -c "import g1_classical_manip.factory"'  # planner build
```

## Run the MVP (against the Isaac sim)
Bring the sim up (see `SIM_NOTES.md`), then:
```bash
cd ~/repos/G1_classical_manip
CYCLONEDDS_HOME=/opt/cyclonedds \
CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
bash -ic 'use_conda g1_curobo && python scripts/mvp_demo.py'   # home → move → close → open → home
```
`scripts/hand_diag.py` isolates the hand command→state loop (state stream + commanded motion).

## Status
cuRobo-native MVP is **sim-validated**: `home → move → close_hand → open_hand → home` runs
end-to-end on `unitree_sim_isaaclab` with zero executor aborts. `home`/`move` are
collision-aware via cuRobo. Hand presets are untuned placeholders. Perception, Rerun
logging, hardware bring-up, and richer primitives (grasp-frame offset, pick composite)
are open — see `HARDWARE_TODO.md` and the PLAN's roadmap.
