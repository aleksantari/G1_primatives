# g1_classical_manip — Plan & Source of Truth

A **cuRobo-native motion library** for the Unitree G1 (29-DoF, Dex3-1 hands), exposing a
small, growing set of **configurable action primitives** intended to be called as tools —
ultimately by an LLM agent. It runs **off-board** on an RTX 5090 workstation talking to the
robot's PC2 (or the Isaac sim) over CycloneDDS. The robot is **suspended on a back-plate
mount**: only the 14 arm joints + 2×7 hand joints are ever commanded.

This document is the source of truth. Read it (and `CLAUDE.md`, `HARDWARE_TODO.md`) before
writing code.

---

## 0. Pivot — what changed and why (read this first)

v1 was a classical **pick → place → handover** pipeline: Cartesian-waypoint planner + stock
dual-arm IK (pinocchio/CasADi/Ipopt) + a Ruckig retimer + an FSM in `tasks/`, with AprilTag
perception. It was built as an experiment and **offline-validated**, but the goal changed:
we want a **clean primitive surface** an agent can compose, not a hard-coded task graph.

So v1's task-specific and now-redundant layers were **deleted** (recoverable from git):
`motion/cartesian_planner.py`, `robot_control/robot_arm_ik.py`, `motion/retimer.py`,
`tasks/*`. cuRobo V2 became the single source of kinematics and planning, which also let us
**drop pinocchio entirely** (the sole reason for the old `numpy<2` pin) and unify onto one
env. `perception/*` is kept on disk but **dormant** (pinocchio-bound) — reworked later.

The throughline that survived: the three-layer motion stack with hard seams, all poses in
the pelvis frame, dual-arm-as-one-14-vector, config-driven, and abort-to-hold safety.

---

## 1. Current state (what exists and works)

**MVP — sim-validated** on `unitree_sim_isaaclab`: `home → move → close_hand → open_hand →
home` runs end-to-end with zero executor aborts; `home`/`move` are collision-aware via cuRobo.

Four primitives (`g1_classical_manip/primitives.py`), each returning `Result(ok, info)`:
- `home(robot)` — both arms to the launch pose (forearms forward), settle.
- `move(robot, side, goal_pose)` — one wrist to `goal_pose` (pelvis frame); the other arm holds.
- `open_hand(robot, side)` / `close_hand(robot, side, verify=)` — block until the motion
  completes; `verify=True` returns whether an object is held (stall / tau / press).

Validated numbers: cuRobo FK parity vs the old pinocchio model < 0.001 mm / 0.027°; `Pose`
util convention-faithful to ~1e-15; cuRobo→sim EE error ~0.45 cm; native plan ~1.1 rad/s.

---

## 2. Architecture

### Motion stack (hard seams)
```
goal Pose ─► CuroboArmPlanner ─► JointTrajectory ─► Executor ─► DDS ─► (sim | robot)
            plan_pose / plan_cspace  (t,q,qd,qdd at    streams @ control_hz,
            (cuRobo native timing)    cuRobo's dt)      abort-to-hold on tracking error
```
cuRobo's trajopt emits a time-parameterized, dynamically-feasible trajectory **directly**, so
the `JointTrajectory` is built straight from `result.get_interpolated_plan()` — **no separate
retimer**. The executor owns the only handle to `G1_29_ArmController`; planning never streams.

### Layout (current)
```
g1_classical_manip/
├── spatial/pose.py            # numpy SE(3) Pose (pelvis frame); the pin.SE3 replacement
├── motion/
│   ├── curobo_planner.py      # CuroboArmPlanner: plan_to_pose, plan_joint, fk, default_q
│   ├── executor.py            # streams JointTrajectory @ control_hz; prime + abort-to-hold; settle
│   └── planner_base.py        # JointPath / JointTrajectory containers (DOF=14)
├── ee/                        # hand_base (ABC), dex3, dex1 — presets + grasp verification
├── robot_control/
│   ├── robot_arm.py           # G1_29_ArmController (vendored: 250 Hz dual-arm streaming)
│   ├── robot_hand_unitree.py  # threaded Dex3/Dex1 controllers (vendored DDS plumbing)
│   └── motion_switcher.py     # Enter/Exit debug mode (hardware)
├── primitives.py              # home / move / open_hand / close_hand  (the agent tool surface)
├── factory.py                 # make_robot(): cuRobo planner + DDS controllers + executor
├── perception/                # DORMANT (pinocchio-bound; reworked cuRobo-native later)
├── image_server/, utils/      # DORMANT / partially used
configs/  robot, planner, hands, curobo/g1_dex3_curobo.yml, cyclonedds_loopback.xml
scripts/  mvp_demo.py, hand_diag.py   (numbered + run_* scripts are legacy; need porting)
tests/    test_pose, test_grasp       (pure-math; 12 pass)
```

### Architecture rules (enforced, not aspirational)
1. **cuRobo is the only planner.** `plan_to_pose(start_q14, side, goal_pose) ->
   JointTrajectory` (`plan_pose`) and `plan_joint(start, goal) -> JointTrajectory`
   (`plan_cspace`, collision-aware). Trajectory built from cuRobo native timing.
2. **All poses are `spatial.pose.Pose` in the pelvis frame.** `perception/transforms.py` is
   the only file allowed to construct frame conversions (when perception returns).
3. **Dual-arm state is one 14-vector everywhere** (left 7 + right 7, upstream G1_29 arm
   order). Single-arm motion = hold the idle arm at its current FK pose; never slice.
4. **Config-driven:** params + the cuRobo robot config live in `configs/*.yaml`;
   `hand: dex3 → dex1` is a one-line switch.
5. **Primitives are the surface.** New behavior goes behind a primitive, not into callers.
6. **Safety:** prime-to-start, stream at `control_hz`, abort-to-hold past
   `tracking_error_abort_rad`; hand/move primitives block until motion completes; home
   starts and ends at the launch pose.

---

## 3. cuRobo integration specifics
- Config `configs/curobo/g1_dex3_curobo.yml` from the **calibrated mode_16 dex3 URDF** (the
  same the sim now runs — sim == cuRobo == real). Tool frames `left/right_wrist_yaw_link`;
  27 `lock_joints` → arms-only 14-DoF (active joints == the repo arm set).
- `self_collision_ignore` patched: `torso_link` ignores all 6 shoulder links. The
  auto-generated matrix asymmetrically ignored only 3, causing a false "start in collision"
  at the home pose (all-zeros). Verified collision-free after the patch.
- API used: `MotionPlanner(MotionPlannerCfg.create(robot=...))` + `warmup(enable_graph=True)`;
  `plan_pose(GoalToolPose, JointState)` and `plan_cspace(JointState, JointState)` →
  `TrajOptSolverResult` (`.success`, `.get_interpolated_plan()` carrying
  position/velocity/acceleration + `dt`); `compute_kinematics(...).tool_poses.get_link_pose`.
- Goal frame is the **wrist-yaw link** directly; the 5 cm `L_ee`/palm grasp-frame offset is a
  later refinement.
- **Speed** is governed by the cuRobo robot config's joint limits — there is no speed knob in
  `planner.yaml` (only `executor.{control_hz, tracking_error_abort_rad}`).

---

## 4. Verified upstream facts (do not re-derive)

Vendored from `unitreerobotics/xr_teleoperate` (Apache-2.0; keep headers + NOTICE):
- **`G1_29_ArmController`** (`robot_control/robot_arm.py`) — 250 Hz dual-arm streaming.
  Debug mode publishes `rt/lowcmd` and **locks every non-arm joint at its current q** (right
  for the suspended robot); motion mode publishes `rt/arm_sdk`. `ctrl_dual_arm(q14, tau14)`,
  `get_current_dual_arm_q/dq()`, `speed_gradual_max()`, `simulation_mode` (bypasses the
  velocity clip). CRC via `unitree_sdk2py.utils.crc.CRC`.
- **Dex3 / Dex1 controllers** (`robot_control/robot_hand_unitree.py`) — DDS topics
  `rt/dex3/{left,right}/cmd|state` (IDL `unitree_hg HandCmd_/HandState_`) and `rt/dex1/*`
  (`unitree_go MotorCmds_/MotorStates_`). Re-wrapped as **plain threaded** classes that
  publish both hands continuously; the RIS-mode bitfield + command construction are verbatim.
  `HandState_` carries `motor_state.{q,dq,tau_est}` **and `press_sensor_state.pressure`**
  (a per-finger array → reduced to a scalar max) — the grasp-detection signals.
- **`MotionSwitcher`** — `Enter_Debug_Mode()` (required before `rt/lowcmd` on hardware).
- **`unitree_sdk2py`** — pip dependency, never vendored: `ChannelFactoryInitialize`,
  publishers/subscribers, IDL types, CRC, `MotionSwitcherClient`.

cuRobo V2 (`NVlabs/curobo`): ships official G1 + Dex3 configs; builds CUDA extensions from
source (here for sm_120 / torch cu128). `plan_grasp` chains approach → grasp → lift with
finger collisions disabled during final approach — maps onto a future pick primitive.

---

## 5. Roadmap (rough order; not a rigid phase gate)
1. **Tune dex3 grasp presets** (`hands.yaml`) — the right thumb stalls on `power_close`.
2. **Grasp-frame offset** — add the wrist→palm offset so `move` goals are grasp poses.
3. **Pick composite** — `home → move-above → move-down → close_hand → lift`, from primitives.
4. **Rerun logging** — current q / target pose / state, on every primitive.
5. **Perception, cuRobo-native** — sim ground-truth (`rt/sim_state`) or AprilTag → block pose
   in pelvis frame; `transforms.py` reworked.
6. **World model** — add the table (and obstacles) to the cuRobo world for real avoidance.
7. **Hardware bring-up** — see `HARDWARE_TODO.md` (DDS, debug mode, tracking, gravity comp).
8. **Agent layer** — expose the primitives as tools.

---

## 6. Out of scope (do not build)
ROS2 / MoveIt; locomotion or waist control; learned perception (beyond an AprilTag/ground-
truth block source); Inspire/BrainCo hands (keep the registry seam, skip the classes);
on-robot (PC2) deployment of planner code.

## 7. Gotchas (encode these, don't rediscover)
- **Debug mode first** on hardware: `MotionSwitcher.Enter_Debug_Mode()` before any `rt/lowcmd`.
- **Velocity clip vs planner**: the controller's clip is a safety net; if it fires during
  nominal execution, the cuRobo joint limits are wrong — treat as a bug.
- **Sim both-hands gate**: the sim applies hand joints only if both left+right cmds are
  non-empty; our controller publishes both. Hand joints map by name.
- **Hand `verify=False` dwell**: primitives block until the fingers finish (they used to
  return instantly → close→open looked like nothing happened).
- **Domain/interface** live ONLY in `configs/robot.yaml`: real = (0, NIC-to-PC2);
  sim loopback = (1, "lo") + `configs/cyclonedds_loopback.xml` on both ends.
- **Env**: `g1_curobo` via the lane wrapper; no pinocchio, no numpy<2 pin; build per
  `requirements-curobo.txt`.
