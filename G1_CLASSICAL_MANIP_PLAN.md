# g1_classical_manip — Plan & Source of Truth

A **cuRobo-native motion library** for the Unitree G1 (29-DoF, Dex3-1 hands), exposing a
small, growing set of **configurable action + perception primitives** an **LLM agent composes**
into pick-and-place tasks. This repo is the **beta baseline** of that primitive surface, to be
expanded. Composite tasks so far: `scripts/07_pick_place.py` (AprilTag pick+lift) and
`scripts/09_graspgen.py` (pick+lift via a pluggable `GraspSource` — AprilTag A-B reference or
learned GraspGenX 6-DoF grasps over a SAM3-segmented point cloud).
It runs **off-board** on an RTX 5090 workstation talking to the
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

**MVP — validated on the REAL G1 (2026-06-18)** and in `unitree_sim_isaaclab`: `home → move →
close_hand → open_hand → home` runs end-to-end (sim: zero executor aborts; real: with gravity
comp + `time_dilation 0.5`, see §3 and `HARDWARE_TODO.md`). `home`/`move` are collision-aware via
cuRobo. `detect()` is sim-validated only (real needs the ZED calibration).

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
├── spatial/                   # pose.py (SE(3) Pose, pin.SE3 replacement) · pointcloud.py (PointCloud)
├── motion/
│   ├── curobo_planner.py      # CuroboArmPlanner: plan_to_pose, plan_to_pose_set, plan_joint, fk
│   ├── executor.py            # streams JointTrajectory @ control_hz; prime + abort-to-hold; settle
│   └── planner_base.py        # JointPath / JointTrajectory containers (DOF=14)
├── ee/                        # hand_base (ABC), dex3, dex1 — presets + grasp verification
├── robot_control/
│   ├── robot_arm.py           # G1_29_ArmController (vendored: 250 Hz dual-arm streaming)
│   ├── robot_hand_unitree.py  # threaded Dex3/Dex1 controllers (vendored DDS plumbing)
│   └── motion_switcher.py     # Enter/Exit debug mode (hardware)
├── grasp/                     # GraspSource: base · apriltag_source (A-B ref) · graspgenx_source
│                              #   · graspgenx_client (ZMQ :5556) · tool_transform (grasp→wrist)
├── primitives.py              # home / move / move_to_candidates / open_hand / close_hand / detect
├── factory.py                 # make_robot(): planner + DDS controllers + executor + perception + grasp
├── perception/                # LIVE: transforms · base · apriltag_block · ground_truth · sim_state
│                              #   · depth (deproject→PointCloud) · segment (SAM3 mask seam)
│                              #   · sam3_client (ZMQ :5557) · segment_gui (cv2 mask GUI)
├── image_server/              # HeadCamera (head-cam color + depth frames; camera_rig dormant)
configs/  robot, planner, hands, camera, perception, grasp, curobo/g1_dex3_curobo.yml, cyclonedds_loopback.xml
scripts/  01_check_dds → 11_capture_frame ladder + 10_graspgen_viz (offline grasp viz) + hand_diag.py
tests/    test_pose/grasp/detect + pointcloud/depth_deproject/tool_transform/graspgenx_client/
          grasp_source/sam3_client/segment   (pure-math; no robot)
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
- Goal frame is the **wrist-yaw link** directly; the palm/grasp-frame offset is applied above the
  planner by the **grasp sources** (`grasp/tool_transform.py`), which return wrist-yaw goals — so
  `move`/`plan_to_pose` stay wrist-yaw. `plan_to_pose_set` plans the first reachable of a ranked
  candidate list (native cuRobo goalset deferred).
- **Speed** is governed by the cuRobo robot config's joint limits, but the executor plays the
  plan back at `planner.yaml: executor.time_dilation` (default 0.5 on real, forced 1.0 in sim).
  Real needs the slowdown: the arm controller's velocity clip is measured-relative, so it caps PD
  torque at `~kp·arm_velocity_limit·control_dt` and the arm can't track a full-speed trajectory →
  tracking error diverges → abort. Hardware run config (2026-06-18, MVP validated on the real G1):
  gravity comp ON, `arm_velocity_limit ≥ 12`, `time_dilation 0.5`, operator-set debug mode (no
  MotionSwitcher). Root fix (deferred): stop re-rate-limiting cuRobo's already-feasible trajectory
  during planned execution. See `docs/gravity_comp.md`, `HARDWARE_TODO.md`.

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
1. **Tune dex3 grasp presets** (`hands.yaml`) — right-thumb `power_close` stall fixed; make
   `pinch` per-hand and tune the `verify` thresholds.
2. ~~**Grasp-frame offset**~~ — **DONE**: the wrist→palm offset lives in `grasp/tool_transform.py`;
   grasp sources return wrist-yaw goals. (`wristyaw_grasp_rpy` rotation seed still EMPIRICAL.)
3. ~~**Pick composite**~~ — **DONE**: `07_pick_place` (AprilTag) + `09_graspgen` (GraspSource).
4. ~~**Grasp sources + perception**~~ — **DONE (offline)**: `GraspSource` seam (`apriltag` |
   `graspgenx`); depth→`PointCloud` (`perception/depth`); SAM3 segmentation (`perception/segment`,
   `:5557`) → GraspGenX (`:5556`) → 6-DoF grasps. Real grasp run pending hardware.
5. **Rerun logging** — current q / target pose / state, on every primitive.
6. **World model** — add the table (and obstacles) to the cuRobo world (depth → ESDF a natural fit).
7. **Hardware bring-up** — see `HARDWARE_TODO.md` (DDS, debug mode, tracking, gravity comp, the
   GraspGenX/SAM3 grasp run + `wristyaw_grasp_rpy` calibration).
8. **Agent layer** — expose the primitives + grasp source as tools.

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
