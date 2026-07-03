# g1_classical_manip — Plan & Source of Truth (HISTORICAL)

> **HISTORICAL DOCUMENT (frozen 2026-07-02).** This is the design plan that carried the
> repo from v1 to the hardware-validated grasp pipeline. The `api_dev` restructure has
> since landed: the package is now **`g1_primitives`** with a `Robot` facade
> (`g1_primitives.connect`), scripts are regrouped under `scripts/{checks,examples,tools}/`,
> and the AprilTag stack (incl. `06_detect`/`07_pick_place` below) was deleted. The
> current source of truth is **`README.md` + `CLAUDE.md`**; script names below are
> pre-restructure (map: `scripts/README.md`). Kept for design rationale and history.

A **cuRobo-native motion library** for the Unitree G1 (29-DoF, Dex3-1 hands), exposing a
small, growing set of **configurable action + perception primitives** an **LLM agent composes**
into pick-and-place tasks. This repo is the **beta baseline** of that primitive surface, to be
expanded. Composite tasks so far: `scripts/07_pick_place.py` (AprilTag pick+lift) and
`scripts/09_graspgen.py` (pick+lift via a pluggable `GraspSource` — AprilTag A-B reference,
learned GraspGenX 6-DoF grasps over a SAM3-segmented point cloud, or `sim_cloud` a ground-truth
cube cloud for in-sim de-risking).
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
│   ├── curobo_planner.py      # CuroboArmPlanner: plan_to_pose, plan_to_pose_set, plan_joint, fk,
│   │                          #   plan_grasp_set / plan_grasp_set_sweep (native cuRobo plan_grasp)
│   ├── executor.py            # streams JointTrajectory @ control_hz; prime + abort-to-hold; settle
│   ├── collision_world.py     # EsdfMapper: head depth → cuRobo Mapper → ESDF VoxelGrid (gated)
│   └── planner_base.py        # JointPath / JointTrajectory containers (DOF=14)
├── ee/                        # hand_base (ABC), dex3, dex1 — presets + grasp verification
├── robot_control/
│   ├── robot_arm.py           # G1_29_ArmController (vendored: 250 Hz dual-arm streaming)
│   ├── robot_hand_unitree.py  # threaded Dex3/Dex1 controllers (vendored DDS plumbing)
│   └── motion_switcher.py     # Enter/Exit debug mode (hardware)
├── grasp/                     # GraspSource: base · apriltag_source (A-B ref) · graspgenx_source ·
│                              #   sim_cloud_source (sim GT cube) · graspgenx_client (ZMQ :5556) ·
│                              #   tool_transform (grasp→wrist)
├── primitives.py              # home / move / grasp_motion / open_hand / close_hand / detect
│                              #   (move_to_candidates = legacy sequential grasp, --legacy only)
├── factory.py                 # make_robot(): planner + DDS controllers + executor + perception + grasp
├── perception/                # LIVE: transforms · base · apriltag_block · ground_truth · sim_state
│                              #   · depth (deproject→PointCloud) · segment (SAM3 mask seam)
│                              #   · sam3_client (ZMQ :5557) · segment_gui (cv2 mask GUI)
├── image_server/              # HeadCamera (head-cam color + depth frames; camera_rig dormant)
configs/  robot, planner, hands, camera, perception, grasp, curobo/g1_dex3_curobo.yml, cyclonedds_loopback.xml
scripts/  01_check_dds → 12_check_world ladder + 10_graspgen_viz (offline grasp viz) + hand_diag.py
          (12_check_world: inspect the depth-ESDF collision world in isolation — no planning/motion)
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
  27 `lock_joints` → arms-only 14-DoF (active joints == the repo arm set). The hands are locked
  at the **DEPLOYED "open" preset** (`configs/hands.yaml dex3.open`: `thumb_1` at the URDF open
  limit — right +0.7243, left −0.7243; all other hand joints 0), **NOT all-zero**: the Dex3 thumb
  opens fully only at `thumb_1` = the open-direction limit, so the planner's static hand collision
  spheres match the hand we actually plan/approach with. (The self-filter masks the LIVE hand
  separately — see §3a.)
- `self_collision_ignore` patched: (a) `torso_link` ignores all 6 shoulder links — the
  auto-generated matrix asymmetrically ignored only 3, causing a false "start in collision"
  at the home pose; (b) `{side}_hand_thumb_1_link ↔ {side}_wrist_yaw_link` — with the thumb
  locked at its open limit the thumb_1 sphere swings back and clips the large wrist-yaw sphere
  (a conservative-sphere artifact; the two are 3 joints apart and can never collide), which
  otherwise flagged HOME as a start-in-collision. Verified collision-free after the patch.
  Regenerate via `configs/curobo/build_g1_dex3.py` (LOCK + `patch()`; the committed yml is
  hand-patched lock-values + ignore-entries only — **no sphere re-fit**, preserving the
  hardware-validated geometry).
- API used: `MotionPlanner(MotionPlannerCfg.create(robot=...))` + `warmup(enable_graph=True)`;
  `plan_pose(GoalToolPose, JointState)` and `plan_cspace(JointState, JointState)` →
  `TrajOptSolverResult` (`.success`, `.get_interpolated_plan()` carrying
  position/velocity/acceleration + `dt`); `compute_kinematics(...).tool_poses.get_link_pose`.
- Goal frame is the **wrist-yaw link** directly; the palm/grasp-frame offset is applied above the
  planner by the **grasp sources** (`grasp/tool_transform.py`), which return wrist-yaw goals — so
  `move`/`plan_to_pose` stay wrist-yaw. The grasp primitive (`grasp_motion`) plans candidates with
  cuRobo's **native** `plan_grasp` (`plan_grasp_set` / `plan_grasp_set_sweep`: a K-candidate goalset
  → cuRobo picks the feasible grasp → + native approach/grasp/lift segments). Because `plan_grasp`
  offsets EVERY goal tool frame, the grasp planner is a dedicated **single-tool-frame** `MotionPlanner`
  per side (`_grasp_planner`), not the main 3-frame planner; a lone candidate is duplicated into a
  2-row goalset so cuRobo always uses the warmed goalset solver (the `num_goalset=1` path is cold).
  `plan_to_pose_set` (sequential, first-reachable of a ranked list) is now only the `--legacy` baseline.
- **Speed** is governed by the cuRobo robot config's joint limits, but the executor plays the
  plan back at `planner.yaml: executor.time_dilation` (default 0.5 on real, forced 1.0 in sim).
  Real needs the slowdown: the arm controller's velocity clip is measured-relative, so it caps PD
  torque at `~kp·arm_velocity_limit·control_dt` and the arm can't track a full-speed trajectory →
  tracking error diverges → abort. Hardware run config (2026-06-18, MVP validated on the real G1):
  gravity comp ON, `arm_velocity_limit ≥ 12`, `time_dilation 0.5`, operator-set debug mode (no
  MotionSwitcher). The full GraspGenX grasp — INCLUDING the collision-world approach — tracked
  CLEAN on real at the default `time_dilation 0.5` with no tracking-error aborts (real-validated
  2026-06-30), so 0.5 is a confirmed real data point. Root fix (deferred, now LOWER priority): stop
  re-rate-limiting cuRobo's already-feasible trajectory during planned execution. See
  `docs/gravity_comp.md`, `HARDWARE_TODO.md`.
  - The cuRobo config plans to **stock-aggressive dynamics** (`g1_dex3_curobo.yml` cspace:
    `max_acceleration 10`, `max_jerk 500`, scales 1.0). The depth-ESDF collision-world APPROACH
    route (§3a) is dynamic enough that the controller can't track it at full-speed playback even in
    sim — the executor's tracking-error abort fires (`approach: tracking error 0.401 > 0.400 rad`).
    Planning *succeeded*; **execution** aborted. Workaround `09_graspgen --speed 0.5` (confirmed). The
    deferred root fix is the same one: lower `acceleration_scale` so the plan is feasible at
    `time_dilation 1.0`, retuned together with real's dilation. Write-up:
    `docs/trajectory_speed_tracking.md`.

### 3a. Depth-ESDF collision world (built; gated OFF; real-validated end-to-end 2026-06-30)

A new collision-world subsystem so the grasp `plan_grasp` **approach routes around** the
object/table instead of barging through it. Head-camera depth → cuRobo `Mapper` → ESDF
`VoxelGrid` → fed to the grasp planner. **Same path sim + real** and **grasp-source-independent**
(driven by head depth, so it works with `apriltag` / `graspgenx` / `sim_cloud` alike).

- **`motion/collision_world.py` → `EsdfMapper`**: `depth_mm` + intrinsics + `T_pelvis_camera` →
  ESDF VoxelGrid via the cuRobo Mapper (`esdf_from_depth`, `occupied_points`, `empty_esdf_grid`
  helpers). The cuRobo camera frame is OpenCV optical = our deproject convention, so the camera
  pose plugs in directly.
- **Planner (`motion/curobo_planner.py`)** builds a voxel-capable grasp planner when the world is
  enabled: `update_grasp_world(side, depth, K, T_pc, q14, hand_q)` builds the ESDF + self-filters +
  `update_world`; `collision_world_enabled` / `set_collision_world(enabled, cfg)` / `_cw_params()`.
  The ACTIVE hand links are collision-**disabled** during the grasp (the open hand may sit in the
  object ESDF — the grasp is meant to CONTACT).
- **`primitives.grasp_motion`** calls `_update_collision_world(robot, side, q0)` before planning
  (gated, best-effort — a perception hiccup never blocks the grasp).
- **Config** `configs/planner.yaml → grasp.collision_world`: `enabled: false` (**default OFF**),
  `grid_center [0.4,0,0.2]`, `extent_m [1.2,1.2,1.0]`, `esdf_voxel_size 0.01`, `tsdf_voxel_size
  0.005`, `depth_min_m 0.1`, `depth_max_m 2.0`, `self_filter: true`, `robot_mask_margin 0.02`.
  Opt in via the flag or `09_graspgen.py --collision-world`. Needs a head depth stream
  (`08_check_depth` green). **SIM depth source**: the (separate, user-managed) `unitree_sim_isaaclab`
  repo publishes head depth on a dedicated ZMQ PUB **:55556** (raw float32 mm); robot side subscribes
  via `image_server/image_client.py` (`get_depth_frame()`), `configs/camera_sim.yaml: depth_port 55556`.

**Robot self-filter — LIVE pose tracking.** The head camera sees the robot's own arm/hands; unfiltered
they fuse into the ESDF and `plan_grasp` starts inside a copy of itself ("Goalset planning returned
None"). cuRobo `RobotSegmenter` zeros depth pixels within `robot_mask_margin` of the robot's collision
spheres **at the CURRENT measured config** — arm q from `robot.arm.get_current_dual_arm_q()` AND fingers
from `robot.hand.get_q(side)`. Because the arms-only PLANNING model locks the hand open, a SEPARATE
hand-active segmenter kinematics is built (hand joints unlocked → 28 active DoF = 14 arm + 14 fingers)
so a bent thumb is masked at its real pose. Notes: `ops_dtype=float32` (cuRobo's default bfloat16 trips
its own float32 check on the cdist path); hand q is mapped **BY NAME** (the Dex3 `get_q` RIGHT order is
thumb,index,middle but cuRobo orders the right hand thumb,middle,index — a raw copy would swap them).
`robot.hand` is a `Dex3Hand` wrapper exposing `get_q(side)` (controller passthrough).

**Inspect it in isolation: `scripts/12_check_world.py`** — the 12th bring-up rung. NO planning/motion:
(1) is the geometry placed right (ESDF occupied voxels vs the raw deproject cloud)? (2) is the robot's
own arm removed (self-filter working)? Connects DDS read-only (`home_on_connect=False` → nothing moves)
and self-filters at the LIVE arm+hand q (matches the grasp path); `--no-dds` filters at home offline.
Flags: `--visualize` (viser ESDF/cloud/wrist overlay :8080), `--probe X Y Z` (is the object in the ESDF
or erased by the self-filter?), `--margin` (sweep the filter margin), `--esdf-voxel`, `--no-self-filter`.

**Status: built; gated OFF by default; real-validated end-to-end 2026-06-30** — the obstacle-aware
approach + the LIVE robot self-filter (arm + finger hand-tracking) ran IN a full GraspGenX grasp on
the real G1 with `09_graspgen --collision-world`, and `12_check_world` inspected the world on the
real ZED depth. Still OFF by default; opt in with `--collision-world`.

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
finger collisions disabled during final approach — **now wired** as the `grasp_motion` primitive
(via `plan_grasp_set_sweep`); it is the default grasp path (`--legacy` restores the old sequential).

---

## 5. Roadmap (rough order; not a rigid phase gate)
1. **Tune dex3 grasp presets** (`hands.yaml`) — right-thumb `power_close` stall fixed; make
   `pinch` per-hand and tune the `verify` thresholds.
2. ~~**Grasp-frame offset**~~ — **DONE**: the wrist→palm offset lives in `grasp/tool_transform.py`;
   grasp sources return wrist-yaw goals. The GraspGenX `wristyaw_grasp_rpy` + `palm_offset_xyz` are
   **DERIVED from kinematics** (`scripts/derive_graspgenx_tool_transform.py`) and **sim-validated
   2026-06-28** (`[π/2,0,π]`, `[0.1142,−0.0286,0]`) and now **HARDWARE-confirmed 2026-06-30** — the
   real GraspGenX grasp picked + lifted the object, validating the transform incl. the closing-roll
   sign (the last "needs one gated HW grasp" item). The planner's DEPLOYED-open thumb lock
   (`thumb_1` ±0.7243 + the `thumb_1_link↔wrist_yaw_link` ignore patch) also executed cleanly from
   the launch pose on real — no start-in-collision.
3. ~~**Pick composite**~~ — **DONE**: `07_pick_place` (AprilTag) + `09_graspgen` (GraspSource),
   the latter on cuRobo's native `plan_grasp` (goalset pick + approach/grasp/lift).
4. ~~**Grasp sources + perception**~~ — **DONE (offline)**: `GraspSource` seam (`apriltag` |
   `graspgenx` | `sim_cloud`); depth→`PointCloud` (`perception/depth`); SAM3 segmentation
   (`perception/segment`, `:5557`) → GraspGenX (`:5556`, **protocol v2**: `planner` topdown/graspmoe/
   diffusion, obb/diff `branch_tags`) → 6-DoF grasps. **Real grasp run DONE 2026-06-30** — live ZED
   depth → SAM3 → GraspGenX (`topdown`) → tool transform → cuRobo `plan_grasp` picked + lifted on real.
5. **Rerun logging** — current q / target pose / state, on every primitive.
6. ~~**World model**~~ — **DONE (built; real-validated end-to-end 2026-06-30; gated OFF)**: the
   depth-ESDF collision world (head depth → cuRobo Mapper → ESDF, with a LIVE-pose robot self-filter)
   so the grasp approach routes around the object/table — see §3a. Ran IN a full real GraspGenX grasp
   via `09_graspgen --collision-world`. `enabled: false` by default.
7. **Hardware bring-up** — see `HARDWARE_TODO.md` (DDS, debug mode, tracking, gravity comp, the
   GraspGenX/SAM3 grasp run + `wristyaw_grasp_rpy` calibration).
8. **Agent layer** — expose the primitives + grasp source as tools.

---

## 6. Out of scope (do not build)
ROS2 / MoveIt; locomotion or waist control; learned perception (beyond an AprilTag/ground-
truth block source); Inspire/BrainCo hands (keep the registry seam, skip the classes);
on-robot (PC2) deployment of planner code.

## 7. Gotchas (encode these, don't rediscover)
- **Debug mode first** on hardware (required before any `rt/lowcmd`), but the **OPERATOR sets it via
  the physical remote** on this rig — `make_robot` does NOT call `MotionSwitcher` (releasing it on an
  already-debug robot drops it back out of low-level control → arms don't move). Opt into the SDK
  release only with `enter_debug_mode=True`. See the CLAUDE.md debug-mode gotcha.
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
