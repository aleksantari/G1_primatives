# CLAUDE.md — conventions digest for g1_classical_manip

Read `G1_CLASSICAL_MANIP_PLAN.md` (source of truth) and `HARDWARE_TODO.md` first.

## What this is
A **cuRobo-native motion library** exposing a small set of **action primitives** (the
eventual LLM-agent tool surface) for the Unitree G1 (29-DoF, Dex3 hands), running off-board
on an RTX 5090 over CycloneDDS. The robot is suspended on a back-plate mount: only the 14
arm joints + 2×7 hand joints are ever commanded. The original Cartesian + pinocchio-IK +
FSM pick-place pipeline was an early experiment and has been **removed** — see the PLAN's
"Pivot" note. cuRobo is now the single source of kinematics and planning.

## Environment
- Conda env **`g1_curobo`** (Python 3.11, numpy 2, torch 2.9.1+cu128 for the RTX 5090 /
  sm_120, **cuRobo V2 built from source**, cyclonedds, unitree_sdk2py). Always run via
  `bash -ic 'use_conda g1_curobo && <cmd>'` (lane wrapper; sets `PYTHONNOUSERSITE=1`
  isolation — `conda run` does NOT and will leak `~/.local`).
- **No pinocchio.** All kinematics (FK/IK) come from cuRobo. The old `numpy<2` pin existed
  only for pinocchio and is gone. The legacy `g1_classical_manip` env (py3.10) is dead.
- Build/install is not plain `pip install` — see the header of `requirements-curobo.txt`
  (torch cu128, cuRobo source build with `TORCH_CUDA_ARCH_LIST="12.0"`, cyclonedds vs
  `/opt/cyclonedds`, editable unitree_sdk2py).

## Architecture rules (enforced)
1. **cuRobo is the only planner.** `CuroboArmPlanner.plan_to_pose(start_q14, side,
   goal_pose) -> JointTrajectory` (cuRobo `plan_pose`) and `plan_joint(start, goal) ->
   JointTrajectory` (cuRobo `plan_cspace`, collision-aware). cuRobo emits a
   time-parameterized, dynamically-feasible trajectory **directly — there is no separate
   retimer** (removed). The executor owns the only handle to `G1_29_ArmController`;
   planning never streams.
2. **All poses are `spatial.pose.Pose` (numpy) in the pelvis frame** — the `pin.SE3`
   replacement (wxyz quats, convention-faithful to ~1e-15). `perception/transforms.py`
   (`Frames`, cuRobo-FK-backed) is the only file allowed to construct frame conversions.
3. **Dual-arm state is one 14-vector everywhere** (left 7 + right 7, upstream G1_29 arm
   order: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw). Single-arm motion = hold
   the idle arm at its current FK pose as the other tool goal; never slice the controller.
4. **Config-driven:** robot/planner/hand params + the cuRobo robot config live in
   `configs/*.yaml`. `hand: dex3 → dex1` is a one-line switch.
5. **Primitives are the surface** (`primitives.py`): `home`, `move`, `open_hand`,
   `close_hand`, `grasp_motion` (native cuRobo `plan_grasp`: goalset pick + approach/grasp/lift),
   each returning `Result(ok, info)`. Tasks are composed from these. Keep new behavior out of
   here unless it's a genuine new primitive.
6. **Safety:** the executor primes to the trajectory start, streams at `control_hz`, and
   **aborts-to-hold** past `tracking_error_abort_rad`; hand/move primitives block until the
   motion completes; `home` starts and ends at the launch pose. **Launch home:**
   `make_robot(connect_dds=True)` first drives the arms to `home_q14_deg` via DIRECT,
   un-planned PD (`Executor.go_home_direct`, velocity-capped, NOT collision-checked) and
   waits for convergence — so the robot starts collision-free and the planned `home`/`move`
   plan from a good start. This exists because cuRobo's collision-aware `plan_cspace` refuses
   to plan *out of* a pose it flags as a self-collision start (e.g. the arms folded at
   power-on). Disable with `home_on_connect=False`.

## Key facts (verified, don't re-derive)
- cuRobo config: `configs/curobo/g1_dex3_curobo.yml`, built from the **calibrated mode_16
  dex3 URDF** (`assets/g1/g1_29dof_mode_16_dex3.urdf`) via cuRobo `RobotBuilder`; tool frames
  `left/right_wrist_yaw_link`; 27 `lock_joints` → arms-only 14-DoF. The hands are locked at the
  **DEPLOYED "open" preset**, NOT all-zero: `thumb_1` at the URDF open limit (right +0.7243, left
  −0.7243; `hands.yaml` dex3.open), all other hand joints 0 — so the planner's static hand
  collision spheres match the hand we actually approach with (0 leaves the thumb half-abducted).
  `self_collision_ignore` patched (`build_g1_dex3.patch()`) for two conservative-sphere artifacts
  that else falsely flag start-in-collision at home: `torso_link` ↔ all 6 shoulder links (the
  auto-matrix asymmetrically missed 3), and `{side}_hand_thumb_1_link` ↔ `{side}_wrist_yaw_link`
  (the open thumb's sphere clips the wrist-yaw sphere; the thumb is 3 joints from the wrist, can't
  reach it). **Regenerate with `configs/curobo/build_g1_dex3.py`** (the provenance/recipe; not run
  at import — only when the URDF or the arms-only reduction changes; the committed config is
  hardware-validated; lock-value + ignore tweaks are hand-patched in place, no sphere re-fit).
- **Home = the sim launch pose: all arm joints 0 = forearms forward (elbows bent ~90°)**,
  collision-free. `configs/robot.yaml: home_q14_deg`.
- cuRobo native plan ≈ 1.1 rad/s, `dt` 0.025 s. Trajectory SPEED is governed by the cuRobo
  robot config's joint limits, but the executor **plays it back at `executor.time_dilation`**
  (planner.yaml; default 0.5 on real, forced 1.0 in sim). Real needs the slowdown: the arm
  controller's velocity clip (`clip_arm_q_target`) is measured-relative, so it caps PD torque at
  `~kp·arm_velocity_limit·control_dt` and the arm can't track a full-speed trajectory → tracking
  error diverges → abort. So on real: `gravity_comp` ON (holds the arm), `arm_velocity_limit ≥ 12`
  (torque headroom), `time_dilation 0.5` (tracks). Sim bypasses the clip entirely, so it runs full
  speed with no gravity comp. The root fix (deferred) is to lower the cuRobo cspace
  `acceleration_scale` so the plan is feasible at `time_dilation 1.0` (then retune sim/real
  together) instead of re-rate-limiting cuRobo's already-feasible trajectory at playback — the
  collision-world approach is dynamic enough to trip this even in sim. See
  `docs/trajectory_speed_tracking.md`, `docs/gravity_comp.md`, `HARDWARE_TODO.md`.
- Goal frame is the **wrist-yaw link** directly in the primitives (keeps them composable).
  The grasp→wrist transform (`grasp/tool_transform.py: build_T_wristyaw_grasp`) is applied by the
  **grasp sources** (`grasp/`). **GraspGenX's transform is DERIVED from kinematics, NOT the
  AprilTag offset reused** (that `[0.1192,−0.0346,0]` was reverse-engineered for a top-down grasp
  and is meaningless for GraspGenX's base-anchored frame; the two paths are now decoupled).
  `scripts/derive_graspgenx_tool_transform.py` (provenance) registers GraspGenX's grasp convention
  (origin=gripper base, +Z approach, +X=thumb-vs-fingers closing, fingertips at +Z=0.07) against
  the Dex3 URDF via `ee/hand_kinematics.py` (hand FK): **rotation** = the fixed axis map (approach
  +Z→wrist +Y, closing +X→wrist −X, spread +Y→wrist +Z) = `wristyaw_grasp_rpy: [π/2,0,π]`
  (SIM-VALIDATED 2026-06-28). The earlier `[π/2,0,π/2]` (approach→wrist +X) put the palm 90° off —
  the Dex3 came down thumb-along-the-top instead of palm-down; a +90° yaw about wrist +Z fixes it
  (the even-older `[0,π/2,0]` was the AprilTag-era guess). NOTE: the Dex3's real thumb-vs-fingers
  opposition is DIAGONAL in the wrist XY-plane (FK ~[0.66,−0.75,0]); this clean-axis map is the
  sim-matched approximation that makes the palm face down — a no-op closing-spin for a symmetric
  cube, revisit per-object. **translation** = our power_close contact midpoint (FK) minus the 0.07
  depth along approach (now wrist +Y) = `graspgenx.palm_offset_xyz: [0.1142,−0.0286,0]` (so fingers
  land ON the object, not 7 cm short). Config stores the RIGHT hand; LEFT is mirrored across the
  wrist Y-plane in code. `09 --source sim_cloud` FK-verifies our fingers straddle the GT cube (the
  check the viser gripper-mesh overlay can't do). Still verify the closing-roll sign on one gated
  hardware grasp. (`07_pick_place`/`apriltag` keep their own `palm_offset` — the A-B reference —
  untouched.)
- **Grasp pipeline** (`grasp/` + `perception/{depth,segment,sam3_client,segment_gui}.py`,
  `spatial/pointcloud.py`): `robot.grasp_source` is a `GraspSource` (`grasp.yaml: grasp_source` =
  `apriltag` A-B ref | `graspgenx` | `sim_cloud`) returning ranked wrist-yaw `GraspCandidate`s;
  the `grasp_motion` primitive plans them with cuRobo's **native** `plan_grasp` (planner
  `plan_grasp_set` / `plan_grasp_set_sweep`: a K-candidate **goalset** — cuRobo picks the feasible
  grasp — + native approach/grasp/lift segments). `plan_grasp` offsets EVERY goal tool frame, so the
  grasp planner is a dedicated **single-tool-frame** `MotionPlanner` per side (`_grasp_planner`); the
  main 3-frame planner can't be used. A lone candidate is duplicated into a 2-row goalset (cuRobo
  `warmup` only primes the goalset solver, never the cold `num_goalset=1` path). `move_to_candidates`
  / `plan_to_pose_set` (sequential, first-reachable) is now only the `09 --legacy` A-B baseline.
  GraspGenX path: head depth → `deproject_depth` (mask-gated, optional `rgb=` →
  per-point color, viz-only) → pelvis `PointCloud` → **SAM3** mask (ZMQ `:5557`, 2D mask applied
  PRE-deproject; `Segmenter.mask(rgb)` seam, interactive cv2 GUI) → **GraspGenX** ZMQ (`:5556`,
  **protocol v2**: `infer` → `(grasps, conf, branch_tags)`; kwargs `planner`
  {diffusion|graspmoe|**topdown**}, `obb_density`, `skip_obb_rule`; `branch_tags[i]` ∈ {obb,diff}.
  `grasp.yaml` defaults to `planner: topdown` for grab-from-above) → 6-DoF grasps → tool transform.
  Viz colors obb (amber) vs diff (purple); `09` prints the obb/diff split + a top-down(approach≈−Z)
  count. `PointCloud` carries optional `colors` (N,3) parallel to `points`
  (transform-carried, voxel-averaged) — purely for viz; consumers send `.points` (N,3), so color
  never reaches GraspGenX. `sim_cloud` (`sim_cloud_source.py`): a SIM-ONLY GT cube cloud from the
  `rt/sim_state` block pose (no camera/depth/SAM3) → same GraspGenX + tool transform; the in-sim
  de-risk path for `wristyaw_grasp_rpy` + 6-DoF execution. Both ZMQ clients are thin standalone
  shims that do NOT import their service package (the multi-GB import trap); deps
  `msgpack`/`msgpack-numpy`. Scripts: `09_graspgen` (`--source`/`--segment`/`--visualize`),
  `10_segment` (SAM3 dev tool; `--image` static, `--save` writes a COLORED `.ply`), `10_graspgen_viz`
  (offline cloud → GraspGenX → viser `:8080`). All offline-tested; the real grasp run is pending hardware.
- **Depth-ESDF collision world** (`motion/collision_world.py`; gated OFF by default —
  `planner.yaml grasp.collision_world.enabled` or `09_graspgen --collision-world`): head depth →
  cuRobo `Mapper` → ESDF `VoxelGrid` (`EsdfMapper`) → the grasp planner, so the native `plan_grasp`
  **approach** routes around the object/table. SAME path sim + real, **source-independent** (head
  depth, any grasp source). The grasp planner is built voxel-capable (`update_grasp_world`); the
  active hand links are collision-disabled during the grasp (the open hand may sit in the object
  ESDF — the grasp is meant to CONTACT). A cuRobo `RobotSegmenter` **self-filter** zeros the robot's
  own pixels from the depth at its **LIVE measured config** — arm q + live finger q via a dedicated
  **hand-active** segmenter kinematics (the arms-only planning model locks fingers open, so a bent
  thumb would otherwise survive into the world); without it the arm/hand fuses in and the grasp
  starts inside a copy of itself ("Goalset planning returned None"). Hand q mapped BY NAME (the Dex3
  `get_q` right-hand order swaps index/middle vs left). `12_check_world` inspects the world in
  isolation (ESDF vs raw cloud + self-view probe). Sim depth = Isaac `front_camera` over ZMQ
  `:55556`. Sim-validated; real grasp run pending. See `docs/depth_integration_handoff.md`,
  `docs/trajectory_speed_tracking.md`.
- Hand control (`robot_control/robot_hand_unitree.py`): threaded `Dex3Controller` /
  `Dex1Controller`, **publishes both hands continuously**; exposes `q/dq/tau/press` (grasp
  signals). Presets + verification in `ee/dex3.py` + `configs/hands.yaml`. Presets are being
  tuned: the right-thumb `power_close` stall is **fixed**; `pinch` (not yet per-hand) and the
  `verify` thresholds still need tuning.
- Arm controller: debug mode = `rt/lowcmd` + locks non-arm joints at current q
  (suspended-robot correct); `simulation_mode` bypasses the velocity clip.
- DDS domain/interface live ONLY in `configs/robot.yaml`. Real = (0, NIC-to-PC2);
  unitree_sim_isaaclab loopback = (1, "lo") + `configs/cyclonedds_loopback.xml` on both
  ends. See SIM_NOTES.md.

## Gotchas
- **Debug mode is set by the OPERATOR via the physical remote**, not by us. Low-level
  `rt/lowcmd` control needs the robot in debug mode; on this rig the operator enables it on
  the suspended robot with the controller before launch (the proven `unitree_lerobot` flow
  does the same and never calls `MotionSwitcher`). `make_robot` therefore does **not**
  auto-enter debug mode — calling `MotionSwitcher.ReleaseMode()` on a robot already in
  physically-set debug mode drops it back OUT of low-level control, after which `rt/lowcmd`
  is ignored and the arms don't move (observed: launch home residual ~87° = zero motion).
  Opt in to an SDK release with `enter_debug_mode=True` (e.g. no remote available).
- The sim now runs the **calibrated mode_16 dex3 USD** (sim == cuRobo == real kinematics)
  and maps hand-command joints **by name**. Its hand-apply path has a **both-hands gate**
  (applies hand joints only if both left+right cmds are non-empty) — our controller
  publishes both, so it's satisfied.
- Hand primitives with `verify=False` now **dwell** until the motion completes (they used
  to return instantly, so close→open fired back-to-back and looked like nothing happened).
- **Live now (not dormant):** the perception stack
  (`perception/{transforms,base,apriltag_block,ground_truth}.py`) and
  `image_server/image_client.py` (head color **+ depth** — `HeadCamera.get_depth_frame()`: the
  real ZED's raw-float32 720×1280 mm stream, or the **sim** Isaac `front_camera` depth over the
  zmq backend `:55556`) + the **grasp pipeline** (`grasp/`,
  `perception/{depth,segment,sam3_client,segment_gui}.py`, `spatial/pointcloud.py`, `viz/` +
  `assets/grippers/`) + the **depth-ESDF collision world** (`motion/collision_world.py`).
  Current scripts: the numbered bring-up ladder (`01_check_dds … 12_check_world`:
  check_dds → check_image → hands → move → mvp_demo → detect → pick_place → check_depth → graspgen →
  segment → graspgen_viz → capture_frame → check_world; each `--target sim|real`, shared
  `scripts/_rig.py`) + `hand_diag.py`. **GraspGenX (`:5556`) / SAM3 (`:5557`) ZMQ servers run SEPARATELY** (own repos/envs,
  launch cmds in README "Grasp pipeline"); `09 --visualize` / `10_graspgen_viz` render cloud + ranked
  grasps in **viser** (`:8080`). Offline demo (no robot): `11_capture_frame` (now also records the
  arm+hand q) → `10_segment --frame … --save` → `10_graspgen_viz --pcd` for grasps, or
  `12_check_world --frame …` to inspect the depth-ESDF collision world (self-filters at the
  capture's recorded q, else home).
- **Dormant on disk** (unimported, kept for later): `image_server/camera_rig.py` +
  `configs/cameras.yaml` (multi-camera). (`robot_control/motion_switcher.py` is wired but
  **opt-in** — `make_robot` lazily imports it only when `enter_debug_mode=True`; default off,
  see the debug-mode gotcha above.)
- **Deleted** (recoverable from git): `robot_control/robot_arm_ik.py`,
  `motion/cartesian_planner.py`, `motion/retimer.py`, `tasks/*`, `utils/*`
  (rerun/episode/filter), the legacy v1 scripts (`run_*`, `0X_*`, `offline_*`,
  `05_reach_check*`), and their `configs/task_*.yaml`.
