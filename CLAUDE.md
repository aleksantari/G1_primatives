# CLAUDE.md — conventions digest for g1_primitives

Read `README.md` (the SDK front door) and `HARDWARE_TODO.md` first.
(`G1_CLASSICAL_MANIP_PLAN.md` is the pre-restructure design doc, kept as history.)

## What this is
A **cuRobo-native motion + perception primitive library** (package **`g1_primitives`**;
the repo dir is still `G1_classical_manip`) for the Unitree G1 (29-DoF, Dex3 hands),
exposing the tool surface an LLM agent composes for pick-and-place. Runs off-board on an
RTX 5090 over CycloneDDS; the robot is suspended on a back-plate mount — only the 14 arm
joints + 2×7 hand joints are ever commanded. cuRobo is the single source of kinematics
and planning (the v1 pinocchio/FSM pipeline is long deleted).

## Environment
- Conda env **`g1_curobo`** (Python 3.11, numpy 2, torch 2.9.1+cu128 for the RTX 5090 /
  sm_120, **cuRobo V2 built from source**, cyclonedds, unitree_sdk2py). Always run via
  `bash -ic 'use_conda g1_curobo && <cmd>'` (lane wrapper; sets `PYTHONNOUSERSITE=1`
  isolation — `conda run` does NOT and will leak `~/.local`).
- **No pinocchio.** All kinematics (FK/IK) come from cuRobo.
- Build/install is not plain `pip install` — see the header of `requirements-curobo.txt`
  (torch cu128, cuRobo source build with `TORCH_CUDA_ARCH_LIST="12.0"`, cyclonedds vs
  `/opt/cyclonedds`, editable unitree_sdk2py). The package is `pip install -e .` as
  `g1_primitives`.
- Suite: `bash -ic 'use_conda g1_curobo && python -m pytest tests/ -q'` — GPU-free by
  design (see the test pattern below); keep it that way.

## Architecture rules (enforced)
1. **Facade-first.** The public surface is `g1_primitives.connect(target) -> Robot`
   (api/robot.py) + the curated `__init__` exports. Scripts/examples/agent hosts go
   through the facade; **never mutate `robot.cfg` and rebuild components by hand — use
   the `set_*` methods** (they keep the config dict and the rebuilt component in
   lockstep). Verb implementations live in `api/primitives.py`; keep new behavior out of
   there unless it's a genuine new primitive.
2. **cuRobo is the only planner** (`motion/planner.CuroboArmPlanner`): `plan_to_pose`
   (cuRobo `plan_pose`), `plan_joint` (`plan_cspace`, collision-aware),
   `plan_grasp_set(_sweep)` (native `plan_grasp`). cuRobo emits a time-parameterized,
   dynamically-feasible trajectory directly — **no separate retimer**. The executor owns
   the only handle to `G1_29_ArmController`; planning never streams. The planner keeps
   ALL planner state; `motion/diagnostics.py` is its stateless friend module
   (`diagnose/world_check/explain_failure(planner, ...)`).
3. **All poses are `spatial.pose.Pose` (numpy) in the pelvis frame** (wxyz quats,
   convention-faithful to ~1e-15). `perception/frames.py` (`Frames`, with the FK callable
   INJECTED — perception never imports motion) is the only file allowed to construct
   frame conversions.
4. **Dual-arm state is one 14-vector everywhere** (left 7 + right 7, G1_29 arm order:
   shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw). Single-arm motion = hold the
   idle arm at its current FK pose; never slice the controller.
5. **Config-driven, one selector shape:** every pluggable seam is a `source:` key naming
   a sibling block (grasp.yaml `source: graspgenx|sim_cloud`; perception.yaml
   `source: sim_state`). `config.load_configs` is the only yaml reader and raises
   actionable `ConfigMigrationError`s on pre-restructure keys. `hand: dex3 → dex1` is a
   one-line switch (dex1 = future-work seam, no curobo yml yet).
6. **Layering:** `spatial` ← `perception`/`grasp`/`motion`/`ee`/`hardware` ← `api`.
   Importing `g1_primitives` must pull NO torch/CUDA/DDS (heavy imports live behind
   `connect()` and the lazy `api/_builders.py`) — `tests/test_import_hygiene.py` enforces
   it; extend its CLEAN_MODULES when adding import-light modules.
7. **Safety:** the executor primes to the trajectory start, streams at `control_hz`, and
   **aborts-to-hold** past `tracking_error_abort_rad`; hand/move primitives block until
   the motion completes. **Launch home:** `connect()` first drives the arms to
   `home_q14_deg` via DIRECT, un-planned PD (`Executor.go_home_direct`, velocity-capped,
   NOT collision-checked) and waits for convergence — cuRobo refuses to plan *out of* a
   pose it flags as a self-collision start (e.g. arms folded at power-on). Disable with
   `home_on_connect=False`.

## Test pattern (GPU-free suite)
Planner tests use `CuroboArmPlanner.__new__` + a `FakeMP` (no CUDA); facade tests use
`Robot(cfg, fake_planner)` + monkeypatched `_builders`; `ensure_debug_mode` takes an
injectable `client_factory` (no DDS). `test_import_hygiene.py` subprocess-checks that the
base/perception layers and the `motion.planner` MODULE import stay torch-free — that
module-scope cleanliness is what the whole fake pattern depends on.

## Key facts (verified, don't re-derive)
- **Debug mode (real):** low-level `rt/lowcmd` control needs the robot in debug mode.
  `connect("real")` runs `hardware.ensure_debug_mode()` by default — **CHECK-FIRST**:
  CheckMode with an empty mode name means the operator already set debug via the physical
  remote, and it returns UNTOUCHED without ever calling ReleaseMode (blind-releasing an
  operator-set debug mode drops the robot OUT of low-level control; observed as launch
  home residual ~87° = zero motion). Only an active named ai/loco mode is released
  (bounded ~10 s loop). `enter_debug_mode=False` skips even the check. The vendored
  `MotionSwitcher.Enter_Debug_Mode` while-loop is exactly the blind-release footgun —
  kept verbatim for provenance, never call it.
- cuRobo config: `configs/curobo/g1_dex3_curobo.yml`, built from the **calibrated mode_16
  dex3 URDF** (`assets/g1/g1_29dof_mode_16_dex3.urdf`); tool frames
  `left/right_wrist_yaw_link`; 27 `lock_joints` → arms-only 14-DoF. Hands locked at the
  **DEPLOYED "open" preset**, NOT all-zero (`thumb_1` at the URDF open limit — right
  +0.7243, left −0.7243; all other hand joints 0) so the planner's static hand spheres
  match the hand we approach with. `self_collision_ignore` hand-patched for two
  conservative-sphere artifacts (torso↔shoulder links; open thumb_1↔wrist_yaw).
  **Regenerate with `configs/curobo/build_g1_dex3.py`** only when the URDF or the
  arms-only reduction changes; the committed config is hardware-validated.
- **Home = the sim launch pose: all arm joints 0 = forearms forward (elbows ~90°)**,
  collision-free. `configs/robot.yaml: home_q14_deg`.
- cuRobo native plan ≈ 1.1 rad/s, `dt` 0.025 s. The executor plays it back at
  `executor.time_dilation` (planner.yaml; default 0.5 on real, forced 1.0 in sim). Real
  needs the slowdown: the arm controller's velocity clip caps PD torque at
  `~kp·arm_velocity_limit·control_dt`, so a full-speed trajectory diverges → abort. On
  real: `gravity_comp` ON, `arm_velocity_limit ≥ 12`, `time_dilation 0.5`. Root fix
  (deferred, lower priority): lower cuRobo cspace `acceleration_scale` instead of
  re-rate-limiting at playback. See `docs/trajectory_speed_tracking.md`,
  `docs/gravity_comp.md`.
- Goal frame is the **wrist-yaw link** directly in the primitives. The grasp→wrist
  transform (`grasp/tool_transform.py: build_T_wristyaw_grasp`) is applied by the grasp
  sources. GraspGenX's transform is **DERIVED from kinematics** (recipe:
  `scripts/tools/derive_tool_transform.py` + `ee/hand_kinematics.py`): rotation =
  approach +Z→wrist +Y, closing +X→wrist −X, spread +Y→wrist +Z =
  `wristyaw_grasp_rpy: [π/2,0,π]` (SIM-VALIDATED 2026-06-28; the Dex3's true opposition
  is diagonal in the wrist XY-plane — this clean-axis map is the sim-matched palm-down
  approximation, revisit per-object); translation = power_close contact midpoint (FK)
  minus the 0.07 m fingertip depth along approach = `palm_offset_xyz: [0.1142,−0.0286,0]`.
  Config stores the RIGHT hand; LEFT is mirrored across the wrist Y-plane in code.
  HARDWARE-CONFIRMED incl. the closing-roll sign (first real grasp, 2026-06-30).
- **Grasp pipeline:** `robot.grasp_source` is a `GraspSource` (grasp.yaml `source:` =
  `graspgenx` | `sim_cloud`) returning ranked wrist-yaw `GraspCandidate`s and retaining a
  typed `SourceSnapshot` (`last_snapshot`: target/mask/cloud/n_candidates — the mask
  feeds `collision_world.exclude_object`, the cloud feeds `perception/validation.py`).
  `grasp_motion` plans them with cuRobo's **native** `plan_grasp`
  (`plan_grasp_set_sweep`: K-candidate **goalset** — cuRobo picks the feasible grasp — +
  approach/grasp/lift segments, strategy sweep from `planner.yaml: grasp.strategies`).
  `plan_grasp` offsets EVERY goal tool frame, so the grasp planner is a dedicated
  single-tool-frame `MotionPlanner` per side (`_grasp_planner`); a lone candidate is
  duplicated into a 2-row goalset (`warmup` never primes the `num_goalset=1` path).
  GraspGenX path: head depth → SAM3 mask (ZMQ `:5557`, applied PRE-deproject) →
  `deproject_depth` → pelvis `PointCloud` → GraspGenX ZMQ (`:5556`, protocol v2:
  `infer` → `(grasps, conf, branch_tags)`; `planner` ∈ {diffusion|graspmoe|**topdown**};
  grasp.yaml defaults `topdown` for grab-from-above) → tool transform. `sim_cloud` =
  SIM-only GT cube cloud from the `rt/sim_state` block pose → same GraspGenX + transform.
  Both ZMQ clients are thin standalone shims that never import their service package
  (the multi-GB import trap); wire deps `msgpack`/`msgpack-numpy`. **REAL-VALIDATED
  2026-06-30** (graspgenx + SAM3 + topdown, full pick+lift).
- **Depth-ESDF collision world** (`motion/collision_world.py`; gated OFF —
  `set_collision_world(True)` / `examples/02_pick.py --collision-world`): head depth →
  cuRobo `Mapper` → ESDF `VoxelGrid` → the grasp planner, so the plan_grasp **approach**
  routes around the table/clutter. Source-independent, same path sim + real. The TARGET
  object STAYS IN the world (`exclude_object` default OFF, 2026-07-03): native plan_grasp's
  per-step link disabling already gives the wanted semantics — the Step-2 APPROACH plans
  with ALL links enabled (arm + hand collision-checked against the target, so the motion
  to the pre-grasp can't sweep through it), while Steps 1/3/4 disable the hand links for
  the intended grasp/lift contact. The old default-ON cut ("in-world target blocks its own
  pre-grasp") predated the voxel-dims fix and made the approach BLIND to the target; it
  remains the opt-in clutter escape hatch (from the source snapshot's SAM3 mask). A cuRobo
  `RobotSegmenter` **self-filter** removes
  the robot's own pixels at the LIVE measured config (arm q + finger q via a dedicated
  hand-active segmenter; hand q mapped BY NAME — Dex3 right-hand `get_q` swaps
  index/middle vs left). `diagnostics.world_check(planner, side, q)` truth-tests a config
  on the grasp planner's own ESDF (activation 0, metrics_base.yml). `tools/check_world.py`
  inspects the world in isolation. **cuRobo voxel bug (found 2026-07-02):** the collision
  kernel truncates float32 grid dims (119.99999 → 119 → wrong strides → phantom hits +
  invisible obstacles); worked around by `collision_world.kernel_safe_dims` on every grid
  we author — do NOT remove until upstream fixes land (a regression test flags when).
  Full report: `docs/curobo_voxel_dims_bug.md`. The 06-30 real run predated this fix, so
  the obstacle-avoidance claim needs re-validation.
- **Perception validation** (`perception/validation.py` + `tools/validate_perception.py`):
  the SIM regression gate scoring the full stack vs `rt/sim_state` GT — cloud
  centroid/chamfer/inliers/bbox, SAM3-mask IoU vs the projected GT (separates
  segmentation from extrinsics error), grasp-implied object error, fingertip FK. Exit
  codes gate on `--max-centroid-mm` / `--max-chamfer-mm`.
- Hand control (`hardware/robot_hand_unitree.py`): threaded `Dex3Controller` /
  `Dex1Controller`, publishes BOTH hands continuously; exposes `q/dq/tau/press`. Presets
  in `ee/dex3.py` + `configs/hands.yaml`; `pinch` + the `verify` thresholds still need
  tuning. Hand verbs report "commanded (unverified)" unless `verify=True`.
- Arm controller: debug mode = `rt/lowcmd` + locks non-arm joints at current q
  (suspended-robot correct); `simulation_mode` bypasses the velocity clip.
- DDS domain/interface live ONLY in `configs/robot.yaml`. Real = (0, NIC-to-PC2); sim =
  loopback (1, "lo") — hardwired into `connect("sim")` — + `configs/cyclonedds_loopback.xml`
  on both ends. See SIM_NOTES.md.

## Gotchas
- The sim runs the **calibrated mode_16 dex3 USD** (sim == cuRobo == real kinematics) and
  maps hand-command joints **by name**. Its hand-apply path gates on BOTH hands having a
  command — our controller publishes both, so it's satisfied.
- Hand primitives with `verify=False` **dwell** until the motion completes (they used to
  return instantly, so close→open looked like nothing happened).
- Camera configs: `connect()` picks `camera_sim.yaml` / `camera_real.yaml` by target.
  `camera_real.yaml` carries the REAL calibration (2026-06-18 hand-eye `T_pelvis_camera`
  + ZED intrinsics) — never regenerate it casually. `cam_config_client.yaml` at the repo
  root is loaded at runtime by the vendored `hardware/zmq_image_client.py`; keep it.
- `hardware/` is vendored upstream code (xr_teleoperate, unitree_lerobot) — naming
  follows upstream, edit sparingly.
- Scripts: `scripts/checks/` (bring-up ladder 01→05) → `scripts/examples/` →
  `scripts/tools/`; the table + old→new name map is `scripts/README.md`. Shared operator
  helpers live in `g1_primitives/api/console.py` (NOT part of the agent surface).
- **Deleted** (recoverable from git): the AprilTag stack (apriltag_block/apriltag_source/
  ground_truth, 06_detect/07_pick_place, `move_to_candidates`/`plan_to_pose_set`),
  `factory.py`/`scripts/_rig.py` (absorbed by the facade + console), camera_rig/
  cameras.yaml (wrist-cam notes moved to HARDWARE_TODO.md), and the v1
  pinocchio/Cartesian/FSM pipeline.
