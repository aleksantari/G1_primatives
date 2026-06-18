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
   `close_hand`, each returning `Result(ok, info)`. Tasks are composed from these. Keep
   new behavior out of here unless it's a genuine new primitive.
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
  `left/right_wrist_yaw_link`; 27 `lock_joints` → arms-only 14-DoF. `self_collision_ignore`
  patched so `torso_link` ignores all 6 shoulder links (the auto-matrix asymmetrically missed
  3, causing false start-in-collision at the home pose). **Regenerate with
  `configs/curobo/build_g1_dex3.py`** (the provenance/recipe; not run at import — only when the
  URDF or the arms-only reduction changes; the committed config is hardware-validated).
- **Home = the sim launch pose: all arm joints 0 = forearms forward (elbows bent ~90°)**,
  collision-free. `configs/robot.yaml: home_q14_deg`.
- cuRobo native plan ≈ 1.1 rad/s, `dt` 0.025 s. Trajectory SPEED is governed by the cuRobo
  robot config's joint limits, but the executor **plays it back at `executor.time_dilation`**
  (planner.yaml; default 0.5 on real, forced 1.0 in sim). Real needs the slowdown: the arm
  controller's velocity clip (`clip_arm_q_target`) is measured-relative, so it caps PD torque at
  `~kp·arm_velocity_limit·control_dt` and the arm can't track a full-speed trajectory → tracking
  error diverges → abort. So on real: `gravity_comp` ON (holds the arm), `arm_velocity_limit ≥ 12`
  (torque headroom), `time_dilation 0.5` (tracks). Sim bypasses the clip entirely, so it runs full
  speed with no gravity comp. The root fix (deferred) is to not re-rate-limit cuRobo's already-
  feasible trajectory during planned execution. See `docs/gravity_comp.md`, `HARDWARE_TODO.md`.
- Goal frame is the **wrist-yaw link** directly. The 5 cm `L_ee`/palm offset is a later
  refinement (goals are wrist-yaw poses for now).
- Hand control (`robot_control/robot_hand_unitree.py`): threaded `Dex3Controller` /
  `Dex1Controller`, **publishes both hands continuously**; exposes `q/dq/tau/press` (grasp
  signals). Presets + verification in `ee/dex3.py` + `configs/hands.yaml`. **Presets are
  PLACEHOLDERS — untuned** (the right thumb stalls on `power_close`).
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
  `image_server/image_client.py`. Current scripts: the numbered bring-up ladder
  `scripts/0{1..6}_*.py` (check_dds → check_image → hands → move → mvp_demo → detect; each
  `--target sim|real`, shared `scripts/_rig.py`) + `hand_diag.py` (low-level hand diagnostic).
- **Dormant on disk** (unimported, kept for later): `image_server/camera_rig.py` +
  `configs/cameras.yaml` (multi-camera). (`robot_control/motion_switcher.py` is wired but
  **opt-in** — `make_robot` lazily imports it only when `enter_debug_mode=True`; default off,
  see the debug-mode gotcha above.)
- **Deleted** (recoverable from git): `robot_control/robot_arm_ik.py`,
  `motion/cartesian_planner.py`, `motion/retimer.py`, `tasks/*`, `utils/*`
  (rerun/episode/filter), the legacy v1 scripts (`run_*`, `0X_*`, `offline_*`,
  `05_reach_check*`), and their `configs/task_*.yaml`.
