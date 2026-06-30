# HARDWARE_TODO — what's left for the robot & camera

The **cuRobo-native MVP now runs on the PHYSICAL G1** (2026-06-18): `home → move → close_hand
→ open_hand → home` end-to-end on the real robot, after first validating in `unitree_sim_isaaclab`.
Getting there needed: operator-set debug mode (no MotionSwitcher), a direct un-planned launch home,
gravity comp on, `arm_velocity_limit ≥ ~12`, and trajectory `time_dilation 0.5` — see the Motion
section below and `docs/gravity_comp.md`. This file lists what still needs the physical G1 / Dex3 /
head camera, with the command and the pass criterion.

> Everything runs in the **`g1_curobo`** env (`bash -ic 'use_conda g1_curobo && …'`). The
> hardware entry points are the numbered bring-up ladder (`scripts/01_*…12_*`, each
> `--target real`), run **in order** — `01_check_dds` (read-only) and `02_check_image` are
> safe/no-motion, `03`–`05`/`07`/`09` command the arms/hands, `06`/`08`/`10`/`11`/`12` are camera-only
> (`08` = head **depth** feed; `10` = SAM3 **segmentation**; `11_capture_frame` saves an offline
> fixture; `12_check_world` = depth-ESDF collision world inspector — no planning, no motion;
> `10_graspgen_viz` needs no robot). `09_graspgen` is the GraspGenX pick+lift; `09 --source
> graspgenx` + `10`/`10_graspgen_viz` need the **GraspGenX** (`:5556`) / **SAM3** (`:5557`) ZMQ
> servers up (started in their own repos/envs — launch cmds in README "Grasp pipeline"). `scripts/hand_diag.py` is the low-level hand
> command→state diagnostic. The real image client (ZED) is **wired** (color + a raw-float32
> **depth** stream); `02`/`06 --target real` use the ZED intrinsics + mount in `configs/camera_real.yaml`.

---

## Launch readiness — where we are
**The action-primitive MVP runs on the physical robot (2026-06-18):** `home → move → close_hand →
open_hand → home` end-to-end. Remaining work is perception calibration (ZED) + tuning, not the
motion code path.

**Validated on hardware (action primitives):**
- DDS arm/hand control (`debug` mode = `rt/lowcmd` + non-arm joints locked).
- **Debug mode is operator-set via the physical remote** — `make_robot` does NOT call
  `MotionSwitcher` by default (auto-`ReleaseMode` dropped the robot out of low-level control;
  opt in with `enter_debug_mode=True`).
- Direct un-planned **launch home** (`Executor.go_home_direct`, `home_on_connect=True`) — moves the
  arms out of the folded power-on pose, which cuRobo's collision-aware planner won't plan out of.
- Velocity cap (`robot.yaml: arm_velocity_limit`) — keep `≥ ~12` (it also caps PD torque).
- **Gravity-comp feed-forward ON for real** (cuRobo RNEA, sign-validated — `docs/gravity_comp.md`),
  forced off in sim.
- **Trajectory `time_dilation 0.5`** so the torque-throttled arm tracks the plan (`planner.yaml`).
- Perception: AprilTag `detect()` + head-camera client. Real ZED **wired** (stereo-slice, RGB-first,
  **+ a raw-float32 depth stream**, `camera_real.yaml`); sim is `camera_sim.yaml`; selected by `--target`.

**Still needs the physical robot / camera:** fill the camera config (ZED intrinsics/mount) and run
`06_detect --target real`; tune hand presets; resolve the velocity-clip torque root-cause so
moves track at full speed (time-dilation is the current workaround); and validate the depth-ESDF
collision world + the planned-trajectory speed/tracking margin on real grasps (see the Motion and
Composite-tasks sections; `docs/trajectory_speed_tracking.md`). Not motion-code blockers.

---

## Config to fill in first (USER-PROVIDED)
| File | Field | What |
|---|---|---|
| `configs/robot.yaml` | `dds.interface` | NIC on this workstation wired to PC2 (e.g. `enp5s0`); set `mode: debug`. Real robot = domain 0; sim loopback = domain 1 / `lo`. |
| `configs/robot.yaml` | `home_q14_deg` | The launch/ready pose (default zeros = forearms forward). Confirm it's safe + reachable on the suspended robot. |
| `configs/hands.yaml` | `dex3.presets` (`open`/`power_close`/`pinch`) + `verify` thresholds | Being tuned: right-thumb `power_close` stall **fixed**; still make `pinch` per-hand and tune the `verify` thresholds against the real hand. |
| `configs/camera_real.yaml` | `intrinsics` (fx,fy,cx,cy @1280×720) + `extrinsics.mount` + `stereo_side` | **Real head = ZED stereo.** Per-eye intrinsics for the chosen eye (ZED calibration) + the `d435_link`→eye `mount` offset (~half the stereo baseline) or detected block poses are laterally biased. |
| `configs/curobo/g1_dex3_curobo.yml` | `lock_joints` values (+ `velocity_scale`) | Leg+waist locked positions if the mount tilts the pelvis; lower `velocity_scale` to slow the robot for bring-up. |
| cuRobo world model | table / obstacles | **Depth-ESDF collision world now wired** (head depth → cuRobo Mapper → ESDF, `motion/collision_world.py`, `planner.yaml: grasp.collision_world`), but **OFF by default** (`enabled: false`) and **pending real validation** — see "Depth-ESDF collision world" under Composite tasks. A static table/obstacle world is still not configured separately; the ESDF is the live-perception alternative. |

---

## Robot interface bring-up
- [ ] **DDS connectivity** — `make_robot(connect_dds=True, dds_domain=0, dds_interface="<NIC>",
      mode="debug")` constructs the arm + Dex3 controllers; reading `arm.get_current_dual_arm_q()`
      returns live joint q from the suspended G1. For a READ-ONLY check first (no controllers,
      no motion), run `scripts/01_check_dds.py --target real`.
- [ ] **Launch home — WIRED (default).** `make_robot(connect_dds=True)` now drives the arms to
      `home_q14_deg` with DIRECT, un-planned PD (`Executor.go_home_direct`, velocity-capped,
      ramped) and waits for convergence BEFORE returning. This is deliberately NOT cuRobo-planned:
      the collision-aware `plan_cspace` refuses to plan *out of* a pose it flags as a self-collision
      START (seen on first contact — the arms powered on folded, so `home()`'s plan failed with
      "Start or End state in collision"). The direct home sidesteps that. **It is NOT collision-
      avoided en route**, so ensure the path from the power-on pose to home is clear and watch the
      e-stop. Override with `home_on_connect=False`. After it converges, the planned `home`/`move`
      primitives plan from a known collision-free start.
- [ ] **Debug mode — OPERATOR-SET (physical remote).** Low-level `rt/lowcmd` control needs the
      robot in debug mode; **set it on the suspended robot with the physical controller before
      launch** (same as the proven `unitree_lerobot` flow, which never calls `MotionSwitcher`).
      `make_robot` does NOT auto-enter debug mode by default: calling `MotionSwitcher.ReleaseMode()`
      on an already-debug robot dropped it back out of low-level control, so `rt/lowcmd` was ignored
      and the arms didn't move (launch home residual ~87° = zero motion). Pass `enter_debug_mode=True`
      only if no remote is available to set it.
- [ ] **Hand state** — `python scripts/hand_diag.py --side left|right` opens/closes each hand and
      streams q / tau_est / press. **Confirm press pressures are non-zero and the closing-direction
      signs in `hands.yaml` are right** (`Dex3Controller` reads `motor_state.{q,dq,tau_est}` +
      `press_sensor_state.pressure`; verify against the real `HandState_`).

## Motion on hardware
- [x] **Tracking — `home` + a `move` run on the physical G1 (2026-06-17).** `home` lands within
      ~3°; a 0.2 m left-wrist lift completes with `max tracking error < 0.20` (no abort) at
      `time_dilation: 0.5`. **Caveat — the velocity clip throttles PD torque.** `clip_arm_q_target`
      caps the commanded-vs-measured error to `arm_velocity_limit·control_dt`, hence PD torque to
      `~kp·arm_velocity_limit·control_dt`. So the clip fires on EVERY tracked move (it is
      measured-relative), and at full trajectory speed the arm can't supply enough torque → the
      tracking error diverges → abort. Two settings tame it: keep `arm_velocity_limit ≥ ~12` (was
      validated at 20) and `time_dilation ≤ 0.5` (below).
- [x] **Speed — `time_dilation` knob in `planner.yaml` (2026-06-17).** cuRobo plans ~1.1 rad/s; the
      real PD can't track that through the torque-throttling clip, so the executor plays the plan
      back slower: `executor.time_dilation` (default **0.5** for real, forced 1.0 in sim). Same
      path/goal, lower velocity + quadratically lower accel. Per-run: `04_move --speed`; dial toward
      1.0 to find the fastest that tracks. **Root-cause fix (deferred):** stop re-rate-limiting
      cuRobo's already-feasible trajectory during planned execution (bypass the clip like sim does,
      or make it command-relative) so full-speed moves track without dilation.
- [x] **Gravity comp — VALIDATED ON HARDWARE, default ON for real (2026-06-17).** cuRobo RNEA
      `G(q)` feed-forward (`planner.gravity_torque` → `executor._tauff`), `configs/planner.yaml`
      `executor.gravity_comp: true` / `gravity_scale: 1.0`; `factory.py` forces it OFF in sim.
      Confirmed on the physical G1: sign correct (elbow droop fell 50°→3° ramping scale 0→1.0),
      `home` lands within ~3°. **Caveat:** the velocity clip caps PD torque at
      `≈ kp·arm_velocity_limit·control_dt`, so gravity comp alone isn't enough — keep
      `arm_velocity_limit ≥ ~12` or the shoulders stall a few deg short of target (full table +
      mechanism in **`docs/gravity_comp.md`**). Ramp/re-check via `04_move --gravity-scale`.
- [ ] **Trajectory speed vs tracking — WATCH on real grasps; root fix DEFERRED** (diagnosed in sim
      2026-06-29, `docs/trajectory_speed_tracking.md`). cuRobo plans to stock-aggressive dynamics
      (`g1_dex3_curobo.yml` cspace: `max_acceleration 10`, `max_jerk 500`, `velocity_scale =
      acceleration_scale = jerk_scale = 1.0` — no de-rating). The depth-ESDF collision-world
      **approach** (curves around the obstacle) is dynamic enough that the executor's tracking-error
      abort fires at full-speed playback — in SIM, with the clip bypassed and a 2× looser 0.40 budget
      (`approach: tracking error 0.401 > 0.400 rad`); planning succeeded, EXECUTION aborted.
      `09_graspgen --speed 0.5` clears it. **Real is the HARDER target, not safe-by-default:** it runs
      the same plan at `time_dilation 0.5` (a head start) but with the tighter **0.20** abort budget
      and the **active** torque-capping velocity clip. The `time_dilation` workaround is in place; the
      **root fix** — lower `acceleration_scale`/`velocity_scale` in the shared cspace so the plan is
      gentle enough to track at `time_dilation 1.0`, retuned together with real's dilation — is
      DEFERRED (it touches the hardware-validated config and would compound with the 0.5 if changed
      naïvely → very slow real motion). **Pass:** real grasps/approaches track with no
      `tracking error > 0.20` abort. If they abort, confirm with `--speed` (lower it) first — see the
      doc's watch list.

## Hand grasp tuning
- [ ] **Presets** — command `open`/`power_close`/`pinch` with `hand_diag.py`, read back q, fix the
      7-vectors + closing-direction signs in `hands.yaml`. Watch for fingers stalling short of
      target with **tau saturating** (joint-limit ceiling — back the preset off for those joints).
- [ ] **Thresholds** — set `verify.{stall_margin_rad, tau_threshold, press_threshold, still_dq}`
      so closing on a block → `grasped=True` and closing on air → `False`. **Pass:** 10/10 each.

## Perception / head camera on hardware
- [x] **Image client — WIRED.** Real head = ZED stereo: `make_robot(camera_config="camera_real.yaml")`
      (via `--target real`) opens the vendored ZMQ client, slices the 720×2560 side-by-side frame to a
      single 1280×720 eye (`stereo_side`), serves RGB-first frames. Sim stays on `camera_sim.yaml`.
- [x] **Head DEPTH — WIRED (2026-06-23).** A dedicated raw-float32 ZMQ stream (server port 56555,
      720×1280, single left eye, **millimeters**, NaN/inf = invalid) consumed via
      `HeadCamera.get_depth_frame()`, gated by `camera_real.yaml: stream.depth`. **Validated on the
      real G1:** (720,1280), ~88.9% finite, ~30 fps, thousands of frames decoded with zero errors.
      **Feed check:** `python scripts/08_check_depth.py --target real` (mm stats + colorized view).
      Wire spec: `docs/depth_integration_handoff.md`. **Deproject → cloud → grasp is now built**
      (`perception/depth.deproject_depth` + the grasp pipeline below); pending = the real grasp run.
- [ ] **ZED calibration (USER-PROVIDED)** — fill `camera_real.yaml` intrinsics (chosen eye @1280×720)
      + `extrinsics.mount` (d435_link→eye, ~half the stereo baseline); pick `stereo_side`.
- [ ] **Feed check** — `python scripts/02_check_image.py --target real` shows one sliced ZED eye.
- [ ] **AprilTag pose** — `python scripts/06_detect.py --target real`; validate the detected block
      pose against a tape-measured position (confirms intrinsics + mount).

## Composite tasks (the LLM-composable baseline)
- [x] **Pick + lift (AprilTag) — DONE** (`scripts/07_pick_place.py`): home → open → detect →
      approach → grasp → close → lift → home, single arm, per-step operator gating, with a
      URDF-measured side-aware palm/grasp offset so a detected pose becomes a grasp pose.
- [x] **Grasp pipeline (GraspGenX + SAM3) — BUILT, offline-tested** (`grasp/`,
      `perception/{depth,segment,sam3_client,segment_gui}.py`, `spatial/pointcloud.py`,
      `scripts/09_graspgen.py` + `scripts/10_segment.py`): `GraspSource` seam (`grasp.yaml:
      grasp_source` = `apriltag` | `graspgenx` | `sim_cloud`); GraspGenX path = head depth → masked
      deproject → pelvis `PointCloud` → **SAM3** segmentation (`:5557`, 2D mask PRE-deproject,
      interactive cv2 GUI) → **GraspGenX** (`:5556`, protocol v2: `planner` topdown/graspmoe/diffusion,
      obb/diff `branch_tags`) 6-DoF grasps → `tool_transform` → **native cuRobo `plan_grasp`**
      (`grasp_motion` → `plan_grasp_set_sweep`: goalset pick + approach/grasp/lift; `--legacy` =
      old sequential `plan_to_pose_set`). Standalone ZMQ clients (no service import). **SAM3
      segmentation validated on static images** (`10_segment --image`); the offline demo
      (`10_segment --save` → COLORED `.ply` → `10_graspgen_viz`) renders cloud + grasps in viser.
  - [x] **Sim de-risk (`--source sim_cloud`) — DONE 2026-06-28**: in Isaac, a GT cube cloud from
        `rt/sim_state` → GraspGenX → the same tool transform + gated motion + cuRobo + physics, with
        NO ZED/SAM3 (only the GraspGenX server + the sim). `09_graspgen --target sim --source
        sim_cloud --visualize` drove a 6-DoF grasp to the correct pose; the FK contact check
        (`--source sim_cloud` straddle-the-GT-cube) confirms our Dex3 fingers land on the object.
        Needs `perception.yaml: detector: sim_state`.
  - [ ] **Real grasp run** — start the GraspGenX (`:5556`) + SAM3 (`:5557`) servers + robot, then
        `09_graspgen --target real --source graspgenx --segment interactive` vs `07_pick_place
        --target real` (the AprilTag baseline — A-B on the same object; 09 is GraspGenX-only).
        `10_segment --target real` first to confirm the live mask + masked
        point count.
  - [ ] **Depth-ESDF collision world — BUILT + sim-validated, PENDING real validation** (commits
        dfca2e7..f4448eb; `motion/collision_world.py`, `planner.yaml: grasp.collision_world`). Head
        depth → cuRobo `Mapper` → ESDF `VoxelGrid` → the grasp planner, so `plan_grasp`'s APPROACH
        routes around the object/table instead of barging through it (SAME path sim + real,
        source-independent). A cuRobo `RobotSegmenter` self-filter zeros depth within
        `robot_mask_margin` (0.02) of the robot's collision spheres at the **LIVE** arm + finger q
        (live hand q via `Dex3Hand.get_q`, a hand-active 28-DoF segmenter kinematics) so the camera
        doesn't fuse the arm into the world. **OFF by default** (`enabled: false`); opt in via the
        flag or `09_graspgen --collision-world`. Inspect IN ISOLATION (no planning/motion) with
        `scripts/12_check_world.py --target real` (`--visualize` viser overlay; `--probe X Y Z` is the
        object in the ESDF or erased by the filter?; `--margin` sweeps the self-filter margin).
        **Pass on real:** ESDF occupied voxels match the object geometry, the robot's own arm is
        removed, and the block stays solid (~5 voxels at the 1cm ESDF, not eroded to ~1). **Watch:**
        too-big a margin erodes the object out of the world; the planned approach is also more
        dynamic → see the trajectory speed/tracking item under Motion.
  - [x] **`wristyaw_grasp_rpy` + `palm_offset_xyz` — DERIVED + sim-validated 2026-06-28**: no longer
        a guess — `scripts/derive_graspgenx_tool_transform.py` derives the grasp→wrist_yaw map from
        Dex3 FK (`[π/2,0,π]`: approach +Z→wrist +Y; `palm_offset [0.1142,−0.0286,0]`), sim-confirmed
        via `--source sim_cloud` (palm faces down, FK contact on the cube). REMAINING: confirm the
        closing-roll sign on one gated hardware grasp (the clean-axis map approximates the Dex3's
        diagonal opposition — a no-op for a symmetric cube; revisit per-object).
  - [ ] **Planner hand locked at the DEPLOYED-open pose — VALIDATE on real** (commit 2a71238).
        The cuRobo config now locks the planner's hand at the deployed "open" preset
        (`g1_dex3_curobo.yml: lock_joints` — `{side}_hand_thumb_1_joint` at the URDF open limit,
        right +0.7243 / left −0.7243, all other hand joints 0), NOT all-zero, so the static hand
        collision spheres match the hand we actually plan/approach with. Side effect FIXED: the open
        thumb's sphere clipped the wrist-yaw sphere → cuRobo flagged HOME as a self-collision start;
        patched `{side}_hand_thumb_1_link <-> {side}_wrist_yaw_link` into `self_collision_ignore`
        (conservative-sphere artifact — same class as the torso↔shoulder patch). **Validate on real:**
        `home`/`move`/`grasp_motion` plan cleanly from the launch pose (no "Start state in collision")
        and the open-thumb geometry the planner assumes matches the real Dex3 open hand. (This
        supersedes the old CLAUDE.md facts "27 lock_joints … all at 0" and the torso↔shoulder-only
        self_collision_ignore note — both now incomplete.)
  - **Open:** place / handover, dual-arm, multi-object; SAM3 `image_jpeg` bandwidth path.
- [ ] **Rerun logging** — wire current q / target pose / state into the primitives for debugging.

---

## Offline status (verified, no robot)
- `pytest tests/` → **green** (`test_pose`/`test_grasp`/`test_detect` + the grasp pipeline:
  `test_pointcloud`, `test_depth_deproject`, `test_tool_transform`, `test_graspgenx_client` +
  `test_sam3_client` (mock ZMQ servers), `test_grasp_source`, `test_segment`).
- cuRobo FK parity vs the old pinocchio model: **< 0.001 mm / 0.027°** (cuRobo is a faithful
  drop-in); `Pose` util convention-faithful to ~1e-15.
- cuRobo → sim execution: EE position error ~0.45 cm, native plan ~1.1 rad/s, zero aborts.
- Every active module imports under the `g1_curobo` env.

## Environment notes
- Run via `bash -ic 'use_conda g1_curobo && <cmd>'` (lane wrapper sets `PYTHONNOUSERSITE=1`).
- **No pinocchio / no numpy<2 pin** — kinematics are cuRobo's. Build is not plain
  `pip install`; see `requirements-curobo.txt`. Sim target = `unitree_sim_isaaclab` (Isaac
  Sim in the `unitree` env), loopback DDS. See `SIM_NOTES.md`.
- **Grasp pipeline deps:** `msgpack` + `msgpack-numpy` (the GraspGenX/SAM3 ZMQ wire protocol)
  — pure pip wheels. The **GraspGenX** (`:5556`) and **SAM3** (`:5557`) inference services run
  **separately on the workstation GPU** (their own repos/envs); this repo only ships thin clients
  that never import those packages. SAM3 launch: `use_conda sam3 && python -m sam3.serving --port 5557`.
