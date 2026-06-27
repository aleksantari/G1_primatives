# HARDWARE_TODO — what's left for the robot & camera

The **cuRobo-native MVP now runs on the PHYSICAL G1** (2026-06-18): `home → move → close_hand
→ open_hand → home` end-to-end on the real robot, after first validating in `unitree_sim_isaaclab`.
Getting there needed: operator-set debug mode (no MotionSwitcher), a direct un-planned launch home,
gravity comp on, `arm_velocity_limit ≥ ~12`, and trajectory `time_dilation 0.5` — see the Motion
section below and `docs/gravity_comp.md`. This file lists what still needs the physical G1 / Dex3 /
head camera, with the command and the pass criterion.

> Everything runs in the **`g1_curobo`** env (`bash -ic 'use_conda g1_curobo && …'`). The
> hardware entry points are the numbered bring-up ladder (`scripts/01_*…11_*`, each
> `--target real`), run **in order** — `01_check_dds` (read-only) and `02_check_image` are
> safe/no-motion, `03`–`05`/`07`/`09` command the arms/hands, `06`/`08`/`10`/`11` are camera-only
> (`08` = head **depth** feed; `10` = SAM3 **segmentation**; `11_capture_frame` saves an offline
> fixture; `10_graspgen_viz` needs no robot). `09_graspgen` is the GraspGenX pick+lift; `09 --source
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
`06_detect --target real`; tune hand presets; and resolve the velocity-clip torque root-cause so
moves track at full speed (time-dilation is the current workaround). Not motion-code blockers.

---

## Config to fill in first (USER-PROVIDED)
| File | Field | What |
|---|---|---|
| `configs/robot.yaml` | `dds.interface` | NIC on this workstation wired to PC2 (e.g. `enp5s0`); set `mode: debug`. Real robot = domain 0; sim loopback = domain 1 / `lo`. |
| `configs/robot.yaml` | `home_q14_deg` | The launch/ready pose (default zeros = forearms forward). Confirm it's safe + reachable on the suspended robot. |
| `configs/hands.yaml` | `dex3.presets` (`open`/`power_close`/`pinch`) + `verify` thresholds | Being tuned: right-thumb `power_close` stall **fixed**; still make `pinch` per-hand and tune the `verify` thresholds against the real hand. |
| `configs/camera_real.yaml` | `intrinsics` (fx,fy,cx,cy @1280×720) + `extrinsics.mount` + `stereo_side` | **Real head = ZED stereo.** Per-eye intrinsics for the chosen eye (ZED calibration) + the `d435_link`→eye `mount` offset (~half the stereo baseline) or detected block poses are laterally biased. |
| `configs/curobo/g1_dex3_curobo.yml` | `lock_joints` values (+ `velocity_scale`) | Leg+waist locked positions if the mount tilts the pelvis; lower `velocity_scale` to slow the robot for bring-up. |
| cuRobo world model | table / obstacles | **Not yet wired** — no world is configured. Add the table (and any obstacle) to the planner's world before relying on collision avoidance near it. |

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
      interactive cv2 GUI) → **GraspGenX** (`:5556`) 6-DoF grasps → `tool_transform` →
      `plan_to_pose_set` picks a reachable one. Standalone ZMQ clients (no service import). **SAM3
      segmentation validated on static images** (`10_segment --image`); the offline demo
      (`10_segment --save` → COLORED `.ply` → `10_graspgen_viz`) renders cloud + grasps in viser.
  - [ ] **Sim de-risk first (`--source sim_cloud`)** — in Isaac, a GT cube cloud from `rt/sim_state`
        → GraspGenX → the same tool transform + gated motion + cuRobo + physics, with NO ZED/SAM3
        (only the GraspGenX server + the sim). Run `09_graspgen --target sim --source sim_cloud
        --visualize` to tune `wristyaw_grasp_rpy` (roll) and watch a 6-DoF grasp execute safely
        BEFORE the robot. Needs `perception.yaml: detector: sim_state`.
  - [ ] **Real grasp run** — start the GraspGenX (`:5556`) + SAM3 (`:5557`) servers + robot, then
        `09_graspgen --target real --source graspgenx --segment interactive` vs `--source apriltag`
        (A-B on the same object). `10_segment --target real` first to confirm the live mask + masked
        point count.
  - [ ] **`wristyaw_grasp_rpy` calibration (EMPIRICAL)** — the grasp(+Z approach,+X closing) →
        wrist_yaw axis map in `configs/grasp.yaml` is a best-guess seed (pitch +90°). Tune the roll
        (closing-axis alignment) in **sim (`--source sim_cloud`)**, then confirm with one gated
        hardware grasp before trusting.
  - **Open:** place / handover, dual-arm, multi-object; native cuRobo goalset (vs the sequential
        `plan_to_pose_set`); SAM3 `image_jpeg` bandwidth path.
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
