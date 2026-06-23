# HARDWARE_TODO — what's left for the robot & camera

The **cuRobo-native MVP now runs on the PHYSICAL G1** (2026-06-18): `home → move → close_hand
→ open_hand → home` end-to-end on the real robot, after first validating in `unitree_sim_isaaclab`.
Getting there needed: operator-set debug mode (no MotionSwitcher), a direct un-planned launch home,
gravity comp on, `arm_velocity_limit ≥ ~12`, and trajectory `time_dilation 0.5` — see the Motion
section below and `docs/gravity_comp.md`. This file lists what still needs the physical G1 / Dex3 /
head camera, with the command and the pass criterion.

> Everything runs in the **`g1_curobo`** env (`bash -ic 'use_conda g1_curobo && …'`). The
> hardware entry points are the numbered bring-up ladder `scripts/0{1..7}_*.py` (each
> `--target real`), run **in order** — `01_check_dds` (read-only) and `02_check_image` are
> safe/no-motion, `03`–`05` command the arms/hands, `06` is camera-only. `scripts/hand_diag.py`
> is the low-level hand command→state diagnostic. The real image client (ZED) is **wired**;
> `02`/`06 --target real` need the ZED intrinsics + mount filled in `configs/camera_real.yaml`.

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
  `camera_real.yaml`); sim is `camera_sim.yaml`; selected by `--target`.

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
- [ ] **ZED calibration (USER-PROVIDED)** — fill `camera_real.yaml` intrinsics (chosen eye @1280×720)
      + `extrinsics.mount` (d435_link→eye, ~half the stereo baseline); pick `stereo_side`.
- [ ] **Feed check** — `python scripts/02_check_image.py --target real` shows one sliced ZED eye.
- [ ] **AprilTag pose** — `python scripts/06_detect.py --target real`; validate the detected block
      pose against a tape-measured position (confirms intrinsics + mount).

## Composite tasks (the LLM-composable baseline)
- [x] **Pick + lift — DONE** (`scripts/07_pick_place.py`): home → open → detect → approach →
      grasp → close → lift → home, single arm, per-step operator gating (`--no-confirm` to skip),
      with a URDF-measured side-aware palm/grasp offset so a detected pose becomes a grasp pose.
      **Open:** place / handover, dual-arm, multi-object, and tuning the grasp on hardware.
- [ ] **Rerun logging** — wire current q / target pose / state into the primitives for debugging.

---

## Offline status (verified, no robot)
- `pytest tests/` → **18 passed** (`test_pose` SE(3) conventions, `test_grasp` verification,
  `test_detect` perception frame-math + stereo/filter).
- cuRobo FK parity vs the old pinocchio model: **< 0.001 mm / 0.027°** (cuRobo is a faithful
  drop-in); `Pose` util convention-faithful to ~1e-15.
- cuRobo → sim execution: EE position error ~0.45 cm, native plan ~1.1 rad/s, zero aborts.
- Every active module imports under the `g1_curobo` env.

## Environment notes
- Run via `bash -ic 'use_conda g1_curobo && <cmd>'` (lane wrapper sets `PYTHONNOUSERSITE=1`).
- **No pinocchio / no numpy<2 pin** — kinematics are cuRobo's. Build is not plain
  `pip install`; see `requirements-curobo.txt`. Sim target = `unitree_sim_isaaclab` (Isaac
  Sim in the `unitree` env), loopback DDS. See `SIM_NOTES.md`.
