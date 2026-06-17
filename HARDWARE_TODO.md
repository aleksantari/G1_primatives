# HARDWARE_TODO — what's left for the robot & camera

The **cuRobo-native MVP is sim-validated** (`home → move → close_hand → open_hand → home`
on `unitree_sim_isaaclab`, zero executor aborts; `home`/`move` collision-aware via cuRobo).
It has **never been exercised on the physical robot** — no real DDS/camera path has been run.
This file lists what still needs the physical G1 / Dex3 / head camera, with the command and
the pass criterion.

> Everything runs in the **`g1_curobo`** env (`bash -ic 'use_conda g1_curobo && …'`). The
> hardware entry points are the numbered bring-up ladder `scripts/0{1..6}_*.py` (each
> `--target real`), run **in order** — `01_check_dds` (read-only) and `02_check_image` are
> safe/no-motion, `03`–`05` command the arms/hands, `06` is camera-only. `scripts/hand_diag.py`
> is the low-level hand command→state diagnostic. The real image client (ZED) is **wired**;
> `02`/`06 --target real` need the ZED intrinsics + mount filled in `configs/camera_real.yaml`.

---

## Launch readiness — where we are
**The code path to a first cautious `home` on the robot is complete.** What's left is config
(the USER-PROVIDED fields below) and on-hardware tuning — **no code blockers remain.**

**Wired & sim-validated (code complete):**
- DDS arm/hand control (`debug` mode = `rt/lowcmd` + non-arm joints locked).
- `MotionSwitcher.Enter_Debug_Mode()` auto-runs on the real path (`mode != "sim"`).
- Velocity cap (`robot.yaml: arm_velocity_limit`) — a real, lowerable last-line-of-defence ceiling.
- Gravity-comp feed-forward (cuRobo RNEA, **off by default**, sign-validated — `docs/gravity_comp.md`).
- Perception: AprilTag `detect()` + head-camera client. Real ZED **wired** (stereo-slice, RGB-first,
  `camera_real.yaml`); sim is `camera_sim.yaml`; selected by `--target`.

**Needs the physical robot / camera:** fill the config table below, then walk the
`01 → 06 --target real` ladder; tune hand presets, the tracking abort threshold, and `gravity_scale`
on hardware. Nothing here is a code change.

---

## Config to fill in first (USER-PROVIDED)
| File | Field | What |
|---|---|---|
| `configs/robot.yaml` | `dds.interface` | NIC on this workstation wired to PC2 (e.g. `enp5s0`); set `mode: debug`. Real robot = domain 0; sim loopback = domain 1 / `lo`. |
| `configs/robot.yaml` | `home_q14_deg` | The launch/ready pose (default zeros = forearms forward). Confirm it's safe + reachable on the suspended robot. |
| `configs/hands.yaml` | `dex3.presets` (`open`/`power_close`/`pinch`) + `verify` thresholds | **Placeholders — untuned.** The right thumb stalls on `power_close` in sim; tune against the real hand. |
| `configs/camera_real.yaml` | `intrinsics` (fx,fy,cx,cy @1280×720) + `extrinsics.mount` + `stereo_side` | **Real head = ZED stereo.** Per-eye intrinsics for the chosen eye (ZED calibration) + the `d435_link`→eye `mount` offset (~half the stereo baseline) or detected block poses are laterally biased. |
| `configs/curobo/g1_dex3_curobo.yml` | `lock_joints` values (+ `velocity_scale`) | Leg+waist locked positions if the mount tilts the pelvis; lower `velocity_scale` to slow the robot for bring-up. |
| cuRobo world model | table / obstacles | **Not yet wired** — no world is configured. Add the table (and any obstacle) to the planner's world before relying on collision avoidance near it. |

---

## Robot interface bring-up
- [ ] **DDS connectivity** — `make_robot(connect_dds=True, dds_domain=0, dds_interface="<NIC>",
      mode="debug")` constructs the arm + Dex3 controllers; reading `arm.get_current_dual_arm_q()`
      returns live joint q from the suspended G1. For a READ-ONLY check first (no controllers,
      no motion), run `scripts/01_check_dds.py --target real`.
- [ ] **Debug mode — WIRED.** `make_robot(connect_dds=True, mode="debug")` now calls
      `MotionSwitcher.Enter_Debug_Mode()` automatically (any mode != "sim"; override with
      `enter_debug_mode=`) BEFORE the arm controller publishes `rt/lowcmd`. Confirm on hardware:
      the printed `Enter_Debug_Mode -> status=..., remaining active mode=...` shows no mode left
      active, and the suspended posture is the intended locked posture before first motion.
- [ ] **Hand state** — `python scripts/hand_diag.py --side left|right` opens/closes each hand and
      streams q / tau_est / press. **Confirm press pressures are non-zero and the closing-direction
      signs in `hands.yaml` are right** (`Dex3Controller` reads `motor_state.{q,dq,tau_est}` +
      `press_sensor_state.pressure`; verify against the real `HandState_`).

## Motion on hardware
- [ ] **Tracking** — run `home` then a small `move` on the robot. **Pass:** max tracking error
      stays under `executor.tracking_error_abort_rad` (0.20) with no aborts, and **no
      velocity-clip activations** in the arm controller (cuRobo's trajectory is the speed
      authority; if the clip fires, the cuRobo joint limits are too aggressive — treat as a bug).
- [ ] **Speed** — cuRobo plans ~1.1 rad/s in sim; if the real PD can't track it, lower the cuRobo
      `velocity_scale` in `configs/curobo/g1_dex3_curobo.yml` (no speed knob in `planner.yaml`).
      Separately, the controller's last-line-of-defence velocity clip is now config-driven — lower
      `robot.yaml: arm_velocity_limit` (e.g. 2–3 rad/s) for a cautious first run.
- [x] **Gravity comp — WIRED (off by default).** cuRobo RNEA `G(q)` feed-forward is plumbed
      (`planner.gravity_torque` → `executor._tauff`), gated by `configs/planner.yaml`
      `executor.gravity_comp` / `gravity_scale`. The **sign is hardware-validated in software**
      (matches the pinocchio convention proven on the real robot via `unitree_lerobot`'s
      `solve_tau`); magnitude ~15-20% above pinocchio's. **To enable on hardware:** set
      `gravity_comp: true`, ramp `gravity_scale` 0→1, confirm tracking error DROPS. Full recipe,
      validation, and caveats in **`docs/gravity_comp.md`**.

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

## Deferred (not on the MVP path)
- [ ] **Composite tasks** — pick (home → move-above → move-down → close → lift) and handover,
      composed from the primitives; add a grasp-frame (palm) offset so goals are grasp poses.
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
