# HARDWARE_TODO — what's left for the robot & camera

The **cuRobo-native MVP is sim-validated** (`home → move → close_hand → open_hand → home`
on `unitree_sim_isaaclab`, zero executor aborts; `home`/`move` collision-aware via cuRobo).
It has **never been exercised on the physical robot** — no real DDS/camera path has been run.
This file lists what still needs the physical G1 / Dex3 / head camera, with the command and
the pass criterion.

> Everything runs in the **`g1_curobo`** env (`bash -ic 'use_conda g1_curobo && …'`). The
> current hardware-capable entry points are `scripts/mvp_demo.py` and `scripts/hand_diag.py`.
> The older numbered scripts (`00_dds_echo`, `04_hand_check`, `run_pick_place`, …) predate the
> cuRobo-native refactor and **need porting** to the new `make_robot` / `primitives` API
> before use on hardware.

---

## Config to fill in first (USER-PROVIDED)
| File | Field | What |
|---|---|---|
| `configs/robot.yaml` | `dds.interface` | NIC on this workstation wired to PC2 (e.g. `enp5s0`); set `mode: debug`. Real robot = domain 0; sim loopback = domain 1 / `lo`. |
| `configs/robot.yaml` | `home_q14_deg` | The launch/ready pose (default zeros = forearms forward). Confirm it's safe + reachable on the suspended robot. |
| `configs/hands.yaml` | `dex3.presets` (`open`/`power_close`/`pinch`) + `verify` thresholds | **Placeholders — untuned.** The right thumb stalls on `power_close` in sim; tune against the real hand. |
| `configs/curobo/g1_dex3_curobo.yml` | `lock_joints` values (+ `velocity_scale`) | Leg+waist locked positions if the mount tilts the pelvis; lower `velocity_scale` to slow the robot for bring-up. |
| cuRobo world model | table / obstacles | **Not yet wired** — no world is configured. Add the table (and any obstacle) to the planner's world before relying on collision avoidance near it. |

---

## Robot interface bring-up
- [ ] **DDS connectivity** — `make_robot(connect_dds=True, dds_domain=0, dds_interface="<NIC>",
      mode="debug")` constructs the arm + Dex3 controllers; reading `arm.get_current_dual_arm_q()`
      returns live joint q from the suspended G1. (Port a minimal echo from `00_dds_echo.py`.)
- [ ] **Debug mode** — confirm `MotionSwitcher.Enter_Debug_Mode()` releases all modes before any
      `rt/lowcmd` publishing (the arm controller locks non-arm joints at current q). Verify the
      suspended posture is the intended locked posture before first motion.
- [ ] **Hand state** — `python scripts/hand_diag.py --side left|right` opens/closes each hand and
      streams q / tau_est / press. **Confirm press pressures are non-zero and the closing-direction
      signs in `hands.yaml` are right** (`Dex3Controller` reads `motor_state.{q,dq,tau_est}` +
      `press_sensor_state.pressure`; verify against the real `HandState_`).

## Motion on hardware
- [ ] **Tracking** — run `home` then a small `move` on the robot. **Pass:** max tracking error
      stays under `executor.tracking_error_abort_rad` (0.20) with no aborts, and **no
      velocity-clip activations** in the arm controller (cuRobo's trajectory is the speed
      authority; if the clip fires, the cuRobo joint limits are too aggressive — treat as a bug).
- [ ] **Speed** — cuRobo plans ~1.1 rad/s in sim; if the real PD can't track it, add a
      conservative joint velocity limit / velocity scale in `configs/curobo/g1_dex3_curobo.yml`
      (there is no longer a speed knob in `planner.yaml`).
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

## Deferred (not on the MVP path)
- [ ] **Perception** — `perception/*` is dormant (pinocchio-bound); rework cuRobo-native
      (AprilTag or sim ground-truth → block pose in pelvis frame) before any perceive-driven task.
- [ ] **Camera** — the URDF carries `d435_link` (head, FK-derived extrinsics); confirm the
      body→optical convention once perception is back.
- [ ] **Composite tasks** — pick (home → move-above → move-down → close → lift) and handover,
      composed from the primitives; add a grasp-frame (palm) offset so goals are grasp poses.
- [ ] **Rerun logging** — wire current q / target pose / state into the primitives for debugging.

---

## Offline status (verified, no robot)
- `pytest tests/` → **12 passed** (`test_pose` SE(3) conventions, `test_grasp` verification).
- cuRobo FK parity vs the old pinocchio model: **< 0.001 mm / 0.027°** (cuRobo is a faithful
  drop-in); `Pose` util convention-faithful to ~1e-15.
- cuRobo → sim execution: EE position error ~0.45 cm, native plan ~1.1 rad/s, zero aborts.
- Every active module imports under the `g1_curobo` env.

## Environment notes
- Run via `bash -ic 'use_conda g1_curobo && <cmd>'` (lane wrapper sets `PYTHONNOUSERSITE=1`).
- **No pinocchio / no numpy<2 pin** — kinematics are cuRobo's. Build is not plain
  `pip install`; see `requirements-curobo.txt`. Sim target = `unitree_sim_isaaclab` (Isaac
  Sim in the `unitree` env), loopback DDS. See `SIM_NOTES.md`.
