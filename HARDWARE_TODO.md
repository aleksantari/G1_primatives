# HARDWARE_TODO — what's left for the robot & camera

Everything in this repo is **code-complete and offline-validated** (Phases 0–6 + the
Phase-7 cuRobo seam as a stub). This file lists every item that still needs the physical
G1 / Dex3 / head camera to finish or verify, grouped by phase, each with the exact command
and the pass criterion from `G1_CLASSICAL_MANIP_PLAN.md`.

The robot was **not available** during this build, so no DDS/camera path was exercised on
hardware. All offline checks (`pytest`, IK, transforms, planner→retimer, FSM graph) pass —
see "Offline status" at the bottom.

---

## Config to fill in first (USER-PROVIDED)
| File | Field | What |
|---|---|---|
| `configs/robot.yaml` | `dds.interface` | NIC on this workstation wired to PC2 (e.g. `enp5s0`). Real robot domain 0; unitree_sim_isaaclab loopback domain 1 / `lo`. |
| `configs/robot.yaml` | `model.locked_reference_deg` | Leg+waist posture if the back-plate mount tilts the pelvis (default zeros). |
| `configs/camera.yaml` | `intrinsics` | fx, fy, cx, cy, distortion from the head RealSense / image server. |
| `configs/camera.yaml` | `extrinsics.body_to_optical` / `extrinsic_correction` | Verify the `d435_link` frame convention; refine hand-eye if needed (see Phase 3). |
| `configs/task_pick_place.yaml` | `table`, `workspace_box`, `place`, `home_q14_deg` | Calibrate the table plane (script 05) and the real reachable workspace. |
| `configs/hands.yaml` | `dex3.presets` (`open`/`power_close`/`pinch`) + `verify` thresholds | **Placeholders** — tune on hardware (Phase 4). |

> Note: camera **extrinsics are derived by FK to the URDF `d435_link`** (fixed to torso),
> so you do NOT hand-measure `T_pelvis_camera`; you only confirm the frame convention and
> optionally add a small `extrinsic_correction`.

---

## Phase 1 — Robot interface bring-up
- [ ] **DDS connectivity** — `python scripts/00_dds_echo.py --domain 0 --interface <NIC>`
      → live dual-arm q prints at 1 Hz from the suspended G1.
- [ ] **Debug mode** — confirm `MotionSwitcher.Enter_Debug_Mode()` releases all modes before
      any `rt/lowcmd` (the arm controller locks non-arm joints at current q). Verify the
      suspended posture is the intended locked posture before first motion.
- [x] **Sim smoke (done in Isaac)** — arm path planning is verified in
      `unitree_sim_isaaclab` (`scripts/01_sim_arm_smoke.py --isaac`, ~0.13 rad tracking,
      zero aborts; see SIM_NOTES.md). On hardware, run `01_sim_arm_smoke.py` (no `--isaac`)
      → homes and executes the pick cycle without limit violations.
- [ ] **Dex3 state** — `python scripts/04_hand_check.py --side left|right`
      → opens/closes each hand; prints q / tau_est / press streams. **Confirm the press
      sensor pressures are non-zero and that closing direction signs in `hands.yaml` are
      right** (the threaded `Dex3Controller` reads `motor_state.{q,dq,tau_est}` +
      `press_sensor_state.pressure`; verify against the real `HandState_`).

## Phase 2 — Motion stack on hardware
- [ ] **Tracking + speed authority** — execute home→hover→descend→lift→home on the robot
      (`run_pick_place.py` up to LIFT, or `01_sim_arm_smoke.py`).
      **Pass:** max tracking error < 0.05 rad and **zero velocity-clip activations** in the
      arm controller (the Ruckig retimer must be the speed limiter, not `clip_arm_q_target`).
      The executor aborts-to-hold past `executor.tracking_error_abort_rad`.
- [ ] **Gravity comp** — confirm `Executor` RNEA feed-forward torque holds posture without
      sag; tune `kp/kd` in `robot_arm.py` if needed (currently your G1_teleop_dex gains).

## Phase 3 — Perception
- [ ] **Image server** — confirm which server runs on PC2. This repo's `HeadCamera` defaults
      to the **`teleimager`** backend (your deployed stack); `image_client.py` also has a
      `unitree_lerobot` ZMQ backend. Set `--backend` / host accordingly.
- [ ] **Camera frame convention** — `python scripts/02_view_camera.py --host <PC2>`
      then `03_static_perception_check.py`. The URDF `d435_link` is treated as a camera
      *body* frame with a ROS body→optical rotation (`camera.yaml: body_to_optical: ros`).
      **Verify**: a tag at a known pelvis-frame location reports the correct pose; if the
      axes are off, switch to `identity` or add an `extrinsic_correction`.
- [ ] **Accuracy** — `python scripts/03_static_perception_check.py --tag 0`
      **Pass:** hand-measured 10.0 cm tag steps report pelvis-frame deltas within ±5 mm.
- [ ] **Tag geometry** — set `perception.yaml: tag.size_m` and `tag_to_block` to the printed
      tag and block.

## Phase 4 — EE layer + grasp verification
- [ ] **Tune presets** — `04_hand_check.py`: command `open`/`power_close`/`pinch`, read back
      q, fix the 7-vectors and closing-direction signs in `hands.yaml`.
- [ ] **Tune thresholds** — set `verify.{stall_margin_rad, tau_threshold, press_threshold,
      still_dq}` so closing on the block → `grasped=True` and closing on air → `False`.
      **Pass:** 10/10 each.

## Phase 5 — Pick & place
- [ ] `python scripts/05_reach_check.py` first (moves to hover and STOPS — calibrate table
      plane / grasp offset safely), then `python scripts/run_pick_place.py`.
      **Pass:** ≥ 8/10 pick-place cycles with the block randomly placed in a 20×20 cm region;
      zero executor aborts; failures auto-detected (grasp verify / perception), never silent.
- [ ] **Grasp-pose conditioning** — the offline retimer flags `near_singular` (`max_time_scale`
      > 3×) when a grasp pose drives the arm near a wrist singularity (slow, safe crawl).
      Pick a grasp orientation / block region that stays well-conditioned, or move to cuRobo.

## Phase 6 — Arm-to-arm handover
- [ ] Tune `configs/task_handover.yaml` (handover pose pair, separations, handshake timings),
      then `python scripts/run_handover.py`.
      **Pass:** ≥ 7/10 full pick→handover→place-with-receiver; the "receiver missed, giver
      still holding" case recovers to a retry rather than dropping (enforced in code: the
      giver never opens unless the receiver grasp is verified).

## Phase 7 — cuRobo (not started)
- [ ] Install cuRobo (`cu12-torch`, torch ≥ 2.5) into the env; implement
      `motion/curobo_planner.py` against the same `JointPath` contract (it's a stub now).
      Build a fixed-base dual-arm config (lock legs+waist; no floating `extra_links`).
      **Pass:** `planner: curobo` in `configs/planner.yaml` passes the Phase-5 test unchanged,
      and a deliberate obstacle between home and pregrasp is avoided.

---

## Offline status (verified during this build, no robot)
- `pytest tests/` → **28 passed** (IK, transforms, planner continuity, retimer limits, FSM
  graph + retry/abort, grasp verification, AprilTag median/staleness).
- IK builds the 14-DoF dual-arm model from `assets/g1/g1_body29_dex3.urdf` and solves to
  **~0.19 mm** translation error (Phase-0 acceptance, runs offline).
- `python scripts/offline_plan_check.py` → plans + retimes home→hover→descend→lift→home:
  continuity 0.097 rad, max|qd| 1.57 ≤ 2.0, max|qdd| 4.85 ≤ 5.0, returns home 0.1 mm — **OK**.
- Every package module imports under the `g1_classical_manip` conda env.

## Environment notes (gotchas baked into the env)
- Run via `bash -ic 'use_conda g1_classical_manip && <cmd>'`.
- **numpy pinned `<2`** (1.26.4) — pinocchio 3.1.0 ABI; **rerun-sdk pinned `==0.20.1`**
  (0.29 requires numpy≥2). Do not blindly `pip install -U` these.
- Sim target is **`unitree_sim_isaaclab`** (Isaac Sim 5.0 in the `unitree` env) — arm path
  planning verified there over loopback DDS. See `SIM_NOTES.md`.
