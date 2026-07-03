# Gravity-compensation feed-forward — findings & how it's wired

**Status:** wired, **ON by default for REAL hardware** (forced OFF in sim by `factory.py`),
gravity-only, **sign + necessity confirmed on the physical robot 2026-06-17**. Config:
`configs/planner.yaml` → `executor.gravity_comp: true`, `gravity_scale: 1.0`; per-run
override via `connect(gravity_comp=, gravity_scale=)` or `checks/05_move --gravity-scale`.

## TL;DR
- The executor can feed a per-joint feed-forward torque to the arm motors so the PD
  doesn't fight gravity alone (which causes droop + tracking error on the real robot;
  in sim it's unnecessary).
- That torque is **cuRobo's own RNEA inverse dynamics** — `G(q) = RNEA(q, q̇=0, q̈=0)`.
  No pinocchio; it builds from the same `KinematicsParams` the planner already holds.
- We confirmed the **sign** is correct **without the robot**, by matching it against the
  pinocchio `pin.rnea` feed-forward that already runs on the physical G1
  (`unitree_lerobot`'s `solve_tau`). All joints, all test poses: signs match, no flips.
- Magnitude runs **~15–20% above pinocchio's** (consistent scale, same URDF) — fine for
  feed-forward; the PD covers the residual.

## Why it was needed
`Executor._tauff` returned `np.zeros(14)`. In sim that's fine (no real gravity to fight).
On hardware, zero feed-forward means the position PD (`kp_low=80, kp_wrist=40`) holds the
arm against gravity by **steady-state position error** → the arm sags (a few degrees) and
tracking error rises, eating into the `tracking_error_abort_rad=0.20` budget. Gravity-comp
feed-forward removes the bulk of that.

## cuRobo V2 has RNEA inverse dynamics
Confirmed in source: `curobo/_src/robot/dynamics/dynamics.py` → `class Dynamics`, native
CUDA RNEA (forward + differentiable backward). Built **from cuRobo's `KinematicsParams`**
(link masses/COMs/inertias) — "no URDF export and no external library." Gravity vector is
configurable (default `[0,0,-9.81]`).

```python
from curobo._src.robot.dynamics.dynamics import Dynamics
from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
dyn = Dynamics(DynamicsCfg(kinematics_config=kp, device_cfg=device_cfg))  # kp = planner kinematics
dyn.setup_batch_size(1)
tau = dyn.compute_inverse_dynamics(JointState(position=q, velocity=q̇, acceleration=q̈))
# gravity-only: q̇ = q̈ = 0  → tau = G(q)
# full computed-torque: pass the trajectory's q̇, q̈ → tau = M q̈ + C q̇ + G
```

## How it's wired here
- **`CuroboArmPlanner.gravity_torque(q_repo14) -> (14,) Nm`** (`motion/planner.py`).
  Lazily builds `Dynamics` from `self._mp.kinematics.kinematics_config` (14-DoF arms, base
  locked), calls `compute_inverse_dynamics` with `q̇=q̈=0`, reorders cuRobo→repo
  (left7+right7). Lazy build ⇒ the default (off) path pays nothing.
- **`Executor._tauff(q)`** (`motion/executor.py`): returns `np.zeros(14)` unless
  `gravity_comp` is on, then `gravity_scale * planner.gravity_torque(q)`. The torque path
  to the motors already existed (`ctrl_dual_arm(q, tauff)` → `motor_cmd[id].tau`), so this
  is the only change. Applied uniformly in `run` / `hold` / `settle` / `prime`.
- **Config** (`configs/planner.yaml` → `executor`): `gravity_comp: true`,
  `gravity_scale: 1.0`. The factory passes these + the `planner` into the `Executor`, and
  **forces `gravity_comp` OFF in sim** (`mode: sim`) unless explicitly overridden. Per-run
  overrides: `connect(gravity_comp=, gravity_scale=)`, surfaced as `checks/05_move --gravity-scale`.
- **Gravity-only**, matching the proven `solve_tau` approach (full computed-torque is a
  later option — pass the trajectory's `q̇/q̈`).

## Validation (software-only; no robot)
Two scratch spikes (in `~/repos/curobo_g1_smoke/`): `grav_spike.py` (cuRobo `G(q)`),
`pin_vs_curobo_grav.py` (pinocchio cross-check on the **same** `g1_29dof_mode_16_dex3.urdf`).

**1. cuRobo `G(q)` is sane.** At home (arms 0 = forearms forward, elbows bent 90°):
`L[-4.93, 0.23, 0.0, -4.65, -0.07, -1.79, 0.0]`, right arm mirrored. Checks:
- Magnitude physical: shoulder_pitch 4.93 Nm ≈ (arm+hand ≈2.5 kg)·9.81·(COM ≈0.2 m). ✓
- Left/right exactly symmetric; shoulder/elbow ~5 Nm, wrist_pitch ~1.8 Nm, roll/yaw ~0. ✓
- Configuration-dependent + monotonic: L shoulder_pitch torque `-7.97 → -4.93 → -0.17`
  as pitch sweeps `-0.6 → 0 → +0.6` (through gravity-neutral); arms decoupled. ✓

**2. Sign confirmed against the hardware-proven convention.**
`unitree_lerobot/.../robot_control/robot_arm_ik.py::solve_tau` runs
`pin.rnea(model, data, q, 0, 0)` and feeds it **straight to the motors, no sign flip, no
reindex**, every frame on the real robot. So pinocchio's URDF-axis `G(q)` == the motor
torque convention (hardware-proven). The chain:
- `pin.rnea(q,0,0)` → motors works on hardware ⟹ pinocchio convention == motor convention.
- cuRobo FK ≡ pinocchio FK (validated earlier to <0.001 mm) ⟹ identical joint-axis convention.
- cuRobo `G(q)` matches pinocchio `G(q)` **sign per-joint, all 6 poses** (cross-check) ⟹
  cuRobo convention == motor convention. **∴ `+G(q)` straight to the motors is correct.**

**3. Magnitude gap.** cuRobo runs **~15–20% higher** than pinocchio (worst per-joint
|cuRobo − pinocchio| = 1.31 Nm), a *consistent scale*, not structural/sign. Both read the
same URDF, so the gap is how each derives link inertials (cuRobo's config likely carries
its own baked inertial params). Negligible for feed-forward; flagged below.

## Hardware validation (2026-06-17) — confirmed on the physical G1
First real-robot bring-up of the launch home (arms power-on folded). Debug mode set by the
**operator via the physical remote** (we do NOT call `MotionSwitcher` — releasing it dropped
the robot out of low-level control). Ramped `gravity_scale` via `checks/05_move --gravity-scale`,
reading the launch-home **per-joint residual** (`connect` prints it):

| `gravity_scale` | `arm_velocity_limit` | max residual | elbow residual | result |
|---|---|---|---|---|
| 0.0 (off)       | 20 | 50° | ~50° (forearm sags) | `home` fails |
| 0.5             | (live) | 37° | ~34° | sign **correct** (droop shrinking) |
| 1.0             | 5  | 9°  | ~3° | elbows fixed; **shoulder pitch** stalls ~9° |
| 1.0             | 12 | **2.7°** | ~0.5° | **`home` ok=True** ✅ |

Takeaways:
- **Sign is correct** (positive `G(q)` straight to motors) — the software/pinocchio
  validation held on hardware. Droop shrank monotonically as scale rose 0 → 1.
- **`gravity_scale: 1.0` is right** — at 1.0 the elbow droop is gone; the ~15–20% over-estimate
  did not cause runaway lift.
- **Velocity cap ↔ torque coupling (important).** Gravity comp only *holds* the arm; the PD
  must still overcome static friction. The velocity clip (`clip_arm_q_target`) caps the
  position error the PD ever sees to `arm_velocity_limit · control_dt`, so it caps PD torque
  at `≈ kp · arm_velocity_limit · control_dt`. At `arm_velocity_limit=5` that is only
  `80·5·0.004 ≈ 1.6 N·m` — below shoulder-pitch static friction, so the shoulders stalled ~9°
  short. Raising to 12 (`≈ 3.8 N·m`) closed it. **Keep `arm_velocity_limit ≥ ~12`.** cuRobo
  plans at ~1.1 rad/s, so a cap of 12 is still only a backstop for planned moves — it does not
  speed them up; it just unstarves holding/correction torque. (Decoupling speed from torque in
  the clip is a future cleanup.)

## Re-validating / sign re-check (procedure)
`gravity_comp` is now the default for real. To re-confirm (e.g. after a model change), drop
it with `checks/05_move --target real --gravity-scale 0.5` and watch the elbow residual: it should
**shrink** vs the off (`--gravity-scale 0.0`) run. If it **grows**, the sign flipped — re-run
with a negative scale and negate `gravity_torque` in the planner. Then ramp to 1.0.

## Open items / caveats
- **Magnitude source.** Resolve the ~15–20% cuRobo-vs-pinocchio gap if precise torques
  matter: check whether cuRobo's `.yml` inertials match the URDF `<inertial>` tags. For
  gravity comp it doesn't block anything.
- **Full computed-torque.** For faster moves, pass the trajectory's `q̇/q̈` into
  `compute_inverse_dynamics` instead of zeros (cancels inertial + Coriolis too). Start
  gravity-only (proven) before this.
- **Per-cycle GPU cost — measured fine.** `_tauff` calls `compute_inverse_dynamics`
  (batch=1, GPU sync) each control tick. Measured **0.13 ms/call** on the RTX 5090 — ~3%
  of the 4 ms period at 250 Hz, so per-cycle is fine and no precompute is needed. (If a
  future change tightens timing, the fallback is to precompute the feed-forward batched
  along the trajectory in `run()` and sample it like `q`.)
- **Sign re-verify** if the URDF joint axes ever change (e.g. a re-calibrated model):
  re-run `pin_vs_curobo_grav.py`.
- **Gravity direction** assumes the torso is upright (base locked at 0). If the back-plate
  mount tilts the pelvis, set the locked reference / gravity vector accordingly.

## References
- cuRobo: `curobo/_src/robot/dynamics/dynamics.py` (`Dynamics.compute_inverse_dynamics`),
  `dynamics_cfg.py`, `benchmark/inverse_dynamics_kernel_benchmark.py` (construction recipe).
- Hardware-proven feed-forward: `unitree_lerobot/unitree_lerobot/eval_robot/robot_control/robot_arm_ik.py::solve_tau`.
- Our code: `motion/planner.py::gravity_torque`, `motion/executor.py::_tauff`,
  `configs/planner.yaml::executor`.
- Spikes: `~/repos/curobo_g1_smoke/{grav_spike.py, pin_vs_curobo_grav.py, wire_check.py}`.
