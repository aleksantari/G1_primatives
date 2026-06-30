# Trajectory speed vs. tracking — the `time_dilation` crutch and the `acceleration_scale` root fix

**Status:** OPEN / deferred. Diagnosed in sim 2026-06-29 on the depth-ESDF collision-world
grasp approach; **the workaround (`time_dilation`) is in place, the root fix (gentler plan via
`acceleration_scale`) is not.** Revisit before/while testing on real hardware — see
[[trajectory-speed-tracking]] in memory.

## TL;DR
- cuRobo plans to **stock-aggressive dynamics** — `g1_dex3_curobo.yml` cspace:
  `max_acceleration = 10 rad/s²`, `max_jerk = 500 rad/s³`, `velocity_scale = acceleration_scale =
  jerk_scale = 1.0` (cuRobo defaults, no de-rating).
- The collision-world **approach** route (curves around the obstacle, unlike a straight reach) is
  dynamic enough that the controller **can't track it at full-speed playback** → the executor's
  tracking-error abort fires.
- We currently paper over this with **`executor.time_dilation`** — playing cuRobo's
  already-feasible trajectory back **slower** (`< 1.0`). It works but is a band-aid: it
  re-rate-limits a plan cuRobo already called feasible.
- The **root fix** is model-level: lower `acceleration_scale` (and/or `velocity_scale`) in the
  cuRobo cspace so the **plan itself** is gentle enough to execute at `time_dilation = 1.0`. Then
  both targets track without per-target speed hacks. (This is the "deferred root fix" CLAUDE.md
  already references.)

## Symptom (observed)
Sim, `09_graspgen --collision-world` (full pipeline, real-grasp execution gated):
```
grasp_motion: approach: tracking error 0.401 > 0.400 rad at traj_t=0.58s
GRASP-GEN ABORTED (RuntimeError): approach: tracking error 0.401 > 0.400 rad at traj_t=0.58s
```
The trajectory **planned fine** — this is an *execution* abort, not a planning failure. The error
is a **hair** over threshold (`0.401 > 0.400`), i.e. marginally infeasible, not grossly broken.

## Diagnosis (confirmed)
`09_graspgen --speed 0.5` (slow the SAME trajectory's playback by half) → the approach **tracks
clean, no abort**. So the cause is the planned **dynamics being too hot for the controller at
full-speed playback**, not the robot model geometry, the self-filter, or the thumb-lock change
(none of which touch the 14-arm-joint trajectory). This is exactly the `build_robot_model`
surface: <https://nvlabs.github.io/curobo/latest/getting-started/build_robot_model.html>
(`velocity_scale` / `acceleration_scale` are the guide's knobs for matching the plan to what the
hardware can execute).

## The sim ↔ real asymmetry (why it surfaced in sim, and why real is NOT safe-by-default)
| | Sim | Real |
|---|---|---|
| `time_dilation` (playback) | **1.0** (factory forces it; sim bypasses the velocity clip) | **0.5** (planner.yaml default) |
| abort threshold (`09`) | **0.40** (loosened for sim in `09_graspgen.py`) | **0.20** (executor default) |
| arm velocity clip (`clip_arm_q_target`, caps PD torque, measured-relative) | **bypassed** (`simulation_mode`) | **active** — the reason real needs 0.5 |

Read the table carefully — the naïve conclusion ("sim-only") is wrong:
- Sim aborted **at full speed with a 2× looser budget (0.40)** and the clip bypassed. That's the
  *easiest* possible tracking case, and it still failed → the plan is genuinely too dynamic.
- Real runs the same plan at **0.5** (= the `--speed 0.5` that fixed sim), so this approach has a
  good chance of tracking on real **for free**. BUT real also has the **tighter 0.20 budget** and
  the **active torque clip** — real is the *harder* target, not the easier one. A trajectory that
  defeats full-speed sim would defeat full-speed real outright; real only survives via the 0.5
  crutch, and the 0.20 budget gives it less margin than sim had.

**Bottom line:** real is not guaranteed to pass just because it defaults to 0.5. Validate it, and
prefer the model-level fix so neither target leans on the playback crutch.

## The two levers
1. **`executor.time_dilation`** (workaround, in place). `planner.yaml: executor.time_dilation`;
   per-run `09_graspgen --speed` / `04_move --speed`. Slows playback of the *same* path. Sim is
   forced to 1.0 in `factory.py`; real defaults to 0.5. **Tuning sim and real means tuning this
   per target.**
2. **`acceleration_scale` / `velocity_scale`** in `configs/curobo/g1_dex3_curobo.yml` cspace
   (root fix, NOT done). Lowering these makes cuRobo **plan** a gentler, dynamically-feasible
   trajectory — the cuRobo-native equivalent of `--speed`, applied at plan time. The abort is
   marginal (0.401 vs 0.400), so a modest reduction (e.g. `1.0 → 0.6–0.7`) should clear it.

## Why we parked it (the sim/real interplay caveat)
`acceleration_scale` lives in the **shared** robot config → it affects **both** targets. On real
the executor *also* applies `time_dilation = 0.5`, so naively lowering `acceleration_scale` would
**compound** with that 0.5 → very slow real motion. The correct change is to retune the pair
**together**: lower `acceleration_scale` so the plan is feasible at `time_dilation = 1.0`, then
raise real's `time_dilation` back toward 1.0 (dropping the crutch) — validated on the physical
robot. That's a deliberate, hardware-gated change, not a one-liner to slip into the
hardware-validated config.

## When this becomes a problem again (watch list)
- **On real hardware:** if planned grasps/approaches abort with `tracking error > 0.20` (the real
  budget). First confirm with `--speed` (lower it); if that fixes it, it's this issue.
- **Tighter obstacle routing:** a more contorted collision-world approach is more dynamic; even
  real's 0.5 may not be enough → the model fix becomes necessary, not optional.
- **If you want faster real motion:** you can't just raise `time_dilation` toward 1.0 without
  first lowering `acceleration_scale`, or tracking diverges (this is the whole point).

## Pointers
- Symptom/abort logic: `g1_classical_manip/motion/executor.py` (`abort_thresh`, `time_dilation`).
- Sim overrides: `09_graspgen.py` (`--speed`, `--abort`; sim sets `abort_thresh = 0.40`).
- Model dynamics: `configs/curobo/g1_dex3_curobo.yml` cspace; recipe
  `configs/curobo/build_g1_dex3.py`.
- Related: `docs/gravity_comp.md` (the other half of real-arm trackability — feed-forward torque),
  `HARDWARE_TODO.md`, CLAUDE.md "cuRobo native plan ≈ 1.1 rad/s" key fact.
