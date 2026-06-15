# CLAUDE.md — conventions digest for g1_classical_manip

Read `G1_CLASSICAL_MANIP_PLAN.md` (source of truth) and `HARDWARE_TODO.md` first.

## Environment
- Conda env **`g1_classical_manip`** (cloned from `tv`, Python 3.10). Always run via
  `bash -ic 'use_conda g1_classical_manip && <cmd>'` (lane wrapper; `.bashrc` must be sourced).
- **numpy is pinned `<2`** (1.26.4): pinocchio 3.1.0 is built against numpy 1.x and segfaults/
  errors (`_ARRAY_API not found`) under numpy 2. **rerun-sdk is pinned `==0.20.1`** because
  0.29 hard-requires numpy≥2. Do not `pip install -U` either without re-checking this.

## Architecture rules (enforced)
1. **Three-layer motion stack:** `Planner.plan -> JointPath` (geometry, no timing) →
   `Retimer.retime -> JointTrajectory` → `Executor.run`. The Cartesian planner and cuRobo
   emit the **same `JointPath`** dataclass. IK never streams to the robot; the executor owns
   the only handle to `G1_29_ArmController`.
2. **All poses are `pin.SE3` in the pelvis frame.** `perception/transforms.py` is the ONLY
   file allowed to construct frame conversions. Camera extrinsics come from FK to the URDF
   `d435_link` (+ optional `extrinsic_correction` in `camera.yaml`), NOT hand-measured numbers.
3. **Dual-arm state is one 14-vector everywhere** (left 7 + right 7, upstream joint order:
   `G1_29_JointArmIndex` 15–28). Single-arm motion = hold the other arm's current q as its IK
   target; never slice the controller.
4. **Config-driven:** task waypoints, speeds, grasp presets, planner choice all live in
   `configs/*.yaml`. `planner: cartesian → curobo` is a one-line switch.
5. **Rerun on every FSM transition** (current q, target pose, perceived block, state name).
6. **Safety:** conservative `arm_velocity_limit`; executor aborts-to-hold past tracking-error
   threshold; every script starts and ends at home; workspace-box check before planning.

## Key facts (verified, don't re-derive)
- Primary model: `assets/g1/g1_body29_dex3.urdf` → lock legs+waist+14 hand joints = **14-DoF**
  dual-arm reduced model. EE frames `L_ee`/`R_ee` = wrist-yaw + `[0.05,0,0]`. IK ~0.2 mm.
- The URDF carries `d435_link` (head camera, fixed to torso) and `left/right_hand_palm_link`
  (grasp frames). `transforms.py` gets camera + palm transforms via FK — no manual extrinsics.
- `HandState_` exposes `motor_state[].{q,dq,tau_est}` and `press_sensor_state[].pressure`
  (7 each per hand) → grasp verification signals. The threaded `Dex3Controller` reads all four.
- Arm controller: debug mode = `rt/lowcmd` + locks non-arm joints at current q (suspended-robot
  correct); motion mode = `rt/arm_sdk`. `simulation_mode` bypasses the velocity clip.
- DDS domain/interface live ONLY in `configs/robot.yaml`. Real robot = (0, NIC-to-PC2);
  unitree_sim_isaaclab on loopback = (1, "lo"); see SIM_NOTES.md.

## Gotchas
- Call `MotionSwitcher.Enter_Debug_Mode()` before any `rt/lowcmd` publishing on hardware.
- Camera `d435_link` is a body/mount frame; AprilTag returns OPTICAL-frame pose. Apply the
  `body_to_optical` rotation (`camera.yaml`) and verify the convention on hardware.
- If the controller's velocity clip activates during nominal execution, the retimer limits are
  wrong — treat as a bug, not a safety save.
