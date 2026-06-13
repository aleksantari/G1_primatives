# g1_classical_manip

Classical (non-learned) **pick → place → arm-to-arm handover** for the Unitree G1
(29-DoF, Dex3-1 hands), running **off-board** on an RTX 5090 workstation that talks to the
robot's PC2 over CycloneDDS. The robot is **suspended on a back-plate mount**: only the
14 arm joints and 2×7 hand joints are ever commanded.

- **Perception v1:** AprilTag on the head RealSense (d435).
- **Planner v1:** Cartesian waypoints + the stock dual-arm IK (Pinocchio/CasADi/Ipopt).
- **Planner v2 (Phase 7):** cuRoboV2 behind the same `Planner` interface.

The authoritative design is **`G1_CLASSICAL_MANIP_PLAN.md`**. This README is the quickstart;
**`HARDWARE_TODO.md`** lists everything still gated on the physical robot/camera.

## Three-layer motion stack (hard seams)
```
Planner.plan(start_q14, goal, world) -> JointPath      # geometry, no timing
Retimer.retime(JointPath)            -> JointTrajectory # Ruckig: t,q,qd,qdd
Executor.run(JointTrajectory)                            # streams @250 Hz, owns the controller
```
The Cartesian planner and cuRobo emit the **same `JointPath`**. IK never streams directly.

## Layout
```
g1_classical_manip/
  robot_control/   arm controller, dual-arm IK, threaded Dex3/Dex1, motion switcher  (vendored+adapted)
  image_server/    image client (teleimager-compatible)
  perception/      transforms.py (all frame math), apriltag_block.py
  motion/          planner_base, cartesian_planner, retimer, executor, curobo_planner (stub)
  ee/              hand_base, dex3, dex1 (grasp presets + verification)
  tasks/           fsm, primitives (move/pick/place/handover), pick_place_handover
  utils/           rerun viz, episode recorder, weighted moving filter
  factory.py       make_robot()-style registry (arm/hand/planner from configs)
configs/           robot, camera, perception, task_pick_place, task_handover, planner (YAML)
scripts/           00_dds_echo .. 05_reach_check, run_pick_place, run_handover
tests/             pure-math unit tests (no robot needed)
```

## Environment
Runs in the **`g1_classical_manip`** conda env (cloned from `tv`; Python 3.10):
pinocchio 3.1.0, casadi, meshcat, unitree_sdk2py + `ruckig`, `pupil-apriltags`,
`rerun-sdk==0.20.1`, numpy **pinned <2** (pinocchio ABI).

```bash
bash -ic 'use_conda g1_classical_manip && pytest tests/'
bash -ic 'use_conda g1_classical_manip && python scripts/offline_plan_check.py'   # plan->retime, no robot
```

## Status
Phases 0–6 are code-complete and offline-validated (IK, transforms, retimer, FSM, plan→retime).
Phase 7 (cuRobo) is a stubbed seam. All DDS/camera/robot acceptance tests are deferred — see
`HARDWARE_TODO.md`.
