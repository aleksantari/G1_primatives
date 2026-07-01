# `09_graspgen` pipeline — end-to-end call map

**What:** how the main grasp script (`scripts/09_graspgen.py`) runs the whole pick-and-lift, from
process wiring through grasp generation, native cuRobo `plan_grasp`, and streamed execution. Covers
the **native** path (default); `--legacy` (sequential `plan_to_pose` probe + manual offsets) is the
A-B baseline and is noted where it diverges. File:function references are clickable in the repo.

## TL;DR
`main()` wires the robot (planner + arm + hand + executor + camera + grasp source), applies CLI
overrides, then runs an operator-gated sequence: **home → open → `grasp_source.grasps()` →
`grasp_motion()` → home**. `grasps()` turns depth/cloud into ranked wrist-yaw `GraspCandidate`s (via
SAM3 + GraspGenX for the real path); `grasp_motion()` optionally builds a depth-ESDF collision world,
solves a cuRobo `plan_grasp` goalset, and streams the approach/grasp/lift segments through the
executor (which aborts-to-hold past the tracking-error threshold).

## High-level flow

```
                        scripts/09_graspgen.py : main()
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │ 1. WIRING   _rig.connect(target)  → factory.make_robot │
        └───────────────────────────┬───────────────────────────┘
                                    │ builds the Robot dataclass:
   ┌──────────────┬─────────────────┼───────────────┬──────────────┬───────────────┐
   ▼              ▼                 ▼               ▼              ▼               ▼
CuroboArm     G1_29_Arm        Dex3Hand        Executor      HeadCamera      GraspSource
 Planner      Controller      (Dex3Controller)               (image_client)  (_build_grasp_source)
(curobo_      (robot_arm.py)  (robot_hand_     (executor.py)  rgb :55555      apriltag│graspgenx│sim_cloud
 planner.py)   rt/lowcmd       unitree.py)                     depth :55556    (grasp/*_source.py)
  warmup()     rt/lowstate     rt/dex3/*/cmd                   (sim) / ZED
  + grasp MP                   rt/dex3/*/state                 :56555 (real)
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │ 2. CLI OVERRIDES (main, lines 152-178)                 │
        │   --source/--segment/--visualize → _build_grasp_source │
        │   --abort / sim→0.40 → executor.abort_thresh           │
        │   --speed → executor.time_dilation                     │
        │   --grasp-only → strategies = [no approach/no lift]    │
        │   --collision-world → planner.set_collision_world(True)│
        └───────────────────────────┬───────────────────────────┘
                                    │
        ┌───────────────────────────▼───────────────────────────┐
        │ 3. PICK SEQUENCE  (each step operator-gated _rig.confirm)│
        └───────────────────────────┬───────────────────────────┘
                                    │
   P.home(robot) ──────────► planner.plan_joint(q, home) ─► executor.run(traj)   [primitives.py:home]
   P.open_hand(robot,side) ─► hand.open(side) ─► Dex3Controller cmd              [primitives.py:open_hand]
   cands = robot.grasp_source.grasps(robot, side, object)   ◄── ❰SUBSYSTEM A❱
   res  = P.grasp_motion(robot, side, cands, close_cb, ...)  ◄── ❰SUBSYSTEM B❱
   P.home(robot)
```

## Subsystem A — grasp generation (`grasp_source.grasps()`)

```
robot.grasp_source.grasps(robot, side, "block")              [grasp/base.py: GraspSource ABC]
   │   09 is GraspGenX-only (--source graspgenx | sim_cloud). AprilTag stays a GraspSource
   │   (grasp/apriltag_source.py) but is the 07_pick_place baseline — not selectable here.
   │
   ├── graspgenx  (grasp/graspgenx_source.py: grasps)         ── REAL / live depth
   │     cam.get_rgb_frame() + cam.get_depth_frame()          [image_server/image_client.py]
   │     frames.T_pelvis_camera(q14)                          [perception/transforms.py]
   │     segmenter.mask(rgb) ──ZMQ :5557──► SAM3 server       [perception/segment.py, sam3_client.py]
   │     deproject_depth(depth, K, T_pc, mask, rgb) → PointCloud (pelvis)   [perception/depth.py]
   │     client.infer(cloud.points, planner=topdown, ...) ──ZMQ :5556──► GraspGenX
   │                                              → (grasps, conf, branch_tags)  [grasp/graspgenx_client.py]
   │
   └── sim_cloud  (grasp/sim_cloud_source.py: grasps)         ── SIM de-risk (no camera/SAM3)
         pose_source.block_pose() ◄── rt/sim_state            [perception/ground_truth.py: SimStateDetector]
         sample_cube(edge, n) → GT cube cloud (pelvis)
         client.infer(...) ──ZMQ :5556──► GraspGenX → (grasps, conf, tags)
         │
         ▼ (both converge)
   candidates_from_grasps(grasps, conf, side, palm_offset_xyz, R_wristyaw_grasp, branch_tags)
         │                                                    [grasp/tool_transform.py]
         │  applies the grasp→wrist-yaw transform (build_T_wristyaw_grasp): each GraspGenX
         │  6-DoF grasp → a wrist_yaw goal Pose; sorts by confidence; stashes branch_tag
         ▼
   List[GraspCandidate]  (.wrist_goal, .grasp_pose, .confidence, .extra["branch_tag"])
         │  (if viz: viz.show_candidates → viser :8080)       [viz/grasp_viz.py]
         ▼
   09 prints obb/diff split + top-down(approach≈−Z) count
```

## Subsystem B — `grasp_motion()` → plan → execute

```
P.grasp_motion(robot, side, candidates, close_cb, confirm_cb, on_selected)   [primitives.py:grasp_motion]
   │  q0 = robot.arm.get_current_dual_arm_q()
   │
   ├─① _update_collision_world(robot, side, q0)   (gated: --collision-world)  [primitives.py:70]
   │      depth = cam.get_depth_frame();  hand_q = {L,R: hand.get_q(side)}
   │      planner.update_grasp_world(side, depth, K, T_pc, q0, hand_q)         [curobo_planner.py:408]
   │         ├─ EsdfMapper.esdf_from_depth(depth, K, T_pc, robot_filter)       [collision_world.py:99]
   │         │     cuRobo Mapper.integrate(CameraObservation) → compute_esdf() → VoxelGrid
   │         │     robot_filter = robot_depth_filter(q0, hand_q)               [curobo_planner.py:388]
   │         │        cuRobo RobotSegmenter (hand-active, ops_dtype=float32)
   │         │        zeros the robot's own pixels at the LIVE arm+finger pose
   │         └─ _grasp_planner(side).update_world(SceneCfg(voxel=[grid]))
   │
   ├─② planner.plan_grasp_set_sweep(q0, side, [c.wrist_goal], strategies, ...)  [curobo_planner.py:536]
   │      for strategy in planner.yaml grasp.strategies:   (approach/lift offset sweep)
   │        plan_grasp_set(...)                                                 [curobo_planner.py:456]
   │          GoalToolPose.from_poses(K wrist goals, num_goalset=K)   (single tool frame)
   │          disable_collision_links = [wrist] (+ HAND_LINKS if cw on)
   │          _grasp_planner(side).plan_grasp(goal, start, approach_axis=y, lift_axis=z, ...)
   │             └─ cuRobo: Stage1 goalset IK (hand off) → Stage2 free APPROACH (hand ON,
   │                world-collision-checked) → Stage3 linear GRASP descent (hand off) → Stage4 LIFT
   │          _hold_idle(traj, side, q0)   (pin the idle arm to home in every segment)
   │      → GraspPlanOutcome(chosen_index, approach, grasp, lift)
   │      on_selected(chosen, out) → _report_choice (viser mark + sim FK contact check)
   │
   └─③ EXECUTE each segment:  executor.run(traj)                                [executor.py:112]
          approach → grasp → [close_cb: settle + P.close_hand] → lift
          run(): prime(traj.q[0]) → stream q at control_hz, clock × time_dilation,
                 tauff = gravity_comp G(q);  abort-to-hold if |q_des−q_meas| > abort_thresh
                 (◄─ the tracking-error abort: full-speed approach can trip 0.40; see
                  docs/trajectory_speed_tracking.md)
   │
   ▼
GraspResult(ok, info, chosen_index)   → on failure: 09 recovers (open_hand + home)
```

## Key files & functions

- Entry: `scripts/09_graspgen.py:98` `main()` · wiring `scripts/_rig.py:44` `connect()` →
  `g1_classical_manip/factory.py:183` `make_robot()` (`_build_grasp_source` picks the source).
- Primitives (`g1_classical_manip/primitives.py`): `home` :33 · `open_hand` :183 · `grasp_motion`
  :101 · `_update_collision_world` :70.
- Grasp sources (`g1_classical_manip/grasp/`): `graspgenx_source.py:35`, `sim_cloud_source.py:73`,
  `apriltag_source.py`; transform `tool_transform.py: candidates_from_grasps` / `build_T_wristyaw_grasp`.
  Wire clients: `graspgenx_client.py` (`infer`, protocol v2 → `(grasps, conf, branch_tags)`),
  `perception/sam3_client.py`.
- Planner (`g1_classical_manip/motion/curobo_planner.py`): `plan_grasp_set_sweep` :536 ·
  `plan_grasp_set` :456 · `update_grasp_world` :408 · `robot_depth_filter` :388 · `_hold_idle` :435 ·
  `_grasp_planner` (single-tool-frame, voxel-capable when the world is on).
- Collision world (`g1_classical_manip/motion/collision_world.py`): `EsdfMapper.esdf_from_depth` :99.
- Executor (`g1_classical_manip/motion/executor.py`): `run` :112 (`prime` → stream ×`time_dilation`
  → abort-to-hold), `_tauff` = gravity-comp `G(q)`.

## Latency profiling (`--latency`)
`09_graspgen --latency` records a timing span at each stage of the call map above and, at the end,
prints a table + saves a **timeline (Gantt) + per-component bar chart** (`--latency-out`, default
`latency.png`; a sibling `.json` of raw spans too). Off by default (zero overhead). Spans, by
category:
- **compute** (the inference we care about): `depth_grab`, `sam3`, `deproject`, `graspgenx`,
  `tool_transform` (subsystem A); `collision_world`, `plan_grasp` (subsystem B). `sam3`/`graspgenx`
  are timed at their **ZMQ round-trips** (`perception/sam3_client.py:segment`,
  `grasp/graspgenx_client.py:infer`) — so SAM3's cv2-GUI refinement is NOT counted (it falls into the
  untimed gap). `sim_cloud` swaps `depth_grab`+`sam3`+`deproject` for one `cloud_build` span.
- **exec** (motion): `exec:approach|grasp|lift` (executor.run in `grasp_motion`), `home:*`, `hand:*`.
- **wait** (human): `wait: <step>` from each `_rig.confirm` keyboard gate.
The summary footer splits total **compute** vs **exec** vs **wait** vs **untimed gap (GUI/idle)** vs
**wall-clock** — i.e. true pipeline speed separated from the human-in-the-loop overhead. The recorder
is `g1_classical_manip/latency.py` (`LOG`, a no-op `LOG.span(...)` unless enabled).

## External services / buses
- ZMQ services (separate repos/envs): **GraspGenX** `:5556`, **SAM3** `:5557`, **viser** `:8080`.
- DDS topics: arm `rt/lowcmd` (cmd) + `rt/lowstate` (state); hands `rt/dex3/{side}/cmd` + `state`;
  sim GT `rt/sim_state`. Domain/iface in `configs/robot.yaml` (real 0/NIC; sim 1/`lo` via
  `cyclonedds_loopback.xml`).
- Head camera (ZMQ): sim color `:55555` / depth `:55556` (`camera_sim.yaml`); real ZED color +
  depth `:56555` (`camera_real.yaml`).

## Gotchas the diagram makes concrete
- **B① collision world + the self-filter are skipped entirely** unless `--collision-world` (or
  `planner.yaml: grasp.collision_world.enabled`). Default OFF → the approach is self-collision-only.
- **B③ tracking-error abort** is the exact spot the full-speed collision-world approach trips
  (`0.401 > 0.400` in sim); the deferred root fix is `acceleration_scale` — see
  `docs/trajectory_speed_tracking.md`.
- The grasp planner is a **dedicated single-tool-frame** `MotionPlanner` per side (`plan_grasp`
  offsets every goal frame, so the main 3-frame planner can't be used); a lone candidate is
  duplicated into a 2-row goalset (cuRobo `warmup` only primes the goalset path).
- `--source sim_cloud` skips the camera/SAM3 entirely (GT cube cloud from `rt/sim_state`) — the
  in-sim de-risk path; the real grasp run is pending hardware.

## Debugging a rejected plan (`--debug-planner`, `--diagnose`)
`plan_grasp` failures surface as a generic status string — **"Start or End state in collision"**
(the *start*, i.e. current config, OR the *end* pre-grasp/grasp goal violates collision) or
**"No grasp in goal set was reachable"** (no candidate solved IK). Two flags open it up:
- **`--debug-planner`** — turns up cuRobo's *own* logger (`set_curobo_log_level("debug")` in
  `motion/curobo_planner.py`) so the graph planner's "Start or End state in collision", the IK
  stage's reachability message, and per-stage trajopt warnings are **printed** instead of swallowed.
  Tells you *which phase* rejected.
- **`--diagnose`** — on any plan/exec abort, runs `planner.diagnose(q_fail, side, world_points=…)`
  for the *per-element* reason, all on the main 14-DoF model:
  - **SELF-COLLISION** — the overlapping link/sphere pairs + penetration mm. Iterates cuRobo's own
    `collision_pairs` (adjacent-link + `self_collision_ignore` entries — incl. `build_g1_dex3.patch()`
    — already removed), so it is **ignore-matrix faithful**. Independent of `--collision-world`.
  - **WORLD/ESDF** — which spheres sit inside the depth voxels + depth mm (sphere-vs-the-same red
    occupied points the viz overlays). Empty when the collision world is off.
  - **JOINT LIMITS** — the side arm's joints closest to a bound (deg); a negative margin = past it.
  - **SINGULARITY** — `sigma_min` / condition number / Yoshikawa manipulability of the side's 6×7
    wrist Jacobian (FD — cuRobo's `tool_jacobians` is a zero placeholder unless built with
    `compute_jacobian=True`). `sigma_min < 0.01` flags near-singular (healthy configs run ~0.02–0.08).
  With `--visualize`, the offending spheres are overlaid in **magenta** on the grasp scene.
  **Because self-collision is world-independent** (`disable_collision_links` only disables links vs
  the WORLD): a config that is self-collision-free with `--collision-world` off stays so with it on —
  so a failure that appears only with the world is a WORLD collision, and a genuine self-collision
  fails in both modes. `12_check_world --diagnose [--side]` runs the same report against a built ESDF
  in isolation. Use this to decide whether a sphere-model rebuild would even help: real hand-sphere
  overlaps → finer spheres help; joint-limit / near-singular / ESDF penetration → they won't.
