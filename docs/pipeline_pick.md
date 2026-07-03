# `examples/02_pick` pipeline — end-to-end call map

**What:** how the main grasp script (`scripts/examples/02_pick.py`) runs the whole pick-and-lift, from
process wiring through grasp generation, native cuRobo `plan_grasp`, and streamed execution. Written for
the pre-restructure layout and refreshed for `g1_primitives`; the flow is unchanged, but line
numbers below are approximate. File:function references are clickable in the repo.

## TL;DR
`main()` wires the robot (planner + arm + hand + executor + camera + grasp source), applies CLI
overrides, then runs an operator-gated sequence: **home → open → `grasp_source.grasps()` →
`grasp_motion()` → home**. `grasps()` turns depth/cloud into ranked wrist-yaw `GraspCandidate`s (via
SAM3 + GraspGenX for the real path); `grasp_motion()` optionally builds a depth-ESDF collision world,
solves a cuRobo `plan_grasp` goalset, and streams the approach/grasp/lift segments through the
executor (which aborts-to-hold past the tracking-error threshold).

## High-level flow

```
                        scripts/examples/02_pick.py : main()
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │ 1. WIRING   g1.connect(target)  → api/robot.Robot      │
        └───────────────────────────┬───────────────────────────┘
                                    │ builds the Robot facade:
   ┌──────────────┬─────────────────┼───────────────┬──────────────┬───────────────┐
   ▼              ▼                 ▼               ▼              ▼               ▼
CuroboArm     G1_29_Arm        Dex3Hand        Executor      HeadCamera      GraspSource
 Planner      Controller      (Dex3Controller)               (camera_client) (api/_builders)
(motion/      (robot_arm.py)  (robot_hand_     (executor.py)  rgb :55555      graspgenx│sim_cloud
 planner.py)   rt/lowcmd       unitree.py)                     depth :55556    (grasp/*_source.py)
  warmup()     rt/lowstate     rt/dex3/*/cmd                   (sim) / ZED
  + grasp MP                   rt/dex3/*/state                 :56555 (real)
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │ 2. CLI OVERRIDES (main)                                │
        │   --source/--segment/--visualize → robot.set_grasp_    │
        │      source / set_segmenter / set_visualize            │
        │   --abort / sim→0.40, --speed → robot.set_executor     │
        │   --grasp-only → GraspOptions(approach/lift=False)     │
        │   --collision-world → robot.set_collision_world(True)  │
        └───────────────────────────┬───────────────────────────┘
                                    │
        ┌───────────────────────────▼───────────────────────────┐
        │ 3. PICK SEQUENCE  (each step operator-gated console.confirm)│
        └───────────────────────────┬───────────────────────────┘
                                    │
   P.home(robot) ──────────► planner.plan_joint(q, home) ─► executor.run(traj)   [api/primitives.py:home]
   P.open_hand(robot,side) ─► hand.open(side) ─► Dex3Controller cmd              [api/primitives.py:open_hand]
   cands = robot.grasp_source.grasps(robot, side, object)   ◄── ❰SUBSYSTEM A❱
   res  = P.grasp_motion(robot, side, cands, GraspOptions)   ◄── ❰SUBSYSTEM B❱
   P.home(robot)
```

## Subsystem A — grasp generation (`grasp_source.grasps()`)

```
robot.grasp_source.grasps(robot, side, "block")              [grasp/base.py: GraspSource ABC]
   │   sources: graspgenx (live depth) | sim_cloud (sim GT) — grasp.yaml `source:` /
   │   --source. Each grasps() call retains a typed SourceSnapshot (mask + cloud).
   │
   ├── graspgenx  (grasp/graspgenx_source.py: grasps)         ── REAL / live depth
   │     cam.get_rgb_frame() + cam.get_depth_frame()          [hardware/camera_client.py]
   │     frames.T_pelvis_camera(q14)                          [perception/frames.py]
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
P.grasp_motion(robot, side, candidates, options: GraspOptions)   [api/primitives.py:grasp_motion]
   │  options: close policy + approach/lift toggles + confirm gate + GraspObserver hooks
   │  q0 = robot.arm.get_current_dual_arm_q()
   │
   ├─① _update_collision_world(robot, side, q0)   (gated: --collision-world)  [api/primitives.py]
   │      depth = cam.get_depth_frame();  hand_q = {L,R: hand.get_q(side)}
   │      planner.update_grasp_world(side, depth, K, T_pc, q0, hand_q)         [planner.py:408]
   │         ├─ EsdfMapper.esdf_from_depth(depth, K, T_pc, robot_filter)       [collision_world.py:99]
   │         │     cuRobo Mapper.integrate(CameraObservation) → compute_esdf() → VoxelGrid
   │         │     robot_filter = robot_depth_filter(q0, hand_q)               [planner.py:388]
   │         │        cuRobo RobotSegmenter (hand-active, ops_dtype=float32)
   │         │        zeros the robot's own pixels at the LIVE arm+finger pose
   │         └─ _grasp_planner(side).update_world(SceneCfg(voxel=[grid]))
   │
   ├─② planner.plan_grasp_set_sweep(q0, side, [c.wrist_goal], strategies, ...)  [planner.py:536]
   │      for strategy in planner.yaml grasp.strategies:   (approach/lift offset sweep)
   │        plan_grasp_set(...)                                                 [planner.py:456]
   │          GoalToolPose.from_poses(K wrist goals, num_goalset=K)   (single tool frame)
   │          disable_collision_links = [wrist] (+ HAND_LINKS if cw on)
   │          _grasp_planner(side).plan_grasp(goal, start, approach_axis=y, lift_axis=z, ...)
   │             └─ cuRobo: Stage1 goalset IK (hand off) → Stage2 free APPROACH (hand ON,
   │                world-collision-checked) → Stage3 linear GRASP descent (hand off) → Stage4 LIFT
   │          _hold_idle(traj, side, q0)   (pin the idle arm to home in every segment)
   │      → GraspPlanOutcome(chosen_index, approach, grasp, lift, strategy)
   │      observer.on_selected(chosen, report) → viser mark + sim FK contact check
   │
   └─③ EXECUTE each segment:  executor.run(traj)                                [executor.py:112]
          approach → grasp → [close: settle + hand.close per options] → lift
          run(): prime(traj.q[0]) → stream q at control_hz, clock × time_dilation,
                 tauff = gravity_comp G(q);  abort-to-hold if |q_des−q_meas| > abort_thresh
                 (◄─ the tracking-error abort: full-speed approach can trip 0.40; see
                  docs/trajectory_speed_tracking.md)
   │
   ▼
GraspResult(ok, info, report)         → on failure: 02_pick recovers (open_hand + home)
```

## Key files & functions

- Entry: `scripts/examples/02_pick.py` `main()` · wiring `g1_primitives/api/robot.py`
  `Robot.connect()` (`api/_builders.build_grasp_source` picks the source).
- Primitives (`g1_primitives/api/primitives.py`): `home` · `open_hand` · `grasp_motion` ·
  `_update_collision_world`.
- Grasp sources (`g1_primitives/grasp/`): `graspgenx_source.py`, `sim_cloud_source.py`;
  transform `tool_transform.py: candidates_from_grasps` / `build_T_wristyaw_grasp`.
  Wire clients: `graspgenx_client.py` (`infer`, protocol v2 → `(grasps, conf, branch_tags)`),
  `perception/sam3_client.py`.
- Planner (`g1_primitives/motion/planner.py`): `plan_grasp_set_sweep` :536 ·
  `plan_grasp_set` :456 · `update_grasp_world` :408 · `robot_depth_filter` :388 · `_hold_idle` :435 ·
  `_grasp_planner` (single-tool-frame, voxel-capable when the world is on).
- Collision world (`g1_primitives/motion/collision_world.py`): `EsdfMapper.esdf_from_depth` :99.
- Executor (`g1_primitives/motion/executor.py`): `run` :112 (`prime` → stream ×`time_dilation`
  → abort-to-hold), `_tauff` = gravity-comp `G(q)`.

## Latency profiling (`--latency`)
`examples/02_pick --latency` records a timing span at each stage of the call map above and, at the end,
prints a table + saves a **timeline (Gantt) + per-component bar chart** (`--latency-out`, default
`latency.png`; a sibling `.json` of raw spans too). Off by default (zero overhead). Spans, by
category:
- **compute** (the inference we care about): `depth_grab`, `sam3`, `deproject`, `graspgenx`,
  `tool_transform` (subsystem A); `collision_world`, `plan_grasp` (subsystem B). `sam3`/`graspgenx`
  are timed at their **ZMQ round-trips** (`perception/sam3_client.py:segment`,
  `grasp/graspgenx_client.py:infer`) — so SAM3's cv2-GUI refinement is NOT counted (it falls into the
  untimed gap). `sim_cloud` swaps `depth_grab`+`sam3`+`deproject` for one `cloud_build` span.
- **exec** (motion): `exec:approach|grasp|lift` (executor.run in `grasp_motion`), `home:*`, `hand:*`.
- **wait** (human): `wait: <step>` from each `console.confirm` keyboard gate.
The summary footer splits total **compute** vs **exec** vs **wait** vs **untimed gap (GUI/idle)** vs
**wall-clock** — i.e. true pipeline speed separated from the human-in-the-loop overhead. The recorder
is `g1_primitives/latency.py` (`LOG`, a no-op `LOG.span(...)` unless enabled).

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
  `motion/planner.py`) so the graph planner's "Start or End state in collision", the IK
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
  It works at TWO scopes: the **START** (current/home) — `diagnose(q_fail, …)`, magenta overlay —
  and the **END** — `diagnose_candidates(side, grasp_goals, pregrasp_goals, …)` which **sweeps the
  top candidates** (`--diagnose-k`, 0 = all; ~0.8s each) because `plan_grasp` picks ONE of the K-goalset
  and the top-confidence grasp being bad doesn't mean all are. Per candidate it tests, on the
  **world-free** main planner (no ESDF), whether the GRASP pose is reachable, whether the PRE-GRASP
  (backed off along the approach) is reachable, and if so whether that config violates the world —
  `plan_grasp` rejects the pre-grasp but never returns its config, so we re-solve one. The
  grasp-vs-pre-grasp comparison removes the trajopt-path confound (a grasp reachable from home but its
  short-back-off pre-grasp not = a genuine reach/geometry boundary).

### `world_check` — the TRUTH predicate (audit 2026-07-02)
  The geometric world test above (sphere vs occupied-voxel centres) is an **approximation** and once
  produced misleading readings. The REAL gate behind "Start or End state in collision" is cuRobo's
  own checker: per-sphere cost `= (radius + eta) − esdf(center)`, **`> 0` = collision, at eta = 0**
  (cuRobo `metrics_base.yml`; `solver.collision_activation_distance` shapes only the *optimizer*
  cost). `planner.world_check(side, q)` queries the grasp planner's OWN loaded ESDF at exactly that
  predicate (+ a "near" tier at the optimizer eta), labels offending spheres by link, and refuses to
  pass a side whose planner still holds the empty build grid. `09 --diagnose --collision-world` runs
  it on the START config; `diagnose_candidates` uses it for the per-candidate world column whenever
  the live ESDF is loaded; `tools/check_world --diagnose [--side] [--exclude-mask m.npy]` runs it
  standalone (live or `--frame` offline), printing the truth table next to the legacy geometric one.

### The corrected plan_grasp mental model (what the audit established)
  - `plan_grasp` commits to **ONE** goalset winner (Step-1 grasp-pose IK, `disable_collision_links`
    spheres zeroed) and plans only *that* grasp's approach; on failure it returns — **no internal
    retry**. The candidate-retry loop in `plan_grasp_set_sweep` (exclude the failed winner, re-call;
    `grasp.max_candidate_retries`, default 12) is the cross-candidate lever — seeds are not.
  - **`disable_collision_links` applies to Steps 1/3 only.** The Step-2 APPROACH plan (home →
    pre-grasp — where the failures live) runs with **ALL links re-enabled, hand included**. (An
    earlier claim that the hand is disabled vs the world "during the grasp" was wrong for the
    approach segment.) Consequence: a target object fused into the ESDF blocks its own pre-grasp
    region — which is why **`collision_world.exclude_object`** (default ON) cuts the target's SAM3
    mask out of the depth pre-fusion, matching the GraspGenX end2end reference ("the object is
    intentionally NOT added — the gripper has to reach it"). Table + other props remain obstacles.
  - **Seeds**: cuRobo defaults (32/4) are back — 128/16 failed identically (the winner's lone
    approach still failed), and the end2end reference warns non-default seeds/tolerances silently
    broke `plan_grasp` on public cuRobo. Raise them only with `world_check`/sweep evidence.

  Reading the sweep summary: some PRE-GRASP reachable + **world-free** (truth predicate) → a
  collision-free approach EXISTS → the retry loop should land it; if it still fails, `world_check`
  the START (a start inside the ESDF fails EVERY plan identically) and check `exclude_object`.
  Reachable but ALL **ESDF-blocked** → real clutter (adjust approach offsets, `exclude_object`,
  de-clutter). Grasp reachable but NO pre-grasp → the back-off leaves reach or hits a wrist limit
  (shrink `approach_dist` / change axis). No grasp reachable → candidates out of reach (bad grasps /
  wrong side).
  **Because self-collision is world-independent**: a config that is self-collision-free with
  `--collision-world` off stays so with it on — a failure that appears only with the world is a
  WORLD collision, and a genuine self-collision fails in both modes. Use this to decide whether a
  sphere-model rebuild would even help: real hand-sphere overlaps → finer spheres help; joint-limit /
  near-singular / ESDF penetration → they won't.

### Offline truth-test chain (no sim/robot needed after one capture)
  `tools/capture_frame --target sim --out captures/X.npz` (records depth + intrinsics + T_pelvis_camera
  + arm/hand q) → `tools/segment --frame captures/X.npz --save` (writes `captures/X_mask.npy`) →
  `tools/check_world --frame captures/X.npz --exclude-mask captures/X_mask.npy --diagnose --side left
  --visualize` → reads out: is the START in collision per the real gate? is the object cut cleanly
  (`--probe <xyz>` flips IN THE WORLD → ERASED)? what remains near the pre-grasp region?

### Live verification matrix (post-audit; operator-run, multi-prop sim unless noted)
  1. `09 … --side left --collision-world --visualize --diagnose` — the failing case, now with
     `exclude_object` + candidate retry + truth diagnostics.
  2. Same with `--side right` — right+CW was never tested in the multi-prop scene (isolates
     left-specific vs world-generic).
  3. `09 … --side left` (no `--collision-world`) — regression: the world-off path must stay green.
  4. RIGHT + `--collision-world` on the single-block redblock scene — the June-30 golden regression.
