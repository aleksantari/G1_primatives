# g1_classical_manip — Implementation Plan for Claude Code

A classical (non-learned) pick → place → arm-to-arm handover pipeline for the Unitree G1
(29-DoF, Dex3-1 hands), running off-board on an RTX 5090 workstation that talks to the
robot's PC2 over CycloneDDS. The robot is **suspended on a back-plate mount**: only the
14 arm joints and 2×7 hand joints are ever commanded. Perception v1 is AprilTag on the
head RealSense; planner v1 is Cartesian waypoints + the stock dual-arm IK; planner v2 is
cuRoboV2 behind the same interface.

This document is the source of truth. Read it fully before writing code.

---

## 0. Verified facts from upstream repos (do not re-derive these)

All claims below were verified by direct inspection of the cloned repos on 2026-06-12.

### 0.1 xr_teleoperate (`unitreerobotics/xr_teleoperate`, Apache-2.0)

| Asset | Path in upstream | What it gives us |
|---|---|---|
| `G1_29_ArmController` | `teleop/robot_control/robot_arm.py` | Complete 250 Hz dual-arm streaming executor. Debug mode publishes `rt/lowcmd` and **locks every non-arm joint at its current q** (exactly right for the suspended robot); motion mode publishes `rt/arm_sdk` and sets the weight via `motor_cmd[kNotUsedJoint0].q`. Velocity-clipped target tracking (`clip_arm_q_target`), `ctrl_dual_arm(q14, tau14)`, `ctrl_dual_arm_go_home()`, `get_current_dual_arm_q/dq()`, `speed_gradual_max()`, `simulation_mode` flag. CRC handled via `unitree_sdk2py.utils.crc.CRC`. |
| `G1_29_ArmIK` | `teleop/robot_control/robot_arm_ik.py` | Dual-arm weighted IK: Pinocchio + CasADi + Ipopt, warm-started, costs = 50·translation + 1·rotation + 0.02·‖q‖² + 0.1·smoothness, `WeightedMovingFilter` output smoothing, gravity-compensation torques via RNEA. API: `solve_ik(left_wrist_4x4, right_wrist_4x4, q14, dq14) → (q14, tau14)`. **Targets are SE(3) in the pelvis/base frame**; EE frames `L_ee`/`R_ee` are +0.05 m along x of the wrist-yaw joints. Builds reduced model by locking legs/waist/hand joints of `g1_body29_hand14.urdf`; caches the model to a pickle. |
| `Dex3_1_Controller` | `teleop/robot_control/robot_hand_unitree.py` | Dex3 over DDS: topics `rt/dex3/{left,right}/cmd` and `/state`, IDL `unitree_hg HandCmd_/HandState_`, RIS mode bitfield helper class, `ctrl_dual_hand(left_q7, right_q7)`. Note: constructor is wired to multiprocessing shared arrays for teleop — **vendor the DDS plumbing and RIS-mode logic, replace the process model with a plain threaded class** mirroring the arm controller's design. `HandState_` carries `motor_state` (q/dq/tau_est) **and `press_sensor_state`** → both grasp-detection signals are available. |
| `Dex1_1_Gripper_Controller` | same file | Dex1 over `rt/dex1/{left,right}/cmd|state` with `unitree_go MotorCmds_/MotorStates_`. Vendor alongside Dex3 to preserve the swap capability. |
| `MotionSwitcher` | `teleop/utils/motion_switcher.py` | `Enter_Debug_Mode()` (release all modes — required before `rt/lowcmd` control of the suspended robot) / `Exit_Debug_Mode()` via `MotionSwitcherClient`. ~50 lines, vendor as-is. |
| G1 model assets | `teleop/assets/g1/` | `g1_body29_hand14.urdf` (29 body + 14 Dex3 joints) + `g1_body29_hand14.xml` (MJCF) + meshes. Vendor the whole folder. |
| Utilities | `teleop/utils/` | `weighted_moving_filter.py` (IK dependency), `rerun_visualizer.py` (vendor for viz), `episode_writer.py` (optional, for scripted-demo recording). |

### 0.2 unitree_lerobot (`unitreerobotics/unitree_lerobot`, Apache-2.0)

| Asset | Path in upstream | What it gives us |
|---|---|---|
| Component factory pattern | `unitree_lerobot/eval_robot/make_robot.py` | Registry dicts `ARM_CONTROLLERS = {"G1_29": {...}}`, `HAND_CONTROLLERS = {"dex3": ..., "dex1": ..., "inspire": ..., "brainco": ...}` composing arm + IK + hand + image client from CLI/config. **This is the swap infrastructure to inherit — mirror the pattern, don't import the file.** |
| `ImageClient` | `unitree_lerobot/eval_robot/image_server/image_client.py` | ZMQ subscriber (`request_bgr=True` → numpy BGR frames), triple ring buffer, FPS monitor. The matching `image_server.py` runs on the G1's PC2 (already deployed in Aleks's stack), enumerates RealSense devices by serial. We consume RGB only — AprilTag needs no depth. |
| Data conversion | `unitree_lerobot/utils/convert_unitree_json_to_lerobot.py` | If we record scripted demos with `EpisodeWriter`, conversion to LeRobot dataset format **already exists**. Zero extra work for the "classical pipeline as demo generator" payoff. |
| Repo format | top level | `pyproject.toml` (setuptools, `requires-python >=3.10,<3.11`), package dir mirrors repo name, `test/`, `docs/`. Mirror this layout. |
| Sim integration pattern | `eval_robot/eval_g1_sim.py`, `utils/sim_state_topic.py` | How to drive `unitree_sim_isaaclab` over the identical DDS interface (`rt/sim_state` topics for scene reset). |

### 0.3 unitree_sdk2_python — used as a **pip dependency**, never vendored

`ChannelFactoryInitialize(domain_id, interface)`, `ChannelPublisher/Subscriber`, IDL types
(`unitree_hg`: `LowCmd_`, `LowState_`, `HandCmd_`, `HandState_`, `PressSensorState_`;
`unitree_go`: `MotorCmds_`, `MotorStates_`), `CRC`, `MotionSwitcherClient`. Reference
examples: `example/g1/high_level/g1_arm7_sdk_dds_example.py`, `example/g1/low_level/`.

Install pinned from git: `unitree_sdk2py @ git+https://github.com/unitreerobotics/unitree_sdk2_python.git`.

### 0.4 unitree_ros2 — **explicitly skipped**

It is message definitions + a CycloneDDS workspace + examples. Everything it provides,
`unitree_sdk2py` provides natively in Python. Do not add ROS2 to this project.

### 0.5 unitree_mujoco

Supports `ROBOT = "g1"` (29-DoF scene) with the same DDS bridge (`config.py`: domain 1,
interface `lo`). **Limitations:** rubber-hand model (no Dex3 joints), no cameras. Use it
as the cheap motion-stack smoke-test target. The full-fidelity option is
`unitree_sim_isaaclab` (G1 + Dex3 + cameras + image server over identical DDS), which is
the better end-to-end sim given Aleks already runs Isaac Lab — but treat it as optional
(Phase 8), not on the critical path.

### 0.6 cuRoboV2 (`NVlabs/curobo`) — **ships official G1 + Dex3 support**

Verified in-repo:

- `curobo/content/configs/robot/unitree_g1.yml` — full-body G1 kinematics config **including all Dex3 hand links** (thumb_0/1/2, index_0/1, middle_0/1 per hand) with pre-fit collision spheres (~676 sphere entries). No hand-fitting work needed.
- `curobo/content/configs/robot/unitree_g1_29dof_retarget.yml` + `content/assets/robot/g1/g1_29dof_rev_1_0.urdf`, `g1_29dof_with_hand_rev_1_0.urdf`.
- `examples/getting_started/motion_planning.py` — v2 planning API, including a **grasp-planning call that chains approach → grasp → lift segments with finger collisions disabled during final approach**. This maps 1:1 onto our pick phase.
- `examples/getting_started/humanoid_retargeting.py` — G1 floating base via `extra_links` (virtual prismatic/revolute chain injected at load time), global-IK → warm-started local-IK/MPC loop.
- `examples/getting_started/reactive_control.py` — MPC. Future reactive upgrade path.
- Packaging: `pyproject.toml` extras `cu12-torch` (torch ≥2.5) / `cu13-torch` (torch ≥2.9); builds CUDA extensions from source. Python 3.10 OK.

Planning cost on a 4090 is ~35–42 ms end-to-end per the cuRoboV2 paper; the 5090 will be
comfortable for per-segment planning and future replanning.

**Open item to resolve during Phase 7 (not before):** the shipped `unitree_g1.yml` is
whole-body. For our fixed-base dual-arm setup we must (a) fix the base (no `extra_links`
floating chain) and (b) lock legs + waist joints at measured values. cuRobo supports
locked joints via the kinematics config (`lock_joints` / cspace locked positions in v1;
verify the v2 field name against `curobo/_src/types` when implementing). Fallback if
locking fights us: derive a reduced 14-DoF URDF from `g1_29dof_with_hand_rev_1_0.urdf`
and subset the sphere config — mechanical work, no research risk.

---

## 1. What we write ourselves (the actual gaps)

1. **Perception**: AprilTag detector → block pose in pelvis frame (~150 LOC, `pupil-apriltags`).
2. **Transforms module**: single owner of pelvis ↔ camera ↔ tag ↔ grasp-pose math.
3. **Planner layer v1**: Cartesian waypoint generator + Ruckig retiming + IK-based path conversion, behind a planner-agnostic interface.
4. **State machine**: pick / place / handover task graph with failure transitions.
5. **EE abstraction + grasp verification**: thin interface over Dex3/Dex1 controllers; success detection from motor stall / tau_est / press sensors.
6. **Handover primitive**: coordinated dual-arm goal (the stock IK already solves both arms simultaneously — exploit that) + close/open handshake with verification.
7. **cuRobo planner adapter** (Phase 7).

Everything else is inherited per §0.

---

## 2. Repository layout (mirrors unitree_lerobot conventions)

```
g1_classical_manip/
├── pyproject.toml                  # setuptools; requires-python >=3.10,<3.11
├── README.md
├── CLAUDE.md                       # conventions digest for future sessions
├── assets/
│   └── g1/                        # vendored from xr_teleoperate/teleop/assets/g1
│       ├── g1_body29_hand14.urdf
│       ├── g1_body29_hand14.xml
│       └── meshes/
├── configs/
│   ├── robot.yaml                  # dds: {domain, interface}, mode: debug|motion|sim, hand: dex3|dex1
│   ├── camera.yaml                 # intrinsics K, dist; T_pelvis_camera (USER-PROVIDED, never computed here)
│   ├── perception.yaml             # tag family, tag size, tag→block-center offset, re-perceive settings
│   ├── task_pick_place.yaml        # home q, table height, hover offset, place pose, speeds
│   ├── task_handover.yaml          # handover pose pair, approach offsets, handshake timings
│   └── planner.yaml                # planner: cartesian | curobo ; per-planner params; vel/acc/jerk limits
├── g1_classical_manip/
│   ├── robot_control/              # vendored + adapted (keep upstream attribution headers)
│   │   ├── robot_arm.py            # G1_29_ArmController (trimmed to G1_29; sim flag kept)
│   │   ├── robot_arm_ik.py         # G1_29_ArmIK (asset paths → assets/g1; cache → ~/.cache)
│   │   ├── robot_hand_unitree.py   # Dex3/Dex1 DDS plumbing, re-wrapped as threaded classes
│   │   └── motion_switcher.py
│   ├── image_server/
│   │   ├── image_client.py         # vendored
│   │   └── image_server.py         # vendored, deployed to PC2 (reference copy)
│   ├── perception/
│   │   ├── apriltag_block.py       # detect → T_cam_tag → T_pelvis_block
│   │   └── transforms.py           # ALL frame math lives here; pin.SE3 everywhere
│   ├── motion/
│   │   ├── planner_base.py         # Planner ABC: plan(start_q14, goal: Goal, world) -> JointPath
│   │   ├── cartesian_planner.py    # v1: waypoints → per-waypoint dual-arm IK → dense JointPath
│   │   ├── curobo_planner.py       # v2 (Phase 7): same interface
│   │   ├── retimer.py              # Ruckig: JointPath -> JointTrajectory (t, q, qd, qdd)
│   │   └── executor.py             # streams trajectory through G1_29_ArmController; tracks error
│   ├── ee/
│   │   ├── hand_base.py            # open(side), close(side), grasped(side) -> bool
│   │   ├── dex3.py                 # grasp q presets; stall/tau/press-sensor verification
│   │   └── dex1.py
│   ├── tasks/
│   │   ├── fsm.py                  # small generic state machine (states, transitions, on-failure)
│   │   ├── primitives.py           # move_to(named pose), pick(block_pose), place(pose), handover(L→R)
│   │   └── pick_place_handover.py  # composes primitives per task config
│   ├── utils/
│   │   ├── rerun_viz.py            # vendored/adapted rerun_visualizer
│   │   └── episode_recorder.py     # vendored EpisodeWriter (optional, Phase 8)
│   └── factory.py                  # make_robot()-style registry: arm, hand, planner from configs
├── scripts/
│   ├── 00_dds_echo.py              # print lowstate joint q at 1 Hz (connectivity check)
│   ├── 01_sim_arm_smoke.py         # unitree_mujoco: go home, sweep a wrist circle
│   ├── 02_view_camera.py           # image_client frames + AprilTag overlay window
│   ├── 03_static_perception_check.py  # tag at known offsets vs reported pelvis-frame pose
│   ├── 04_hand_check.py            # open/close each Dex3, print q/tau/press during close
│   ├── 05_reach_check.py           # move wrist to perceived hover pose, STOP (no grasp)
│   ├── run_pick_place.py
│   └── run_handover.py
└── tests/                          # pure-math unit tests (transforms, retimer, fsm); no robot needed
```

---

## 3. Architecture rules (enforced, not aspirational)

1. **Three-layer motion stack with hard seams.**
   `Planner.plan(...) -> JointPath` (geometry only, no timing) →
   `Retimer.retime(JointPath) -> JointTrajectory` →
   `Executor.run(JointTrajectory)`.
   The Cartesian planner must output the **same `JointPath` dataclass** cuRobo will later
   output. IK never streams directly to the robot. The executor owns the only handle to
   `G1_29_ArmController`.
2. **All poses are `pin.SE3` expressed in the pelvis frame** (the IK's native frame; the
   robot is suspended so pelvis ≡ world up to the mount). `transforms.py` is the only
   file allowed to construct frame conversions. Camera extrinsics `T_pelvis_camera` come
   from `configs/camera.yaml` and are treated as ground truth.
3. **Dual-arm state is one 14-vector everywhere** (left 7 + right 7, upstream joint
   order). Single-arm motions are produced by holding the other arm's current q as its
   IK target — never by slicing the controller.
4. **Config-driven, code-stable**: task waypoints, speeds, grasp presets, planner choice
   all live in YAML. Switching `planner: cartesian` → `planner: curobo` must be a
   one-line config change.
5. **Every state transition in the FSM logs to Rerun** (current q, target pose, perceived
   block pose, state name). Debuggability is a feature of v1, not a later add.
6. **Safety defaults**: conservative `arm_velocity_limit` until Phase 5 sign-off; the
   executor aborts (hold position) if joint tracking error exceeds a config threshold;
   every script starts from and returns to the home pose; workspace box check on every
   Cartesian goal before planning (reject targets outside the table volume).

---

## 4. Phases with acceptance criteria

Work strictly in order. Each phase ends with its acceptance test passing and committed.

### Phase 0 — Scaffold, vendor, environment
Create the layout above. Vendor files per §0 with upstream attribution headers. Env:
Python 3.10 venv; `unitree_sdk2py` (git), `pin` (pinocchio), `casadi`, `meshcat`,
`ruckig`, `pupil-apriltags`, `opencv-python`, `pyzmq`, `rerun-sdk`, `numpy`, `pyyaml`,
`tyro`. Do **not** install curobo yet.
**Accept:** `pytest tests/` green on transforms/fsm stubs; `python -c "from g1_classical_manip.robot_control.robot_arm_ik import G1_29_ArmIK; G1_29_ArmIK()"` builds the model from `assets/g1` and solves a smoke IK to <1 mm translation error.

### Phase 1 — Robot interface bring-up
Adapt vendored controllers (threaded Dex3 class; config-driven mode selection
debug/motion/sim; `ChannelFactoryInitialize` from `configs/robot.yaml`).
**Accept (sim):** `01_sim_arm_smoke.py` against unitree_mujoco (`ROBOT="g1"`, domain 1,
iface `lo`) goes home and traces a 10 cm wrist circle without limit violations.
**Accept (hw):** `00_dds_echo.py` prints live joint q from the suspended G1;
`04_hand_check.py` opens/closes both Dex3 hands and prints q/tau_est/press streams.
Hardware arm motion happens only after the human confirms the e-stop procedure.

### Phase 2 — Motion stack v1
`planner_base` + `cartesian_planner` (linear interp in SE(3) between waypoints, IK per
sample with warm start, continuity check between consecutive q) + `retimer` (Ruckig,
limits from `planner.yaml`) + `executor` (interpolates the timed trajectory at 250 Hz
into `ctrl_dual_arm`, monitors tracking error). Rerun: planned vs executed q overlay.
**Accept:** in mujoco, home → hover → descend → lift → home executes with max tracking
error < 0.05 rad and zero velocity-clip activations in the executor (i.e., the retimer,
not the controller's safety clip, is the speed authority).

### Phase 3 — Perception
`apriltag_block.py`: ImageClient frame → pupil-apriltags detect (tag family + size from
config) → `T_cam_tag` → `T_pelvis_block` via configured extrinsics and tag→block offset.
Median-filter over N frames; staleness + ambiguity rejection.
**Accept:** `03_static_perception_check.py` — tag moved on the table by hand-measured
10.0 cm steps reports pelvis-frame deltas within ±5 mm; reported pose renders correctly
in Rerun against the robot model.

### Phase 4 — EE layer + grasp verification
`dex3.py`: named presets (`open`, `power_close`, `pinch`) as 7-vectors in config; close =
command preset, then verify within timeout: any of (q stalls > ε short of preset target,
|tau_est| > threshold, press sensors > threshold) ⇒ grasped. Release = open preset +
verify q reaches it.
**Accept:** `04_hand_check.py` extended — closing on the block reports `grasped=True`,
closing on air reports `False`, 10/10 each.

### Phase 5 — Pick & place on hardware
FSM: `HOME → PERCEIVE → PREGRASP(hover +z offset) → REPERCEIVE(correct) → DESCEND →
GRASP(verify) → LIFT → TRANSPORT → RELEASE → RETREAT → HOME`, failure edges back to
PERCEIVE (or ABORT-to-home after k retries). Left arm only; right arm holds home.
**Accept:** ≥ 8/10 pick-place cycles on the real robot with the block randomly placed in
a 20×20 cm region; zero executor aborts; failures auto-detected (grasp verify or
perception), never silent.

### Phase 6 — Arm-to-arm handover
`handover()` primitive: plan **both wrists simultaneously** to a configured handover pose
pair in front of the torso (one IK problem — the stock solver already takes both
targets); right hand pre-shaped open; handshake = right close → verify right grasped →
left open → verify left released → arms separate. If right grasp verification fails,
left does not open (no drops by construction).
**Accept:** ≥ 7/10 full pick → handover → place-with-right runs; the failure mode
"right missed, left still holding" recovers to a retry rather than dropping.

### Phase 7 — cuRoboV2 planner behind the same interface
Install curobo from source (`cu12-torch` extra) in the same env (torch ≥2.5). Build a
fixed-base dual-arm planning config from the shipped `unitree_g1.yml`: no floating
`extra_links`, legs+waist locked at measured q (resolve the v2 lock-joints mechanism per
§0.6; fallback = reduced URDF + sphere subset). World model: table as a cuboid primitive
+ block cuboid from perception. Implement `curobo_planner.py` returning the standard
`JointPath`; map the pick phase onto cuRobo's approach/grasp/lift grasp-planning call.
Keep Ruckig retiming optional behind a flag (cuRobo trajectories are already timed;
compare both).
**Accept:** `planner: curobo` passes the Phase 5 acceptance test unchanged; a deliberate
obstacle (box between home and pregrasp, added to the world config) is avoided — which
the Cartesian planner demonstrably cannot do.

### Phase 8 (optional, unblocked any time after 5)
(a) `episode_recorder` wiring → record scripted pick-place demos → convert with
unitree_lerobot's `convert_unitree_json_to_lerobot.py` → scripted-demo dataset for VLA
fine-tuning. (b) `unitree_sim_isaaclab` as a full-fidelity sim target using the
`eval_g1_sim.py` / `sim_state_topic.py` pattern.

---

## 5. Known gotchas (encode these, don't rediscover them)

- **Debug mode first**: on hardware, call `MotionSwitcher.Enter_Debug_Mode()` before any
  `rt/lowcmd` publishing; verify with `CheckMode()` that no mode is active. The vendored
  arm controller locks non-arm joints at current q — confirm the suspended posture is the
  intended locked posture before first motion.
- **Pelvis frame ≠ table frame**: the mount may tilt the pelvis slightly; calibrate the
  table plane height/normal in pelvis frame empirically (script 05) rather than assuming
  z-up equals table-up.
- **IK EE frame offset**: `L_ee`/`R_ee` sit 0.05 m along wrist-yaw x. Grasp poses must be
  defined for this frame, with the Dex3 palm offset layered in `transforms.py` — one
  constant, one place.
- **Ipopt IK is local**: seed each waypoint solve with the previous waypoint's solution
  (the class already warm-starts via `init_data`); reject paths where consecutive
  solutions jump > threshold (wraps/branch flips).
- **Velocity clip vs retimer**: the controller's `clip_arm_q_target` is a safety net. If
  it activates during nominal execution, the retimer limits are wrong — treat as a bug.
- **Dex3 RIS mode**: command messages need the RIS mode byte set per joint (id/status
  bitfield) and kp/kd populated; copy the vendored `_RIS_Mode` logic verbatim.
- **Python pin**: `>=3.10,<3.11` (matches unitree_lerobot; sdk2py and curobo both fine).
- **Licensing**: all vendored Unitree code is Apache-2.0 — keep headers, add NOTICE.
- **Domain/interface**: real robot = `ChannelFactoryInitialize(0, "<NIC to PC2>")`;
  unitree_mujoco = `(1, "lo")`. Both live in `configs/robot.yaml`, nowhere else.

## 6. Out of scope (do not build)

ROS2 / MoveIt integration; locomotion or waist control; depth-based or learned perception
(post-AprilTag upgrades come later); visual servoing beyond the single re-perceive
correction; Inspire/BrainCo hand support (keep the registry seam, skip the classes);
on-robot (PC2) deployment of any planner code.
