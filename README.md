# g1_primitives

A **cuRobo-native motion + perception primitive library** for the Unitree G1 (29-DoF,
Dex3-1 hands): a small, typed surface meant to be **composed by an LLM agent** as a
baseline for general-purpose pick-and-place. It runs **off-board** on an RTX 5090
workstation that talks to the robot's PC2 (or the Isaac sim) over CycloneDDS. The robot
is **suspended on a back-plate mount**: only the 14 arm joints and 2×7 hand joints are
ever commanded.

Hardware-validated: the MVP motion loop ran on the physical G1 on 2026-06-18, and the
full learned-grasp pick-and-lift (ZED depth → SAM3 → GraspGenX → cuRobo `plan_grasp`)
on 2026-06-30. `HARDWARE_TODO.md` tracks what's still gated on the robot;
`SIM_NOTES.md` covers the Isaac sim rig. (`G1_CLASSICAL_MANIP_PLAN.md` is the
historical design document from before the API restructure.)

## Quickstart

```python
import g1_primitives as g1

robot = g1.connect("sim")                  # or "real" — DDS, camera, debug mode, launch home
robot.home()                               # planned, collision-aware
result = robot.grasp("right", "block")     # candidates -> goalset plan -> approach/grasp/close/lift
if not result:
    print(result.info)                     # why it failed
    print(result.report.phases)            # {"approach": "ok", "grasp": "failed", ...}
robot.open_hand("right")
```

`connect(target)` does everything the old script boilerplate did: target-keyed DDS
(sim = loopback domain 1/`lo`, real = `configs/robot.yaml`), the matching camera config,
**check-first debug-mode entry** on real (an operator-set debug mode is detected and left
untouched; an active ai/loco mode is released via the SDK), and the direct launch home so
planners start from a known collision-free pose. `Robot.offline()` builds the planner +
perception stack with no DDS at all.

The reference example is [`scripts/examples/01_hello_motion.py`](scripts/examples/01_hello_motion.py);
the full pick is [`scripts/examples/02_pick.py`](scripts/examples/02_pick.py).

## The Robot facade

| | |
|---|---|
| **Lifecycle** | `g1.connect(target, ...)` · `Robot.offline(...)` · `robot.close()` |
| **Action verbs** | `home()` · `move(side, pose)` · `open_hand(side)` · `close_hand(side, fraction=)` · `grasp(side, target, options: GraspOptions)` |
| **Perception** | `wait_for_frames(rgb=, depth=)` · `rgb()` · `depth()` · `detect(target) -> Detection` · `grasp_candidates(side, target)` |
| **Reconfigure** | `set_grasp_source(name, **ov)` · `set_segmenter(mode, **ov)` · `set_visualize(on, **ov)` · `set_executor(speed=, abort_thresh_rad=, gravity_comp=, gravity_scale=)` · `set_collision_world(on, **ov)` · `set_grasp_strategies([...])` |

Reconfiguration goes through `set_*` — never mutate `robot.cfg` and rebuild components by
hand; the methods keep the config dict and the rebuilt component in lockstep.

**Results.** Action verbs return `Result(ok, info)` (truthy on success). `grasp` returns a
`GraspResult` whose serializable `report` carries `chosen_index`, per-phase states
(`ok | failed | skipped | pending`), `failed_phase`, the raw cuRobo `planner_status`, and
the winning sweep `strategy`. **Commanded vs verified:** hand verbs default to
`verify=False` and say so — `"close commanded (unverified)"` means the command was sent
and the motion dwell completed, not that an object is held; pass `verify=True` (or
`GraspOptions(verify_close=True)`) to gate on the grasp-signal check.

**Options.** `GraspOptions` is the grasp verb's policy: `close_hand/close_fraction/
verify_close`, `approach/lift` toggles, `grasp_z_offset`, `max_candidates`,
`confirm=` (the operator gate) and `observer=` (a `GraspObserver` for viz/diagnostics
hooks: `on_world_built / on_selected / on_phase`). The no-options default is the full
autonomous pick.

## Layout

```
g1_primitives/
  __init__.py      curated exports: connect, Robot, GraspOptions, Result(s), Pose, ...
  config.py        load_configs — the only yaml reader (+ key-migration errors)
  api/             robot.py (the facade) · primitives.py (verb implementations) ·
                   results.py · options.py · console.py (script helpers) · _builders.py
  spatial/         pose.py — numpy SE(3) Pose · pointcloud.py — PointCloud
  motion/          planner.py — CuroboArmPlanner (the ONLY planner) · executor.py ·
                   trajectory.py · diagnostics.py (explain_failure/world_check/...) ·
                   collision_world.py — EsdfMapper (depth -> cuRobo ESDF)
  perception/      frames.py (frame math, fk injected) · base.py (Detection/Detector) ·
                   sim_state.py · depth.py · segment.py / segment_gui.py · sam3_client.py ·
                   validation.py (GT comparison metrics)
  grasp/           base.py (GraspSource/GraspCandidate/SourceSnapshot) ·
                   graspgenx_source.py · sim_cloud_source.py · graspgenx_client.py (ZMQ) ·
                   tool_transform.py (grasp -> wrist_yaw)
  ee/              hand_base · dex3 · hand_kinematics · dex1 (FUTURE WORK seam, unwired)
  hardware/        vendored endpoints: robot_arm · robot_hand_unitree · motion_switcher
                   (+ ensure_debug_mode) · camera_client · zmq_image_client
  viz/             GraspViz (viser) — cloud + ranked grasps + gripper mesh
configs/           robot, planner, hands, camera_{sim,real}, perception, grasp
                   (+ curobo/, cyclonedds_loopback.xml)
scripts/           checks/ (bring-up ladder) · examples/ (facade demos) · tools/ (dev
                   utilities) — see scripts/README.md for the table + old->new map
tests/             GPU-free suite (fakes for planner/DDS; import-hygiene guard)
```

Layering: `spatial` ← `perception`/`grasp`/`motion`/`ee`/`hardware` ← `api`. Perception
never imports motion (the FK callable is injected into `Frames`); motion never imports
grasp; `api` is the only cross-layer assembler. Importing `g1_primitives` pulls **no
torch/CUDA/DDS** — heavy stacks load on `connect()` (enforced by `tests/test_import_hygiene.py`).

## Environment

The **`g1_curobo`** conda env (Python 3.11, numpy 2, torch 2.9.1+cu128, cuRobo V2 from
source, cyclonedds, unitree_sdk2py). **No pinocchio.** Always use the lane wrapper:

```bash
bash -ic 'use_conda g1_curobo && python -m pytest tests/ -q'      # GPU-free suite
bash -ic 'use_conda g1_curobo && python -c "import g1_primitives"'
```

### Replicating the env

Build/install is **not** plain `pip install` — three deps compile against system
libraries, in this order, *before* the requirements file (which pins everything else,
including the msgpack ZMQ wire deps). Assumes system CUDA 12.8 at `/usr/local/cuda-12.8`
(the cuRobo arch flag below targets the RTX 5090, sm_120 — adjust `TORCH_CUDA_ARCH_LIST`
for another GPU) and the CycloneDDS C lib at `/opt/cyclonedds`.

```bash
conda create -n g1_curobo python=3.11 -y
bash -ic 'use_conda g1_curobo && \
  # 1. CUDA torch (cu128 wheel; sm_120-capable)
  pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 && \
  # 2. cuRobo V2 from source (commit pin in requirements-curobo.txt)
  git clone https://github.com/NVlabs/curobo ~/repos/curobo ; \
  CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="12.0" pip install -e "~/repos/curobo[cu12]" --no-build-isolation && \
  # 3. CycloneDDS bindings against the system C lib
  CYCLONEDDS_HOME=/opt/cyclonedds pip install cyclonedds==0.10.2 && \
  # 4. Unitree SDK (editable, --no-deps so it cannot downgrade numpy)
  git clone https://github.com/unitreerobotics/unitree_sdk2_python ~/repos/unitree_sdk2_python ; \
  pip install -e ~/repos/unitree_sdk2_python --no-deps && \
  # 5+6. everything else, then this package
  pip install -r requirements-curobo.txt && pip install -e .'
```

Verify: the two commands at the top of this section (import check + suite), then
`scripts/checks/01_dds.py` against the sim. The same recipe lives in the header of
`requirements-curobo.txt`. The GraspGenX / SAM3 **servers** are separate repos with their
own envs (see "Grasp pipeline — servers and data flow" below) — this env only ships
their thin ZMQ clients.

## Run against the Isaac sim

Two terminals share one DDS bus on loopback (full notes + gotchas in `SIM_NOTES.md`).

**Terminal 1 — Isaac sim** (`launch_sim.sh` activates the `unitree` env itself):
```bash
cd ~/repos/unitree_sim_isaaclab
UNITREE_DDS_IFACE=lo \
CYCLONEDDS_URI=file://$HOME/repos/G1_classical_manip/configs/cyclonedds_loopback.xml \
TASK=Isaac-PickPlace-RedBlock-G129-Dex3-Joint \
./launch_sim.sh                 # opens the Isaac viewer; add --headless for none
```
Wait until it prints `[DDSManager] DDS system initialized` and is stepping.

**Terminal 2 — control stack** (the `g1_curobo` env):
```bash
cd ~/repos/G1_classical_manip
export CYCLONEDDS_HOME=/opt/cyclonedds
export CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml   # MUST match the sim
use_conda g1_curobo

# bring-up ladder (each --target sim|real), run in order:
python scripts/checks/01_dds.py    --target sim   # READ-ONLY arm + hand state (no motion)
python scripts/checks/02_camera.py --target sim   # head-cam RGB feed
python scripts/checks/03_depth.py  --target sim   # head-cam DEPTH feed (sim ZMQ :55556 / real ZED)
python scripts/checks/04_hands.py  --target sim --side right   # close/open primitives
python scripts/checks/05_move.py   --target sim   # planned home (add --dz 0.1 for a lift)

# then the examples:
python scripts/examples/01_hello_motion.py --target sim
python scripts/examples/02_pick.py --target sim --source sim_cloud --visualize
```
`checks/01–03` are read-only (safe first contact); `04/05` and the examples command the
arms/hands. `scripts/README.md` has the full table (incl. the `tools/`) and the
old→new script-name map.

**Reset the block (sim only)** — re-place the red block between pick attempts (runs in
the **`unitree`** sim env, same loopback bus):
```bash
UNITREE_DDS_IFACE=lo CYCLONEDDS_URI=file://$HOME/repos/G1_classical_manip/configs/cyclonedds_loopback.xml \
  bash -ic 'use_conda unitree && python reset_pose_test.py'
```

**Multi-prop robustness scene** — `TASK=Isaac-PickPlace-Props-G129-Dex3-Joint` spawns a
6-prop clutter grid (red cube `object` + cylinder, sphere + mug, toy truck + brick);
`PROPS_ROW=front|back|both` toggles rows. Run the pick with `--segment interactive` and
choose a prop in the SAM3 GUI (`sim_cloud` stays cube-only). See SIM_NOTES.md.

## Grasp pipeline — servers and data flow

Grasp poses come from `robot.grasp_source` (grasp.yaml `source:`):
- **`graspgenx`** — the real pipeline: head depth → SAM3 mask (applied pre-deproject) →
  pelvis-frame `PointCloud` → GraspGenX ZMQ → ranked 6-DoF grasps → the kinematics-derived
  grasp→wrist transform (`grasp.yaml: wristyaw_grasp_rpy` + `palm_offset_xyz`,
  hardware-confirmed 2026-06-30; recipe: `scripts/tools/derive_tool_transform.py`).
- **`sim_cloud`** — SIM-only de-risk: an analytic GT cube cloud at the live `rt/sim_state`
  block pose → the same GraspGenX + transform (no camera/depth/SAM3).

`grasp_motion` plans candidates with cuRobo's **native `plan_grasp`**: a K-candidate
goalset (cuRobo picks the feasible grasp) + approach/grasp/lift segments, sweeping the
configured strategies (`planner.yaml: grasp.strategies`).

The **GraspGenX** model and **SAM3** segmenter run as separate GPU services in their own
repos + conda envs; this repo ships only thin ZMQ clients. One-time wire deps per server
env: `bash -ic 'use_conda <env> && python -m pip install msgpack msgpack-numpy'`. Launch
(each in its own terminal):

```bash
# SAM3 segmentation server (:5557)
bash -ic 'use_conda sam3 && cd ~/repos/sam3 && python -m sam3.serving --host 0.0.0.0 --port 5557 --device cuda'

# GraspGenX grasp server (:5556)
bash -ic 'use_conda graspgenx && cd ~/repos/GraspGenX && python client-server/graspgenx_server.py --config ext/graspgenx_checkpoints/release --assets_dir assets --default_gripper unitree_g1 --port 5556'
```

**On the robot** — the full pick+lift (operator-gated each step; watch the e-stop; needs
both servers + the ZED; debug mode is checked/entered on connect):
```bash
bash -ic 'use_conda g1_curobo && python scripts/examples/02_pick.py --target real --segment interactive --visualize'
```

**Offline demo (no robot)** — capture once, then iterate segmentation + grasps on the
saved frame:
```bash
python scripts/tools/capture_frame.py --target real --out captures/scene1.npz
python scripts/tools/segment.py --frame captures/scene1.npz --mode interactive --save
python scripts/tools/graspgen_viz.py --pcd captures/scene1_cloud.ply    # viser :8080
```
`tools/grasp_preview.py` is the same loop LIVE against the camera (sim or real, no
motion) — the place to iterate SAM3 prompts and GraspGenX params (`--latency` profiles
each component at its ZMQ round-trip; instrumentation lives in `g1_primitives/latency.py`).

**Perception regression gate (sim)** — score the full stack against `rt/sim_state`
ground truth (cloud centroid/chamfer, SAM3-mask IoU vs the projected GT, grasp-implied
object error, fingertip FK):
```bash
python scripts/tools/validate_perception.py --segment auto --max-centroid-mm 20 --max-chamfer-mm 15
```

## Depth-ESDF collision world (`--collision-world`)

Head depth → cuRobo `Mapper` → an ESDF `VoxelGrid` handed to the grasp planner, so the
`plan_grasp` **approach** routes *around* the table/clutter — **including the target
object itself**: the object stays IN the world, and native `plan_grasp` gives the
per-phase semantics we want (the Step-2 approach collision-checks the arm AND hand
against the target so the motion to the pre-grasp can't sweep through it; the goalset
pick / grasp descent / lift disable the hand links so the intended contact is allowed).
A cuRobo `RobotSegmenter` **self-filter** removes the robot's own arm/fingers from the
depth at the live measured config. `collision_world.exclude_object` (cut the target via
its SAM3 mask) is an opt-in clutter escape hatch, default OFF. The whole world is OFF by
default; opt in with `examples/02_pick.py --collision-world` or
`robot.set_collision_world(True)` (tuning in `planner.yaml: grasp.collision_world`).

Inspect the world in isolation with `tools/check_world.py` (`--visualize`, `--probe X Y Z`,
`--frame captures/*.npz` for fully-offline replay, `--diagnose` for the planner's own
collision gate). When a plan is rejected, `examples/02_pick.py --diagnose` /
`--debug-planner` explain WHY (self-collision pairs, ESDF-penetrating spheres, joint
limits, singularity) — see `docs/pipeline_pick.md`.

> **Known upstream bug (worked around):** cuRobo's voxel collision kernel truncates
> float32 grid dims (phantom hits + invisible obstacles). Every grid we author goes
> through `collision_world.kernel_safe_dims` — do NOT remove it until upstream fixes
> land. Full report: `docs/curobo_voxel_dims_bug.md`.

## Status

- **2026-06-18 (real):** MVP `home → move → close → open → home` end-to-end on the
  physical G1. Real needs: gravity comp ON, `time_dilation 0.5`, `arm_velocity_limit ≥ 12`
  (the controller's velocity clip throttles PD torque; see
  `docs/trajectory_speed_tracking.md`, `docs/gravity_comp.md`).
- **2026-06-30 (real):** full GraspGenX pick-and-lift — ZED depth → SAM3 → GraspGenX
  (`planner: topdown`) → derived transform → native `plan_grasp` — picked and lifted the
  object, with `--collision-world` + the live self-filter, tracking clean at 0.5.
- Open on hardware: `detect` validation + ZED intrinsics/mount, `pinch`/`verify` hand
  preset tuning, the `acceleration_scale` root fix (deferred), collision-world
  obstacle-avoidance re-validation (the 06-30 run predated the voxel-bug fix), and the
  motion-switcher A/B (operator-set debug must log "already in debug (untouched)").
- **Dex1 (gripper) support** is a kept seam (`ee/dex1.py`, `hand: dex1`), planned for a
  future iteration — no `g1_dex1_curobo.yml` exists yet, so it is NOT runnable today.
