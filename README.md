# g1_classical_manip

A **cuRobo-native motion library** for the Unitree G1 (29-DoF, Dex3-1 hands), exposing a
small set of **action + perception primitives** meant to be **composed by an LLM agent** as a
**baseline for pick-and-place** tasks. This is the **beta** primitive surface — to be expanded.
It runs **off-board** on an RTX 5090 workstation that talks to the robot's PC2 (or the Isaac
sim) over CycloneDDS. The robot is **suspended on a back-plate mount**: only the 14 arm
joints and 2×7 hand joints are ever commanded.

The authoritative design is **`G1_CLASSICAL_MANIP_PLAN.md`**; this README is the quickstart.
**`HARDWARE_TODO.md`** lists what's still gated on the physical robot/camera. **`SIM_NOTES.md`**
covers the Isaac sim.

> **History:** v1 was a classical Cartesian-waypoint + pinocchio-IK + FSM pick-place pipeline.
> That was an experiment; the goal is now a clean primitive surface. v1's Cartesian planner,
> dual-arm IK, Ruckig retimer, and `tasks/` FSM have been removed — cuRobo is the single
> source of kinematics and planning. Perception is kept on disk but dormant.

## Motion stack
```
goal Pose ─► CuroboArmPlanner ─► JointTrajectory ─► Executor ─► DDS ─► (sim | robot)
            plan_pose / plan_cspace   (t,q,qd,qdd,    streams @ control_hz,
            (cuRobo native timing)     cuRobo dt)      aborts-to-hold on tracking error
```
cuRobo emits a time-parameterized, dynamically-feasible trajectory directly — no separate
retiming step. The executor holds the only handle to `G1_29_ArmController`.

## Primitives (`g1_classical_manip/primitives.py`)
```python
home(robot)                       # both arms to the launch pose (forearms forward), settle
move(robot, side, goal_pose)      # one wrist to goal_pose (pelvis frame); other arm holds
grasp_motion(robot, side, candidates, ...)   # native cuRobo plan_grasp: goalset pick +
                                  #   approach → grasp → close → lift
open_hand(robot, side)            # blocks until the fingers finish moving
close_hand(robot, side, verify=, fraction=)  # fraction<1 closes partway; verify=True = held?
detect(robot, target="block")     # head-cam AprilTag → object pose (pelvis frame)
```
Action verbs return `Result(ok, info)`; `detect` returns a `Detection` whose `.pose` feeds
straight into `move`. Tasks are composed from these. **Grasp poses** come from a
`robot.grasp_source` (a `GraspSource`: `apriltag` A-B reference · `graspgenx` learned 6-DoF
grasps from a segmented depth cloud · `sim_cloud` sim-only GT cube cloud) returning ranked
wrist-yaw `GraspCandidate`s that `grasp_motion` plans via cuRobo's native `plan_grasp`
(it picks the feasible grasp from the goalset). `move_to_candidates` (sequential, first-reachable)
is the `--legacy` fallback.

## Layout
```
g1_classical_manip/
  spatial/         pose.py — numpy SE(3) Pose (pin.SE3 replacement) · pointcloud.py — PointCloud
  motion/          curobo_planner — the only planner (plan_to_pose / plan_to_pose_set / plan_joint, fk)
                   executor       — streams JointTrajectory @ control_hz, abort-to-hold
                   planner_base   — JointPath / JointTrajectory containers
                   collision_world — EsdfMapper: head depth → cuRobo Mapper → ESDF for the grasp planner
  ee/              hand_base, dex3, dex1 — grasp presets + verification
  robot_control/   robot_arm (G1_29_ArmController), robot_hand_unitree (threaded Dex3/Dex1)
  grasp/           base (GraspSource ABC + GraspCandidate) · apriltag_source (A-B ref) ·
                   graspgenx_source · graspgenx_client (ZMQ :5556) · tool_transform (grasp→wrist)
  primitives.py    home / move / grasp_motion / open_hand / close_hand / detect
  factory.py       make_robot() — cuRobo planner + DDS controllers + executor + perception + grasp
  image_server/    HeadCamera — head-cam color + depth frames (zmq | teleimager | unitree_lerobot)
  perception/      transforms (frame math, cuRobo FK) · base (Detector seam) · apriltag_block ·
                   ground_truth · depth (deproject→PointCloud) · segment (SAM3 mask seam) ·
                   sam3_client (ZMQ :5557) · segment_gui (cv2 mask GUI)
  viz/             GraspViz (viser) — point cloud + ranked grasps + gripper mesh (offline)
configs/           robot, planner, hands, camera, perception, grasp (+ curobo/, cyclonedds_loopback.xml)
assets/grippers/   <gripper>/{config.json, coll_mesh.obj} for the grasp viz (unitree_g1)
scripts/           01_check_dds → 12_check_world ladder + 10_graspgen_viz (offline grasp viz) + hand_diag.py
tests/             test_pose/grasp/detect + pointcloud/depth_deproject/tool_transform/ graspgenx_client/
                   grasp_source/sam3_client/segment/viz   (pure-math, no robot)
```

## Environment
Runs in the **`g1_curobo`** conda env (Python 3.11, numpy 2, torch 2.9.1+cu128, cuRobo V2
from source, cyclonedds, unitree_sdk2py). **No pinocchio.** Build/install is not plain
`pip install` — see the header of `requirements-curobo.txt`. Always use the lane wrapper:

```bash
bash -ic 'use_conda g1_curobo && pytest tests/'                 # 18 pure-math tests
bash -ic 'use_conda g1_curobo && python -c "import g1_classical_manip.factory"'  # planner build
```

## Run against the Isaac sim
Two terminals share one DDS bus on loopback (full notes + gotchas in `SIM_NOTES.md`).

**Terminal 1 — Isaac sim** (`launch_sim.sh` activates the `unitree` env itself; keep it running):
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

# incremental bring-up ladder (each `--target sim|real`; sim shown), run in order:
python scripts/01_check_dds.py  --target sim     # READ-ONLY arm + hand state (no motion)
python scripts/02_check_image.py --target sim    # head-cam feed (sim mono / real ZED eye)
python scripts/03_hands.py       --target sim --side right   # close/open hand primitives
python scripts/04_move.py        --target sim                # home (add --dz 0.1 for a lift)
python scripts/05_mvp_demo.py    --target sim    # home → move → close → open → home
python scripts/06_detect.py      --target sim    # AprilTag feed: 3D pose axes + rpy vs ground truth
python scripts/07_pick_place.py  --target sim    # pick+lift: home→open→detect→grasp→lift→home
python scripts/08_check_depth.py  --target sim    # head-cam DEPTH feed (sim ZMQ :55556 / real ZED): mm stats + colorized view
python scripts/09_graspgen.py    --target real --source graspgenx --segment interactive  # GraspGenX pick+lift
python scripts/10_segment.py     --frame captures/scene1.npz --mode interactive --save   # SAM3 segment + save colored .ply
python scripts/10_graspgen_viz.py --pcd captures/scene1_cloud.ply                         # GraspGenX grasps in viser
python scripts/11_capture_frame.py --target real --out captures/scene1.npz               # save a frame (offline demo)
python scripts/12_check_world.py --target sim --visualize  # inspect the depth-ESDF collision world (no motion): geometry + self-filter
```
Run them in order — `01`/`02` are read-only/no-motion (safe first contact), `03`–`05`/`07`/`09`
command the arms/hands, `06`/`08`/`10`/`11`/`12` are camera/inspection-only (`08` = head **depth** feed;
`10` = SAM3 **segmentation**; `11` = capture a frame for the offline demo; `12` = inspect the
**depth-ESDF collision world** with no planning/motion; `10_graspgen_viz` = no robot at all).
`02`/`06`/`07` work on both targets; `08`/`12` work on both (sim depth over ZMQ `:55556`, real over the
ZED); `09 --source graspgenx`/`11` need the real ZED; `10` segments either the live ZED (`--target
real`), a captured frame (`--frame`), or a static image (`--image`, no robot). The **grasp servers +
the full offline grasp demo** (`09`/`10`/`10_graspgen_viz`/`11`) are in **Grasp pipeline** below.
`--target real` uses the ZED head via `camera_real.yaml`; the camera scripts don't use the
`CYCLONEDDS_*` exports. `scripts/hand_diag.py` remains a low-level hand diagnostic.

**Reset the block (sim only)** — re-place the red block at its spawn pose between pick
attempts. Runs in the **`unitree`** (sim) env, on the same loopback bus:
```bash
UNITREE_DDS_IFACE=lo CYCLONEDDS_URI=file://$HOME/repos/G1_classical_manip/configs/cyclonedds_loopback.xml \
  bash -ic 'use_conda unitree && python reset_pose_test.py'
```

## Grasp pipeline — servers, offline demo, on-robot run
The **GraspGenX** grasp model and the **SAM3** segmenter run as **separate GPU services** in their
own repos + **conda envs** (`graspgenx`, `sam3`); this repo ships only thin ZMQ clients — no torch /
checkpoints here. Each server needs the ZMQ wire deps in ITS env (one-time; use `python -m pip` so
they land in the env that runs the server, not base/user-site):

```bash
bash -ic 'use_conda graspgenx && python -m pip install msgpack msgpack-numpy'
bash -ic 'use_conda sam3       && python -m pip install msgpack msgpack-numpy'   # if not already present
```

Then start whichever a run needs (each in its own terminal; leave running):

```bash
# SAM3 segmentation server (:5557) — `sam3` env. First run downloads the checkpoint.
bash -ic 'use_conda sam3 && cd ~/repos/sam3 && python -m sam3.serving --host 0.0.0.0 --port 5557 --device cuda'

# GraspGenX grasp server (:5556) — `graspgenx` env (paths relative to the repo after cd).
bash -ic 'use_conda graspgenx && cd ~/repos/GraspGenX && python client-server/graspgenx_server.py --config ext/graspgenx_checkpoints/release --assets_dir assets --default_gripper unitree_g1 --port 5556'
```

**Offline demo (no robot)** — capture one frame on the robot, then iterate segmentation + grasps
on the saved frame, visualized in viser. Needs the SAM3 + GraspGenX servers above:
```bash
# 1. capture a frame on the robot (camera-only, no motion) -> captures/scene1.npz (+ preview PNGs)
bash -ic 'use_conda g1_curobo && python scripts/11_capture_frame.py --target real --out captures/scene1.npz'
# 2. segment it + save the single-object COLORED cloud -> captures/scene1_cloud.ply  (SAM3 :5557)
bash -ic 'use_conda g1_curobo && python scripts/10_segment.py --frame captures/scene1.npz --mode interactive --save'
# 3. run GraspGenX on the cloud + visualize ranked grasps    (GraspGenX :5556 -> viser :8080)
bash -ic 'use_conda g1_curobo && python scripts/10_graspgen_viz.py --pcd captures/scene1_cloud.ply'
```
Open `http://localhost:8080` for the viser viewer (the **colored** cloud + ranked grasps + gripper
mesh + a confidence slider). `--save` writes a colored `.ply` (per-point RGB sampled from the
aligned image; XYZ-only is still what GraspGenX receives) plus `_mask.npy` / `_overlay.png`.
`10_segment` also segments a static image (`--image foo.jpg`, SAM3 only) or the live ZED
(`--target real`); `11_capture_frame` saves a reusable offline fixture each run.

**On the robot** — the full pick+lift via the learned grasp (operator-gated each step; watch the
e-stop; needs both servers + the ZED + debug mode set on the remote):
```bash
bash -ic 'use_conda g1_curobo && python scripts/09_graspgen.py --target real --source graspgenx --segment interactive --visualize'
```
A-B against the AprilTag baseline via `07_pick_place --target real` (09 is GraspGenX-only).
`--visualize` opens the same viser view
(the chosen reachable grasp in green). The grasp→wrist transform (`grasp.yaml: wristyaw_grasp_rpy`
+ `palm_offset_xyz`) is **DERIVED from kinematics** and **hardware-confirmed** (real grasp picked
the object 2026-06-30, incl. the closing-roll sign; see
`scripts/derive_graspgenx_tool_transform.py`). The grasp plans on cuRobo's native `plan_grasp` (goalset pick +
approach/grasp/lift); `--legacy` restores the old sequential path. Useful flags: `--select
{reachable,first}` (all candidates vs the single top-confidence one), `--grasp-only` (one move
straight to the grasp, no approach/lift — a frame sanity check), `--collision-world` (route the
approach around the object/table via the depth-ESDF world — see below), `--speed 0.5` (slower
playback if the dynamic approach trips the tracking-error abort).

**In the Isaac sim (de-risk before the robot)** — `--source sim_cloud` builds a ground-truth cube
cloud from the live `rt/sim_state` block pose → GraspGenX → the same tool transform + gated motion,
so you can watch a 6-DoF grasp execute in physics with **no ZED, no SAM3** (only the GraspGenX
server + the sim). This is the path that **sim-validated the transform** (the run prints an FK check
that our Dex3 fingers straddle the GT cube). Needs `perception.yaml: detector: sim_state`:
```bash
CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
  bash -ic 'use_conda g1_curobo && python scripts/09_graspgen.py --target sim --source sim_cloud --visualize'
```

**Depth-ESDF collision world (`--collision-world`)** — head-camera depth → cuRobo `Mapper` →
an ESDF `VoxelGrid` (`motion/collision_world.EsdfMapper`) handed to the grasp planner, so the
native `plan_grasp` **approach** routes *around* the object/table instead of barging through it.
The **same path works in sim and real** and is **source-independent** (it uses head depth, so any
grasp source — apriltag / graspgenx / sim_cloud — benefits). A cuRobo `RobotSegmenter` **self-filters**
the robot out of the depth (zeroing pixels within `robot_mask_margin` of the collision spheres at
the **LIVE** measured config — arm q from DDS *and* finger q from the Dex3, via a hand-active
segmenter kinematics); without it the grasp starts inside a baked-in copy of its own arm
("Goalset planning returned None"). The active hand links are collision-disabled during the grasp
(the open hand may sit in the object ESDF — the grasp is meant to *contact*). **OFF by default**;
opt in with `09_graspgen.py --collision-world` or `planner.yaml: grasp.collision_world.enabled`
(that block tunes `grid_center` / `extent_m` / `esdf_voxel_size` / `robot_mask_margin` / depth
crop). The approach route is dynamic enough that full-speed playback can trip the tracking-error
abort, but the default `time_dilation 0.5` tracks clean — the **real grasp ran WITH
`--collision-world` + the live self-filter, no aborts, at 0.5 (2026-06-30)** (see
`docs/trajectory_speed_tracking.md`).

Inspect the world IN ISOLATION (no planning, no motion) with **`scripts/12_check_world.py`** — it
builds the ESDF from one depth frame and answers: (1) is the geometry placed right (ESDF occupied
voxels vs the raw deproject cloud)? (2) is the robot's own arm removed (self-filter working)? It
connects DDS read-only (`home_on_connect=False` → nothing moves) and self-filters at the live
arm+hand q, matching the real grasp path. Flags: `--visualize` (viser overlay :8080),
`--probe X Y Z` (is the object still in the ESDF or erased by the filter?), `--margin` (sweep the
self-filter margin), `--esdf-voxel` (override resolution), `--no-self-filter`, `--no-dds` (offline,
filter at home), `--frame captures/scene.npz` (**fully offline replay** of an `11_capture_frame`
capture — no robot/camera/DDS; self-filters at the capture's recorded arm+hand q, else home). The
**sim** depth comes from the `unitree_sim_isaaclab` repo's head-depth ZMQ PUB (`:55556`, raw float32
mm, `front_camera` `distance_to_image_plane`); `camera_sim.yaml` subscribes via `stream.depth_port:
55556`. Offline demo chain: `11_capture_frame` (records depth + intrinsics + pose) →
`12_check_world.py --frame …`.

## Status
cuRobo-native MVP is **sim-validated**: `home → move → close_hand → open_hand → home` runs
end-to-end on `unitree_sim_isaaclab` with zero executor aborts; `home`/`move` are
collision-aware via cuRobo. `detect()` is **sim-validated** too — the head-cam AprilTag
(ID 14) block pose lands within ~3.5 cm of ground truth. Hand presets are untuned
placeholders. **On hardware (2026-06-18): the MVP `home → move → close_hand → open_hand → home`
runs end-to-end on the physical G1.** The arms power-on folded, so a direct un-planned launch home
(PD, velocity-capped) brings them to the launch pose, then the collision-aware planned `home` lands
within ~3° and `move` tracks clean. What makes it work on real: gravity comp **on** (hardware-
validated, off in sim), trajectory **`time_dilation 0.5`** (the velocity clip throttles PD torque,
so the full-speed plan can't be tracked — play it back slower), **`arm_velocity_limit ≥ ~12`**, and
debug mode set by the operator via the physical remote (we don't call `MotionSwitcher`). See
`docs/gravity_comp.md`. **On hardware (2026-06-30): the full GraspGenX pick-and-lift ran
end-to-end on the physical G1** — live ZED depth → SAM3 → GraspGenX (`topdown`) → the derived
grasp→wrist transform → cuRobo native `plan_grasp` (approach → grasp → close → lift) → picked +
lifted the object, **with `--collision-world` + the live robot self-filter**, tracking clean at
the default `time_dilation 0.5` (no aborts). This hardware-confirms the derived transform
(`wristyaw_grasp_rpy` + `palm_offset_xyz`, incl. the closing-roll sign). Still to do on the robot:
`detect` validation + ZED intrinsics/mount, `pinch`/`verify` hand-preset tuning, and the
velocity-clip torque root-fix (the `acceleration_scale` fix is deferred + now lower-priority — 0.5
is a real data point that the collision-world approach tracks). Rerun
logging is still open. `scripts/07_pick_place.py` is the first **composite task** — a single-arm
pick+lift (home→open→detect→approach→grasp→close→lift→home) with a URDF-measured palm/grasp
offset, the LLM-composable baseline to expand (place/handover, dual-arm, multi-object) — see
`HARDWARE_TODO.md` and the PLAN's roadmap.

The head-camera **depth** stream is consumed (`HeadCamera.get_depth_frame()` — the real ZED's
dedicated raw-float32 720×1280 mm stream; validated on the G1 at ~89% finite / ~30 fps via
`scripts/08_check_depth.py`), feeding a **grasp pipeline**: `perception/depth.deproject_depth`
turns masked depth into a pelvis-frame `PointCloud`, segmented by **SAM3** over ZMQ
(`perception/{segment,sam3_client,segment_gui}`, `:5557`) — a 2D mask applied before deproject,
with an interactive cv2 GUI (`scripts/10_segment.py`, validated on static images; `--save` writes a
**colored** `.ply` for the offline viz). The single-object cloud feeds the **GraspGenX** service
(`grasp/graspgenx_client`, `:5556`, **protocol v2**: a `planner` mode — `topdown` for grab-from-above,
`graspmoe`, or `diffusion` — and obb/diff `branch_tags`) which returns ranked 6-DoF grasps; a
kinematics-DERIVED grasp→tool transform (`grasp/tool_transform`) maps them to wrist-yaw goals that
the `grasp_motion` primitive plans with cuRobo's **native** `plan_grasp` (a K-candidate goalset →
cuRobo picks the feasible grasp → + approach/grasp/lift segments; `--legacy` = old sequential
`plan_to_pose_set`). `scripts/09_graspgen.py --source graspgenx|sim_cloud` runs the GraspGenX
pick+lift (AprilTag is the `07_pick_place` A-B baseline; `sim_cloud` = a ground-truth cube cloud
from `rt/sim_state` → GraspGenX, **sim-only**, no
ZED/SAM3 — the path that first **sim-validated** the transform + 6-DoF execution). The transform
(`wristyaw_grasp_rpy` + `palm_offset_xyz`, incl. the closing-roll sign) is now
**hardware-confirmed**: the real GraspGenX/SAM3 grasp picked + lifted the object on 2026-06-30
(with `--collision-world` + the live self-filter) — see `HARDWARE_TODO.md`. Depth wire spec:
`docs/depth_integration_handoff.md`.
