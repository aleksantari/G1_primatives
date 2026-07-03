# Sim notes — unitree_sim_isaaclab

The motion stack drives the Isaac sim over the **same DDS interface** the real robot uses
(`rt/lowcmd` ← `rt/lowstate`, plus `rt/dex3/*`), so plan → execute runs end-to-end with
**no code changes**. The sim's dex3 robot is the **calibrated mode_16 dex3 USD** — the same
kinematics cuRobo plans on (sim == cuRobo == real).

**Result:** the cuRobo-native MVP (`scripts/examples/01_hello_motion.py`) runs `home → move → close_hand →
open_hand → home` with **zero executor aborts**; `home`/`move` are collision-aware via cuRobo.
The G1+Dex3 task fixes the base, uses Isaac's PD actuators, and the Dex3 hands connect over DDS.

## Run it (two terminals: sim + control stack, one DDS bus on loopback)

**Terminal 1 — Isaac sim** (keep it running; `launch_sim.sh` activates the `unitree` env itself):
```bash
cd ~/repos/unitree_sim_isaaclab
UNITREE_DDS_IFACE=lo \
CYCLONEDDS_URI=file://$HOME/repos/G1_classical_manip/configs/cyclonedds_loopback.xml \
TASK=Isaac-PickPlace-RedBlock-G129-Dex3-Joint \
./launch_sim.sh                 # opens the Isaac window so you can watch; add --headless for none
```
Wait until it prints `[DDSManager] DDS system initialized` and is stepping (the `[GT] R_w_cam`
lines are harmless RedBlock-task noise).

**Terminal 2 — our control stack** (the **`g1_curobo`** env):
```bash
cd ~/repos/G1_classical_manip
export CYCLONEDDS_HOME=/opt/cyclonedds
export CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml   # MUST match the sim's
use_conda g1_curobo
# bring-up ladder (all default --target sim):
python scripts/checks/01_dds.py     # READ-ONLY arm + hand state (no motion)
python scripts/checks/02_camera.py   # head-cam live feed
python scripts/checks/04_hands.py --side right   # close/open hand primitives
python scripts/checks/05_move.py          # home (add --dz 0.1 for a lift)
python scripts/examples/01_hello_motion.py      # full MVP cycle
python scripts/06_detect.py        # AprilTag feed + pose vs ground truth
```
Expected (examples/01_hello_motion): `home: ok`, `move: ok`, `close: grasped`, `open: opened`, `home: ok`, `DONE`.

> If it hangs on `Waiting to subscribe dds...`, the sim's DDS bridge has gone stale (happens
> after the sim has been up a long time / many reconnects) — restart Terminal 1.

## Gotchas (encoded so they aren't rediscovered)
- **MUST launch the sim via the `use_conda unitree` lane wrapper** (e.g. `launch_sim.sh`), NOT
  the env's python binary directly. The conda **activation hooks** set `LD_LIBRARY_PATH` /
  CUDA paths that Warp needs inside Omniverse Kit. Bypassing them makes Warp's `cuda_init`
  fail (`CUDA error 36`) and the kit process aborts ~6 s into boot with no Python traceback.
- **Loopback DDS**: `lo` has no MULTICAST flag, so CycloneDDS default discovery fails. Use
  `configs/cyclonedds_loopback.xml` (unicast localhost peers) on **both** processes via
  `CYCLONEDDS_URI`, plus `UNITREE_DDS_IFACE=lo` for the sim.
- **DDS interface override**: `unitree_sim_isaaclab/dds/dds_master.py` hardcoded
  `ChannelFactoryInitialize(1, "wlp13s0")`; made it `os.environ.get("UNITREE_DDS_IFACE",
  "wlp13s0")` so loopback (`lo`) is selectable.
- **Dex3 hand-apply has a both-hands gate.** The sim applies hand joints only when *both*
  `rt/dex3/left/cmd` and `rt/dex3/right/cmd` are non-empty (`action_provider_dds.py`); when
  that gate is skipped the hand joints fall through to **zero (open)**. Our `Dex3Controller`
  publishes **both** hands continuously, so the gate is satisfied — but a client that only
  commands one hand would see neither hand move. Hand commands map to the articulation **by
  joint name** (robust to joint-order shifts).
- **Dex3 `press_sensor_state[].pressure` is a per-finger ARRAY, not a scalar** — our
  `robot_hand_unitree.py` reduces it to a scalar (max). This bug (found in Isaac) would have
  hit the real Dex3 too.
- Only the 14 arm joints (`rt/lowcmd[15:29]`) drive the articulation; leg/waist commands are
  ignored by the task — our debug-mode leg-lock is harmless.
- `examples/01_hello_motion.py --target sim` sets a 0.40 rad abort threshold (Isaac PD lags transiently on
  fast moves); the cuRobo trajectory itself tracks well within that. (`--target real` keeps the
  config default; override with `--abort`.)
- **Sim executes at `time_dilation=1.0`** (factory-forced full-speed playback — the sim bypasses
  the arm controller's measured-relative velocity clip, so it tracks the un-dilated cuRobo
  trajectory; real defaults `0.5`). Caveat: a sufficiently dynamic route (e.g. the collision-world
  grasp APPROACH) can still exceed the 0.40 abort budget at full speed — planning succeeds but
  execution aborts (`approach: tracking error 0.401 > 0.400 rad`); `examples/02_pick --speed 0.5`
  fixes it. See `docs/trajectory_speed_tracking.md`.
- **Sim head DEPTH stream.** Beyond the JPEG color PUB (`:55555`), the sim opens a 2nd ZMQ PUB
  carrying the head `front_camera` `distance_to_image_plane` as a raw float32 `(480,640)` map in
  MILLIMETERS (meters→mm, mirroring the real ZED). `image_client.py`'s `zmq` backend subscribes
  on the configured `depth_port` and exposes it via `get_depth_frame()`; `configs/camera_sim.yaml`
  sets `stream.depth_port 55556`, `depth_height 480`, `depth_width 640` (match the sim env's
  `ISAAC_HEAD_DEPTH_PORT`). This feeds the SAME deproject → PointCloud → cuRobo Mapper/ESDF
  collision world as real, and is what lets `scripts/tools/check_world.py` run in sim. **The sim-side
  depth publisher (`camera_state._publish_head_depth`) lives in the SEPARATE `unitree_sim_isaaclab`
  repo (uncommitted, user-managed) — it is NOT in this repo.**

## Tasks available (G1-29dof + Dex3)
`Isaac-PickPlace-RedBlock-G129-Dex3-Joint`, `Isaac-AprilTag-Calibration-G129-Dex3-Joint`,
`Isaac-PickPlace-Cylinder-G129-Dex3-Joint`, `Isaac-Stack-RgyBlock-G129-Dex3-Joint`,
`Isaac-Pick-Redblock-Into-Drawer-G129-Dex3-Joint`,
`Isaac-PickPlace-Props-G129-Dex3-Joint` (multi-prop clutter — below). Set via `TASK=...` for `launch_sim.sh`.

### Multi-prop "clutter" scene (`Isaac-PickPlace-Props-G129-Dex3-Joint`)
A robustness-testing scene: instead of only the single red cube it puts several
geometrically-distinct props on the table (world x/y):

| x \ y | −4.03 (front) | −4.15 (back) |
|---|---|---|
| **−3.90** (robot's left) | sphere | mug (USD, 0.7×) |
| **−4.20** (middle) | cylinder `object` | — |
| **−4.50** (robot's right) | toy_truck (USD, 90°, 1.2×) | brick (90°) |

The middle-front prop keeps the name `object` (so `rt/sim_state`/the redblock termination+reward,
all keyed to "object", keep working) and is **always spawned** — the env's managers reference it by
name, so removing it crashes env build. It's currently a **cylinder** (was the red cube); note
`sim_cloud` is cube-only so it no longer matches `object`'s true shape — use the `graspgenx` path,
which is shape-agnostic.

**Row toggle:** `PROPS_ROW=front|back|both` (default `both`) picks which row spawns —
`PROPS_ROW=front TASK=Isaac-PickPlace-Props-G129-Dex3-Joint ./launch_sim.sh`. `object` stays put in
every mode; the toggle adds/removes the other 5. (front → object+sphere+toy_truck; back →
object+cylinder+mug+brick.) It's an env var read in `base_scene_pickplace_props.py`.

Drives the full `graspgenx` path on non-box geometry + distractors (SAM3 segments whichever prop you
click; the depth-ESDF collision world sees all active props). The redblock task is untouched. Files
in `unitree_sim_isaaclab`: `tasks/common_scene/base_scene_pickplace_props.py` (scene + `PROP_NAMES`,
which follows `PROPS_ROW`), `tasks/g1_tasks/pick_place_props_g1_29dof_dex3/` (task, in
`tasks/g1_tasks/__init__.py`). USD props are in `assets/objects/{mug,toy_truck}/` (IsaacLab props are
Nucleus/S3-only — not local by default). Run it:
```bash
TASK=Isaac-PickPlace-Props-G129-Dex3-Joint ./launch_sim.sh          # add PROPS_ROW=front|back
# then, from this repo:
bash -ic 'use_conda g1_curobo && python scripts/examples/02_pick.py --target sim \
    --source graspgenx --segment --collision-world --visualize --grasp-only'
```
Notes: `--source sim_cloud` still works but only grasps the `object` cube (cube-only; point it at
another prop via `perception.sim_state.object_key`). Prop poses are a first cut — eyeball/tune
`pos`/`z` (and USD `scale`) in the viewport. If PhysX warns about contact-pair buffers, the env cfg
already bumps `gpu_*_aggregate_pairs_capacity`; raise further if needed.

## Perception in sim (now wired)
Head **color** (JPEG, `:55555`) and **depth** (raw float32 mm, `:55556`) both stream over ZMQ
PUB sockets the sim opens (see the head-depth gotcha above); `image_client.py`'s `zmq` backend
SUBs both. Block ground-truth pose is on `rt/sim_state` (JSON), consumed by the `sim_cloud`
grasp source. Episode reset is `rt/reset_pose/cmd` (String_, category int). This is what enables
the in-sim de-risk path for the grasp pipeline + the depth-ESDF collision world
(`scripts/examples/02_pick.py`, `scripts/tools/check_world.py`).
