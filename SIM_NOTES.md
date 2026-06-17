# Sim notes — unitree_sim_isaaclab

The motion stack drives the Isaac sim over the **same DDS interface** the real robot uses
(`rt/lowcmd` ← `rt/lowstate`, plus `rt/dex3/*`), so plan → execute runs end-to-end with
**no code changes**. The sim's dex3 robot is the **calibrated mode_16 dex3 USD** — the same
kinematics cuRobo plans on (sim == cuRobo == real).

**Result:** the cuRobo-native MVP (`scripts/05_mvp_demo.py`) runs `home → move → close_hand →
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
python scripts/01_check_dds.py     # READ-ONLY arm + hand state (no motion)
python scripts/02_check_image.py   # head-cam live feed
python scripts/03_hands.py --side right   # close/open hand primitives
python scripts/04_move.py          # home (add --dz 0.1 for a lift)
python scripts/05_mvp_demo.py      # full MVP cycle
python scripts/06_detect.py        # AprilTag feed + pose vs ground truth
```
Expected (05_mvp_demo): `home: ok`, `move: ok`, `close: grasped`, `open: opened`, `home: ok`, `DONE`.

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
- `05_mvp_demo.py --target sim` sets a 0.40 rad abort threshold (Isaac PD lags transiently on
  fast moves); the cuRobo trajectory itself tracks well within that. (`--target real` keeps the
  config default; override with `--abort`.)

## Tasks available (G1-29dof + Dex3)
`Isaac-PickPlace-RedBlock-G129-Dex3-Joint`, `Isaac-AprilTag-Calibration-G129-Dex3-Joint`,
`Isaac-PickPlace-Cylinder-G129-Dex3-Joint`, `Isaac-Stack-RgyBlock-G129-Dex3-Joint`,
`Isaac-Pick-Redblock-Into-Drawer-G129-Dex3-Joint`. Set via `TASK=...` for `launch_sim.sh`.

## Not yet wired (future)
Perception in sim uses the `videohub` RPC (`VideoClient.GetImageSample`), not teleimager/ZMQ
— needs a small image backend. Episode reset is `rt/reset_pose/cmd` (String_, category int).
Block ground-truth pose is on `rt/sim_state` (JSON). These are needed once perception is
reworked cuRobo-native; the current arm+hand MVP needs none of them.
