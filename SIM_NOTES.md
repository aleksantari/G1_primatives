# Sim notes — unitree_sim_isaaclab

The motion stack drives the Isaac sim over the **same DDS interface** the real robot uses
(`rt/lowcmd` ← `rt/lowstate`, plus `rt/dex3/*`), so plan → retime → execute runs end-to-end
with **no code changes**.

**Result:** arm path planning runs cleanly. `01_sim_arm_smoke.py --isaac` homes and executes
the home → hover → descend → lift → return pick cycle with **max tracking error ~0.13 rad and
zero executor aborts**. The G1+Dex3 task fixes the base and uses Isaac's PD actuators, and the
Dex3 hands connect over DDS.

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

**Terminal 2 — our control stack** (the `g1_classical_manip` env):
```bash
use_conda g1_classical_manip
cd ~/repos/G1_classical_manip
export CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml   # MUST match the sim's
python scripts/00_dds_echo.py --domain 1 --interface lo   # sanity: live arm q from the sim (Ctrl-C)
python scripts/01_sim_arm_smoke.py --isaac                # home + pick cycle; prints tracking error
```
Expected: `home: ok`, then `ExecutionResult(success=True, aborted=False, max_tracking_error≈0.13)`.

> If the smoke hangs on `Waiting to subscribe dds...`, the sim's DDS bridge has gone stale
> (happens after the sim has been up a long time / many reconnects) — restart Terminal 1.

## Gotchas (encoded so they aren't rediscovered)
- **MUST launch via the `use_conda unitree` lane wrapper** (e.g. `launch_sim.sh`), NOT the
  env's python binary directly. The conda **activation hooks** set `LD_LIBRARY_PATH` / CUDA
  paths that Warp needs inside Omniverse Kit. Bypassing them makes Warp's `cuda_init` fail
  (`CUDA error 36: cuDeviceGetUuid not supported`) and the kit process aborts ~6 s into boot
  with no Python traceback.
- **Loopback DDS**: `lo` has no MULTICAST flag, so CycloneDDS default discovery fails. Use
  `configs/cyclonedds_loopback.xml` (unicast localhost peers) on **both** processes via
  `CYCLONEDDS_URI`, plus `UNITREE_DDS_IFACE=lo` for the sim.
- **DDS interface override**: `unitree_sim_isaaclab/dds/dds_master.py` hardcoded
  `ChannelFactoryInitialize(1, "wlp13s0")`; made it `os.environ.get("UNITREE_DDS_IFACE",
  "wlp13s0")` so loopback (`lo`) is selectable.
- **Dex3 `press_sensor_state[].pressure` is a per-finger ARRAY, not a scalar** — our
  `robot_hand_unitree.py` reduces it to a scalar (max). This bug (found in Isaac) would have
  hit the real Dex3 too.
- Only the 14 arm joints (`rt/lowcmd[15:29]`) drive the articulation; leg/waist commands are
  ignored by the task — our debug-mode leg-lock is harmless.
- The `--isaac` smoke path sets `connect_hand=True` and a 0.40 rad abort threshold (Isaac PD
  lags transiently on the fast initial homing move; the pick cycle itself tracks to ~0.13).

## Tasks available (G1-29dof + Dex3)
`Isaac-PickPlace-RedBlock-G129-Dex3-Joint`, `Isaac-AprilTag-Calibration-G129-Dex3-Joint`,
`Isaac-PickPlace-Cylinder-G129-Dex3-Joint`, `Isaac-Stack-RgyBlock-G129-Dex3-Joint`,
`Isaac-Pick-Redblock-Into-Drawer-G129-Dex3-Joint`. Set via `TASK=...` for `launch_sim.sh`.

## Not yet wired (future, see HARDWARE_TODO/plan)
Perception in sim uses the `videohub` RPC (`VideoClient.GetImageSample`), not teleimager/ZMQ —
needs a small image backend. Episode reset is `rt/reset_pose/cmd` (String_, category int).
Block ground-truth pose is on `rt/sim_state` (JSON). These enable the full pick-place/handover
FSM in sim; the arm-path-planning path above needs none of them.
