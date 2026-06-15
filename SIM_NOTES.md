# Sim notes — running the stack in simulation

The motion stack drives a sim over the **same DDS interface** the real robot uses
(`rt/lowcmd` ← `rt/lowstate`, plus `rt/dex3/*`), so plan → retime → execute runs end-to-end
with no code changes. Two targets are documented here: **unitree_sim_isaaclab** (full
fidelity — works, recommended) and **unitree_mujoco** (lightweight — partial).

---

# unitree_sim_isaaclab (Isaac Sim 5.0) — WORKS ✅

**Result:** arm path planning runs cleanly. `01_sim_arm_smoke.py --isaac` homes and executes
the home→hover→descend→lift→home pick cycle with **max tracking error ~0.13 rad and zero
executor aborts**. The G1+Dex3 task fixes the base and uses Isaac's PD actuators, so the
mujoco droop/fall problems do not occur, and the Dex3 hands connect over DDS.

## Run it (two conda envs, one DDS bus on loopback)
```bash
# 1) sim — in the `unitree` env via the lane wrapper (conda activation is REQUIRED, see gotcha)
cd ~/repos/unitree_sim_isaaclab
UNITREE_DDS_IFACE=lo \
CYCLONEDDS_URI=file://$HOME/repos/G1_classical_manip/configs/cyclonedds_loopback.xml \
TASK=Isaac-PickPlace-RedBlock-G129-Dex3-Joint ./launch_sim.sh --headless

# 2) our stack — in the `g1_classical_manip` env, SAME CYCLONEDDS_URI
cd ~/repos/G1_classical_manip
CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
python scripts/00_dds_echo.py --domain 1 --interface lo     # live arm q from Isaac
CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
python scripts/01_sim_arm_smoke.py --isaac                  # home + pick cycle, tracking report
```

## Gotchas (Isaac)
- **MUST launch via the `use_conda unitree` lane wrapper** (e.g. `launch_sim.sh`), NOT the
  env's python binary directly. The conda **activation hooks** set `LD_LIBRARY_PATH` / CUDA
  paths that Warp needs inside Omniverse Kit. Bypassing them makes Warp's `cuda_init` fail
  (`CUDA error 36: cuDeviceGetUuid not supported`) and the kit process aborts ~6 s into boot
  with no Python traceback.
- **DDS interface override**: `unitree_sim_isaaclab/dds/dds_master.py` hardcoded
  `ChannelFactoryInitialize(1, "wlp13s0")`; made it `os.environ.get("UNITREE_DDS_IFACE",
  "wlp13s0")` so loopback (`lo`) is selectable. Loopback still needs
  `configs/cyclonedds_loopback.xml` (no multicast on `lo`) on BOTH processes.
- **Dex3 `press_sensor_state[].pressure` is a per-finger ARRAY, not a scalar** — our
  `robot_hand_unitree.py` reduces it to a scalar (max). This bug (found in Isaac) would have
  hit the real Dex3 too.
- Only the 14 arm joints (`rt/lowcmd[15:29]`) drive the articulation; leg/waist commands are
  ignored by the task — our debug-mode leg-lock is harmless.
- The `--isaac` smoke path sets `connect_hand=True` and a 0.40 rad abort threshold (Isaac PD
  lags transiently on the fast initial homing move; the pick cycle itself tracks to ~0.13).

---

# unitree_mujoco — partial ⚠️

## Status
- ✅ unitree_mujoco G1 sim runs **headless**; our pipeline connects over DDS, streams
  `rt/lowcmd`, and reads `rt/lowstate`. Joint indexing verified against the official
  `g1_joint_index_dds.md` (IDL 15–28 = our `G1_29_JointArmIndex`, exact match).
- ✅ Executor safety abort + IK branch-flip guard both fire correctly in the loop.
- ⚠️ **Clean joint tracking is blocked by the unitree_mujoco g1 model, not our code.**
  With the arm commanded to *any* target (home or zeros), **left/right shoulder-yaw and
  elbow sit at a fixed ~±150°/115°** and do not respond, while the other 12 joints track.
  Root cause: the arm motors are torque-clipped (`ctrlrange="-25 25"`, wrists `-5 5`) and
  those two joints don't reach their setpoint under our torque law — a sim model / sensor /
  self-collision issue on the mujoco side. Raising kp (80→300) and adding RNEA gravity
  feed-forward did not change it (the command saturates the ±25 Nm clip). This is sim
  physics tuning, separable from the planner/IK/retimer (all validated offline).

## Setup performed (in ~/repos/unitree_mujoco, a separate repo)
- `pip install mujoco pygame` into the `g1_classical_manip` env (numpy stayed 1.26.4).
- `simulate_python/config.py`: `ROBOT="g1"`, `USE_JOYSTICK=0`.
- `simulate_python/run_headless_g1.py`: headless runner (no GLFW viewer) honoring a
  `ROBOT_SCENE` env override.
- `unitree_robots/g1/scene_fixed.xml`: welds the pelvis (suspended-robot equivalent), so the
  free base doesn't fall under gravity.

## Run it
```bash
# 1) sim (headless, fixed base), domain 1 / lo
cd ~/repos/unitree_mujoco/simulate_python
ROBOT_SCENE=../unitree_robots/g1/scene_fixed.xml \
CYCLONEDDS_URI=file://$HOME/repos/G1_classical_manip/configs/cyclonedds_loopback.xml \
python run_headless_g1.py

# 2) our stack against it (separate shell, SAME CYCLONEDDS_URI)
cd ~/repos/G1_classical_manip
CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
python scripts/00_dds_echo.py --domain 1 --interface lo        # connectivity
CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \
python scripts/01_sim_arm_smoke.py --sim                       # plan->retime->execute
```

## Gotchas (encoded so they aren't rediscovered)
- **Loopback DDS**: `lo` has no MULTICAST flag, so CycloneDDS default discovery fails.
  Use `configs/cyclonedds_loopback.xml` (unicast localhost peers) on **both** the sim and
  the client, via `CYCLONEDDS_URI`.
- **No hands in the g1 scene**: `make_robot(connect_hand=False)` (the `--sim` path) skips the
  Dex3 controller, which would otherwise time out waiting for `rt/dex3/*/state`.
- **Free base falls**: use `scene_fixed.xml` (welded pelvis) to emulate the suspended mount.
- **Sim gains**: the mujoco arm is torque-controlled with our kp/kd and clipped at ±25 Nm,
  so it tracks with far more lag than the real robot's high-bandwidth position loop. The
  `--sim` path slows the trajectory and loosens the executor abort threshold accordingly.
- The viewer (`unitree_mujoco.py`) needs a GLFW window; prefer the headless runner for
  automated/SSH runs.
