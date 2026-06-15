# Sim notes — unitree_mujoco (arm path-planning in sim)

The motion stack can be driven against `unitree_mujoco`'s G1 scene over the **same DDS
interface** the real robot uses, so plan → retime → execute runs end-to-end in sim. This
file records the bring-up and a known sim-side limitation.

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
