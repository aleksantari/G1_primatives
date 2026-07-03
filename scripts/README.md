# scripts/ — bring-up checks, examples, and dev tools

Three tiers, run in order on a new rig:

- **`checks/`** — the numbered bring-up ladder. Run 01→05 against `--target sim`
  first, then `--target real` (suspended robot, e-stop in reach). Each step
  validates one seam before the next depends on it.
- **`examples/`** — end-to-end demos of the `g1_primitives` facade: the reference
  code an agent-host developer starts from.
- **`tools/`** — perception / grasp / collision-world development utilities
  (no arm motion unless stated).

All scripts run inside the `g1_curobo` env: `bash -ic 'use_conda g1_curobo && python scripts/<tier>/<name>.py ...'`.
Sim targets need `CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml` when they
touch DDS (the script warns if it is unset).

| script | does | moves the robot? |
|---|---|---|
| `checks/01_dds.py` | read-only arm + dex3 state streams | no |
| `checks/02_camera.py` | head-camera RGB feed | no |
| `checks/03_depth.py` | head-camera depth feed + stats | no |
| `checks/04_hands.py` | close/open hand primitives + state verdict | hands (+ arms home on connect) |
| `checks/05_move.py` | planned home + optional wrist lift | **arms** |
| `examples/01_hello_motion.py` | home → move → close → open → home | **arms + hands** |
| `examples/02_pick.py` | full pick-and-lift (grasp source + SAM3 + plan_grasp) | **arms + hands** |
| `tools/segment.py` | SAM3 mask + object cloud (live / `--frame` / `--image`) | no |
| `tools/grasp_preview.py` | live SAM3 + GraspGenX preview loop in viser | no |
| `tools/graspgen_viz.py` | offline cloud → GraspGenX → viser | no |
| `tools/capture_frame.py` | save an RGB-D + pose `.npz` for offline replay | no |
| `tools/check_world.py` | inspect the depth-ESDF collision world | no |
| `tools/validate_perception.py` | sim GT gate: score the full perception stack vs rt/sim_state | no |
| `tools/derive_tool_transform.py` | provenance: derive the GraspGenX→wrist transform | no |
| `tools/hand_loop_diag.py` | raw dex3 command→state loop diagnosis (sim) | hands |

External services for the grasp pipeline (own repos/envs, see the root README):
GraspGenX ZMQ `:5556`, SAM3 ZMQ `:5557`, viser GUI `:8080`.

## Old → new map (pre-restructure names)

| old | new |
|---|---|
| `01_check_dds.py` | `checks/01_dds.py` |
| `02_check_image.py` | `checks/02_camera.py` |
| `03_hands.py` | `checks/04_hands.py` |
| `04_move.py` | `checks/05_move.py` |
| `05_mvp_demo.py` | `examples/01_hello_motion.py` |
| `08_check_depth.py` | `checks/03_depth.py` |
| `09_graspgen.py` | `examples/02_pick.py` (slimmed: `--latency`, `--show-spheres` and the vestigial `--grasp-roll/pitch/yaw-deg` sweep flags were dropped — profiling lives in `tools/grasp_preview.py`, sphere overlays in `tools/check_world.py`, the transform recipe in `tools/derive_tool_transform.py`) |
| `10_segment.py` | `tools/segment.py` |
| `10_graspgen_viz.py` | `tools/graspgen_viz.py` |
| `10_grasp_preview.py` | `tools/grasp_preview.py` |
| `11_capture_frame.py` | `tools/capture_frame.py` |
| `12_check_world.py` | `tools/check_world.py` |
| `derive_graspgenx_tool_transform.py` | `tools/derive_tool_transform.py` |
| `hand_diag.py` | `tools/hand_loop_diag.py` |
| `_rig.py` | gone — target plumbing lives in `g1_primitives.connect(target)`; the console helpers (`confirm`/`Viewer`/`do_move`/`add_target_arg`/`dds_for`) in `g1_primitives.api.console` |
| `06_detect.py`, `07_pick_place.py` | deleted (AprilTag era) |
