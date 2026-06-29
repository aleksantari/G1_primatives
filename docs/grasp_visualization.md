# GraspGenX grasp visualization — what it is & how it's wired

**Status:** implemented 2026-06-26. Static checks + a loader unit check pass in
`g1_curobo`; the two **viser GUI paths still need a manual run** (live GraspGenX
server + a real cloud + a browser) — see [Still to verify](#still-to-verify).
Opt-in, **off by default**. The GraspGenX server is **not** modified by any of this.

**For:** the next agent touching the grasp pipeline or wanting to see GraspGenX
grasps against the object cloud (the client-side equivalent of GraspGenX's
`scripts/demo_object_pc.py`).

---

## TL;DR

- New client-side **viser** viewer: point cloud + ranked 6-DoF grasps (gripper-shaped
  markers, colored by confidence) + the gripper **collision-mesh overlay** + a
  confidence threshold slider. Renders in the **pelvis frame**, so grasps sit on the cloud.
- Two ways in:
  - **Integrated** (during a pick): `python scripts/09_graspgen.py --source graspgenx --visualize ...`
  - **Standalone** (inspect a saved cloud, no robot): `python scripts/10_graspgen_viz.py --pcd <cloud>`
- Lives **entirely on the client**. The ZMQ server stays a headless inference box.
- Imports **nothing** from the `graspgenx` package (that would pull torch + a multi-GB
  auto-download). The viser primitives + gripper loader are vendored, exactly like
  `grasp/graspgenx_client.py` already vendors the ZMQ client.
- Colors: by default a **confidence gradient** (red→green) for all grasps; when GraspGenX
  protocol-v2 `branch_tags` are passed, grasps are instead colored by branch — **amber** = OBB
  (top-down) grasp, **purple** = diffusion grasp. **blue** = model's top-confidence grasp,
  **green** = the grasp cuRobo actually selected. The `[GraspViz]` summary line reports the
  obb/diff split.

Open the viewer at `http://localhost:8080` (forward it if headless:
`ssh -N -L 8080:localhost:8080 <host>`).

---

## Why client-side (the decision, so you don't re-litigate it)

| | Reason |
| --- | --- |
| Server stays pure | Viz on the server would re-couple a web GUI into the REQ/REP inference service. |
| Only the client has the story | The server sees a bare centered cloud + grasps. The client has the **pelvis cloud, all ranked candidates, which grasp cuRobo chose, reachability** — the stuff worth looking at. |
| No new deps | `viser`, `trimesh`, `yourdfpy` are already in `g1_curobo` via `requirements-curobo.txt`. Nothing to install. |
| Hard dependency wall | This repo is **numpy>=2**; GraspGenX pins **numpy==1.26.4** (+ old `diffusers`/`hf-hub`). You *cannot* import graspgenx into this env. The msgpack-numpy wire crosses that boundary; a code import would not. |

---

## How to use

### Integrated (see grasps during a real pick)

```bash
bash -ic 'use_conda g1_curobo && python scripts/09_graspgen.py \
    --target real --source graspgenx --segment interactive --visualize'
```

`--visualize` flips `graspgenx.visualize.enabled` on for that run and rebuilds the
grasp source. Sequence: the source draws the cloud + all ranked grasps right after
inference; after `_select_candidate` picks the reachable one, the script overlays it
in **green**. The viser server runs in the background — it does **not** block the
operator-gated step prompts.

### Standalone (inspect a saved cloud, no robot, no motion)

```bash
bash -ic 'use_conda g1_curobo && python scripts/10_graspgen_viz.py \
    --pcd captures/cloud.npy --gripper_name unitree_g1'
```

Loads an `(N,3)` pelvis-frame cloud (`.npy/.npz/.xyz/.ply/.obj`), hits the GraspGenX
server via the existing thin client, shows the result, and blocks on the viser server
(Ctrl+C to quit). This is the tool for iterating on `num_grasps` / `topk` /
`grasp_threshold` / gripper without running the arm. Threshold tuner is on here.

### Config (`configs/grasp.yaml` → `graspgenx.visualize`)

```yaml
visualize:
  enabled: false          # opt-in; --visualize flips it for one run
  port: 8080
  show_mesh: true         # overlay the gripper mesh at the top/chosen grasp
  threshold_tuner: false  # confidence slider; OFF in the gated pick flow, ON in 10_graspgen_viz
  max_markers: 100        # cap markers (viser slows with thousands)
  gripper_asset_dir: "assets/grippers"   # <dir>/<gripper_name>/{config.json, coll_mesh.obj}
```

---

## How it's wired (data flow)

```
GraspGenXGraspSource.grasps()                 # grasp/graspgenx_source.py
  depth -> mask -> deproject -> pelvis cloud
  client.infer(cloud) -> grasps (K,4,4), conf (K,)   [pelvis frame]
  └─ if self.viz: viz.show_candidates(cloud.points, grasps, conf)   # cloud + all grasps

09_graspgen.py
  chosen = _select_candidate(...)              # highest-confidence reachable grasp
  └─ viz.mark_chosen(chosen.grasp_pose.homogeneous)   # green overlay

factory._build_grasp_source()                  # builds GraspViz from config, injects viz=
```

**Files** (all new/edited in this repo; nothing in GraspGenX):

| File | Role |
| --- | --- |
| `g1_classical_manip/viz/viser_primitives.py` | Drawing fns **vendored** from GraspGenX `utils/viser_utils.py` (viser/trimesh/numpy only). |
| `g1_classical_manip/viz/gripper_geom.py` | Torch-free loader: `config.json` → `sweep_volume`, `coll_mesh.obj` → mesh. Graceful fallback. |
| `g1_classical_manip/viz/grasp_viz.py` | `GraspViz`: `show_candidates()`, `mark_chosen()`, `spin()`. |
| `g1_classical_manip/viz/__init__.py` | Lazy `GraspViz` (importing the package does **not** pull viser). |
| `grasp/graspgenx_source.py` | `+viz=` param; one guarded `show_candidates(...)` after inference. |
| `factory.py` | `_build_grasp_viz()` builds it from config; injected in the `graspgenx` branch. |
| `configs/grasp.yaml` | `graspgenx.visualize` block. |
| `scripts/09_graspgen.py` | `--visualize` flag + the `mark_chosen` overlay. |
| `scripts/10_graspgen_viz.py` | Standalone inspector. |
| `assets/grippers/unitree_g1/` | `config.json` + `coll_mesh.obj`, copied from the gripper_descriptions tree. |

---

## Frames (the thing to get right)

Everything is rendered **directly in the pelvis frame — no centering**. The cloud is
already pelvis-frame (`deproject_depth` with `T_pelvis_camera`); the server centers/
uncenters internally and returns grasps in the same frame it received (verified:
GraspGenX `grasp_server.py sample()` adds the centroid back). The gripper mesh is in the
canonical grasp frame and is transformed by each grasp pose, so it lands where the hand
will go.

> **Sanity check:** if the gripper mesh renders at the origin instead of on the cloud,
> that's a **frame-contract** bug (the server stopped returning grasps in the input
> frame — e.g. a botched server upgrade), **not** a viz bug. See the wider integration
> note about the GraspMoE upgrade preserving the center→plan→un-center wrapper.

---

## Gripper asset

`assets/grippers/unitree_g1/` holds `config.json` (the `sweep_volume` that shapes the
markers) and `coll_mesh.obj` (the overlay, 12,130 verts). Copied from
`GraspGenX/ext/gripper_descriptions/.../assets/x_grippers/unitree_g1/`. The loader
prefers `coll_mesh.obj`, then `vis_mesh.obj`, else falls back to a dummy box and
wireframe-only markers — it never crashes the grasp flow on a missing asset. To add a
gripper, drop its `<name>/{config.json, coll_mesh.obj}` in here.

Note `unitree_g1` is the **Dex3-1** 3-finger hand (`config.json` → `"type": "revolute_3f"`),
despite the upstream gripper_descriptions README mislabeling it.

---

## Verified

- All new/edited files compile; **zero** `graspgenx` imports under `viz/`.
- Lazy import holds: `import g1_classical_manip.viz` does **not** load viser.
- Loader against the real asset (`g1_curobo`): `sweep_volume (6,)`, `has_mesh=True`, 12,130 verts.
- GraspGenX repo untouched by this work.

## Still to verify

Need a **live GraspGenX server + a real pelvis cloud + a browser** (couldn't be run headless):

1. **Standalone:** server up → `scripts/10_graspgen_viz.py --pcd <cloud>` → viser at :8080
   shows cloud + confidence-colored markers + blue top grasp + gripper mesh; slider toggles markers.
2. **Integrated:** `scripts/09_graspgen.py --source graspgenx --segment none --visualize` → cloud +
   ranked grasps + **green** chosen grasp before the approach move. This also confirms the
   pelvis-frame contract (mesh on the cloud, not at the origin).

---

## Notes / gotchas

- `threshold_tuner` is **off** in the pick flow by default (keep the operator's attention on
  the gated steps); the standalone tool turns it on.
- `max_markers` caps how many grasps are drawn (top-N by confidence). Bump it if you want all.
- E501 line-length warnings from the IDE are **repo-wide pre-existing** (this repo runs ~90-col
  lines); the new code matches that, not the 79-col default.
- The cloud source is the head-cam depth stream — see `docs/depth_integration_handoff.md`.
