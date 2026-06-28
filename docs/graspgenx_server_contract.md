# GraspGenX ZMQ server ↔ client contract (+ how to use it effectively)

**Status:** server upgraded to GraspMoE / top-down 2026-06-28 (`protocol_version: 2`).
The GraspGenX server lives in a **separate repo** at `/home/santari/repos/GraspGenX`; this
repo (`G1_classical_manip`) is the **client**. Wire = msgpack over ZMQ REQ/REP.

**For:** the agent wiring grasps into the pick pipeline — so the client and its callers
request grasps from the server effectively (right planner, right frame, right fallbacks).

---

## TL;DR

- Send a **pelvis-frame, meters** `(N,3)` cloud → get back **`(K,4,4)` SE(3) grasps + `(K,)`
  confidences + `branch_tags`**, ranked best-first, **in the same (pelvis) frame** you sent.
- **`planner`** picks the candidate source:
  - `diffusion` — diffusion model only (free-form orientations).
  - `graspmoe` — diffusion **∪** OBB top-down/side, all discriminator-scored (default).
  - `topdown` — GraspMoE but **OBB-only** (top-down/side; diffusion grasps dropped).
- For **"grab from above"**: `planner: topdown` + `obb_density: dense` (current config default).
- The thin client (`grasp/graspgenx_client.py`) imports **nothing** from the `graspgenx`
  package — it's a pure wire shim. Keep it that way (importing graspgenx pulls torch + a
  multi-GB auto-download).

---

## Wire contract (msgpack, `msgpack_numpy.patch()` so numpy travels natively)

Source of truth: `GraspGenX/graspgenx/serving/zmq_server.py`.

### Actions
| request | response |
|---|---|
| `{"action": "health"}` | `{"status": "ok"}` |
| `{"action": "metadata"}` | `{protocol_version, default_gripper, loaded_grippers[], planner_default, planners[], model{...}, assets_dir}` |
| `{"action": "infer", ...}` | see below |
| (any error) | `{"error": "<Type>: <msg>"}` → client raises `RuntimeError` |

### `infer` request
| field | type | default | meaning |
|---|---|---|---|
| `point_cloud` | (N,3) f32 | **required** | object cloud in the **sender's frame** (we send pelvis, meters) |
| `gripper_name` | str | server default | `unitree_g1` (= Dex3) |
| `planner` | str | `"graspmoe"` | `diffusion` \| `graspmoe` \| `topdown` |
| `obb_density` | str | `"dense"` | `sparse`/`dense` = top-down only; `dense-topandside` = + side faces |
| `skip_obb_rule` | str | `"auto"` | `never` = force OBB even when the object is wider than the gripper |
| `num_grasps` | int | 200 | diffusion samples per pass (diffusion/graspmoe only) |
| `grasp_threshold` | float | -1.0 | discriminator-score floor; -1 = none, rely on top-k |
| `topk_num_grasps` | int | 100 | keep top-K by confidence (applied **after** the topdown filter) |

### `infer` response
| field | type | meaning |
|---|---|---|
| `grasps` | (K,4,4) f32 | SE(3), **sender's frame**, **sorted by confidence desc** |
| `confidences` | (K,) f32 | discriminator score ∈ [0,1] (predicted grasp validity) |
| `branch_tags` | list[str] | per-grasp `"obb"` (top-down/side) or `"diff"` (diffusion) |
| `gripper_name` | str | gripper used |
| `planner` | str | planner used |
| `timing` | {infer_ms: float} | server inference time |

Empty result = `grasps` shape `(0,4,4)`, `confidences` `(0,)`, `branch_tags` `[]`.

---

## Planner semantics & effective use

- **`topdown` runs diffusion internally and discards it** — it's **not faster** than
  `graspmoe`, just filtered. (A real speedup would need a change in `graspmoe.run_graspmoe`.)
- **Preference vs fallback (the key tradeoff):**
  - `topdown` → **strong top-down, NO fallback.** If no top-down grasp is reachable it returns
    nothing (or empty if the OBB branch was skipped — see below).
  - `graspmoe` → diffusion grasps remain as fallback, but **no top-down preference** (the client
    picks the highest discriminator score, which may be a side/diffusion grasp).
  - Getting *both* preference and fallback needs a **client-side re-rank** (not built yet).
- **`skip_obb_rule: auto` can silently empty a `topdown` result.** The OBB branch auto-skips
  when **every** OBB extent exceeds the gripper jaw width (`unitree_g1` sweep_volume[0] = 0.10 m).
  Big objects → OBB skipped → `topdown` returns nothing. Set `skip_obb_rule: never` for large
  objects.
- **`obb_density`:** `sparse` = single top-down pose at the centroid; `dense` = top-down sweep
  along the object's long axis (more candidates); `dense-topandside` = also adds 4 horizontal
  side approaches (**not** what you want for "from above").
- **Latency:** `graspmoe`/`topdown` (OBB sweep + extra discriminator scoring) are slower than
  `diffusion`; still under the client's 60 s timeout.

---

## Frame & grasp-pose conventions (get these right)

- **Frame in = frame out.** The server centers/un-centers internally (diffusion in `sample()`,
  `grasp_server.py:265`+`:301`; OBB placed via `pc_center` in the input frame), so a pelvis-frame
  cloud yields pelvis-frame grasps. **Send pelvis, meters; receive pelvis.** No client centering.
- **Grasp frame:** GraspGenX convention is **+Z = approach into the object**, **+X = the
  jaw-closing / opposition axis** (see `grasp/tool_transform.py`). The approach axis of a grasp
  is `grasp_pose.rotation[:, 2]`.
- **Top-down = approach pointing down:** a top-down grasp has `R[:, 2] ≈ [0, 0, -1]`. The OBB
  branch defines "down" as the **input cloud's −Z**, i.e. **pelvis −Z** — which ≈ gravity-down for
  the upright-suspended G1. If the pelvis ever tilts, top-down would track pelvis-down, not
  gravity-down (future fix: rotate the cloud gravity-up before infer, grasps back after).

---

## Client wiring in THIS repo

- **`grasp/graspgenx_client.py`** — `GraspGenXClient.infer(cloud, gripper_name, num_grasps,
  grasp_threshold, topk_num_grasps, planner, obb_density, skip_obb_rule) -> (grasps, conf,
  branch_tags)`. Pure `msgpack`/`zmq`/`numpy`. `EXPECTED_PROTOCOL_VERSION = 2`; `metadata()`
  warns on a server/client mismatch — keep both in lockstep.
- **`grasp/graspgenx_source.py`** — `GraspGenXGraspSource` reads `planner`/`obb_density`/
  `skip_obb_rule` from `grasp.yaml: graspgenx` and stashes each grasp's tag on
  `GraspCandidate.extra["branch_tag"]` (debug/viz). Selection downstream
  (`09_graspgen.py::_select_candidate`) is **first reachable in confidence order**.
- **`configs/grasp.yaml` → `graspgenx`** — `planner` / `obb_density` / `skip_obb_rule` plus
  `num_grasps` / `topk` / `grasp_threshold` / `voxel_m` and the tool-transform constants.
  Current default: `planner: topdown`, `obb_density: dense`.
- **Viz:** `branch_tags` is first-class, so `viz/grasp_viz.py` can color `"obb"` (top-down) vs
  `"diff"` grasps distinctly. See `docs/grasp_visualization.md`.

### Minimal call
```python
from g1_classical_manip.grasp.graspgenx_client import GraspGenXClient
with GraspGenXClient(host="127.0.0.1", port=5556) as c:
    grasps, conf, tags = c.infer(cloud_xyz, gripper_name="unitree_g1", planner="topdown")
# grasps: (K,4,4) pelvis-frame, conf-desc; tags[i] in {"obb","diff"}
```

---

## Running the server (on the GPU box, GraspGenX repo)

Needs the `serve` extra (`pyzmq msgpack msgpack-numpy`), a GPU, and the checkpoints
(auto-downloaded into `ext/graspgenx_checkpoints/`).

```bash
# from /home/santari/repos/GraspGenX
python client-server/graspgenx_server.py \
    --config    ext/graspgenx_checkpoints/release \
    --assets_dir ext/gripper_descriptions/gripper_descriptions/assets \
    --default_gripper unitree_g1 --port 5556
```

`--config` is the checkpoint **root** (contains `gen/` + `dis/`). `--assets_dir` must contain
`x_grippers/` (where `unitree_g1` lives). First request per gripper loads the model (slow);
subsequent are hot. One sampler is cached per gripper.

---

## GraspGenX files to reference (when you need guidance)

| File (under `/home/santari/repos/GraspGenX`) | What it tells you |
|---|---|
| `graspgenx/serving/zmq_server.py` | **The contract** — `_handle_infer` / `_handle_metadata`, exact request/response handling. |
| `graspgenx/samplers/planner.py` | `run_planner_on_object` — the dispatch the server calls; **all `moe_*` params** (yaws, z-offsets, spacing) if we ever expose more. |
| `graspgenx/samplers/graspmoe.py` | `run_graspmoe`, `_run_obb_branch`, `_world_aligned_top_down_grasp` — the OBB **top-down geometry**, the `obb_density` modes, and the **skip-OBB rule**. |
| `graspgenx/grasp_server.py` | `GraspGenXSampler.sample()`/`run_inference` — the **frame contract** (center @265, un-center @301), confidence/threshold/top-k. |
| `graspgenx/x_grippers.py` | gripper config (`sweep_volume`, `gripper_type`, fingertip `depth`) — `unitree_g1` = Dex3 (`type: revolute_3f`). |
| `end2end/e2e_grasp_demo.py` (`run_graspgen`, ~L515-634) | Canonical end-to-end pattern: sample → center → `run_planner_on_object` → **`topdown`/`obb_only` filter** (L587-621) → un-center. |
| `scripts/demo_object_pc.py` | Reference interactive usage + the `--moe_*` CLI flags and their defaults. |
| `client-server/README.md` | The original (pre-upgrade) server/client doc. |

---

## Effective-use checklist

- [ ] Cloud is **pelvis-frame, meters**, single-object (segment first). Grasps come back pelvis-frame.
- [ ] Want from-above? `planner: topdown`, `obb_density: dense`. Accept the **no-fallback** behavior,
      or use `graspmoe` if you need a fallback.
- [ ] Object likely wider than the 10 cm jaw? Set `skip_obb_rule: never`, else `topdown` may return empty.
- [ ] Treat `confidences` as the rank; the server already sorts desc and applies `topk`.
- [ ] Keep client `EXPECTED_PROTOCOL_VERSION` in lockstep with the server's `PROTOCOL_VERSION`.
- [ ] Never `import graspgenx` on the client — it's a wire shim by design.

Related: `docs/grasp_visualization.md` (viewing grasps), `docs/depth_integration_handoff.md`
(the depth → cloud source).
