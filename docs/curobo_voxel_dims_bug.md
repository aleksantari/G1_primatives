# cuRobo bug report: voxel collision kernel truncates float32 grid dims → skewed ESDF queries

**Status:** workaround shipped in this repo (`motion/collision_world.py: kernel_safe_dims`,
commit `a1dcc80`); local `~/repos/curobo` build left unpatched; NVIDIA not yet notified.
**Found:** 2026-07-02, debugging phantom `plan_grasp` "Start or End state in collision" failures.
**Affected version:** cuRobo V2 source build, commit `ec2bfa9f9ea2673abe00cf911ef39353ddb79c29`
(2026-06-12).

## Summary

`VoxelData` stores a voxel grid's shape as **float32 ratios** (`dims / voxel_size`) and the warp
collision kernel recovers the integer shape with a **truncating** `wp.int32()` cast. When the
`VoxelGrid` was authored by cuRobo's own `Mapper` (`BlockSparseESDFIntegrator.get_voxel_grid`),
the dims are products of a float32 voxel size, and the ratio can land **just below** the true
integer (e.g. `119.99999`). Truncation then yields `119` instead of `120`, which corrupts the
kernel's **flat-index strides** — every sphere-vs-voxel collision query beyond the first x-slab
of the grid silently samples a **skewed memory location**.

The failure is silent and severe: free space can read as **in collision** (phantom hits) and real
obstacles can read as **free** (invisible to planning). It is deterministic and unaffected by
seeds, activation distances, or any solver setting, which makes it exceptionally misleading to
debug — the checker's answers look plausible, they are just spatially wrong.

Notably, this breaks the **first-party pipeline**: `Mapper.compute_esdf()` →
`MotionPlanner.update_world(SceneCfg(voxel=[grid]))` → `plan_grasp`. A `VoxelGrid` authored by
hand in float64 (e.g. `dims = [120 * 0.01, ...]` in Python) usually lands *above* the integer
(`120.000007` → `120`) and works — so unit tests built on hand-authored grids can pass while the
Mapper path is broken.

## Reproduction (arithmetic core, no GPU needed)

```python
import numpy as np
k, v = 120, 0.01
mapper_dims = np.float32(k) * np.float32(v)          # 1.1999999  (integrator_esdf.get_voxel_grid)
ratio = np.float32(mapper_dims) / np.float32(v)      # 119.99999  (VoxelData.load_batch)
int(ratio)                                           # 119        (wp.int32 in the kernel)  ← BUG
```

Full-system reproduction (as observed live on a G1 head-depth ESDF, 120×120×100 @ 1 cm,
pelvis-frame pose `[0.4, 0, 0.2]`):

1. Fuse a depth frame with `Mapper`, `compute_esdf()`, load via
   `MotionPlanner.update_world(SceneCfg(voxel=[grid]))`.
2. `scene_collision_checker.data.voxels.params` reads `[119.99999, 119.99999, 100.0, 0.00999...]`.
3. Query `get_sphere_distance_raw` at a point whose grid cell provably holds a large positive
   (free) ESDF value → the kernel returns a positive collision cost (phantom hit), while a point
   inside a real fused obstacle can return 0 (free).
4. Overwrite `params[..., :3] = [120, 120, 100]` in place → every query immediately returns the
   correct answer. (This was the decisive confirmation.)

Observed spatial signature: with strides `119*99/99` against a `120*100/100` buffer, reads skew
progressively across the grid — in our scene the robot's left palm (free air, ESDF +78 mm)
"collided" with cells belonging to a ball ~15 cm away in y, while the ball itself read free.

## Root cause (three code locations)

1. **Dims authored in float32** — `curobo/_src/perception/mapper/integrator_esdf.py:737`
   (`get_voxel_grid`): `dims = [nx * voxel_size, ...]` where `voxel_size` is
   `self._esdf_voxel_size.item()` — the float32 representation of e.g. 0.01
   (`0.00999999977648`), so `120 * it = 1.1999999...`.
2. **Shape stored as a float ratio** — `curobo/_src/geom/data/data_voxel.py:258,438`
   (`create_cache` / `load_batch`): `grid_t = dims_t / size_t` in float32;
   `params_t = torch.cat([grid_t, size_t], dim=-1)`. `1.1999999 / 0.00999999977 = 119.9999995`.
3. **Truncating recovery in the kernel** — `curobo/_src/geom/data/data_voxel.py:1148-1150,1192-1194`
   (`is_obs_enabled` bounds path / `compute_local_sdf_with_grad`):
   `dims_x = wp.int32(obs_set.params[flat_idx, 0])` — C-style truncation, `119.9999995 → 119`.
   The wrong dims then drive both the trilinear sample coordinates (`half_x = dims*0.5`) and,
   fatally, the flat-index strides (`stride_x = dims_y*dims_z`, `stride_y = dims_z`) used to read
   the feature buffer, whose true layout is the original `120*100 / 100`.

## Suggested upstream fix

Any one of these closes the bug; (a) is the most robust:

- **(a) Round, don't truncate, in the kernel:**
  `dims_x = wp.int32(wp.round(obs_set.params[flat_idx, 0]))` (and y/z, both call sites). The
  ratio is integral by construction, so rounding is always exact.
- **(b) Store exact shapes:** compute `grid_t = torch.round(dims_t / size_t)` in
  `create_cache`/`load_batch` (data_voxel.py:258,438), so the kernel's truncation is safe.
- **(c) Author exact dims:** in `get_voxel_grid`, compute `dims` in float64 or as
  `grid_shape * float(voxel_size_f64)` such that the downstream float32 ratio lands ≥ the integer.
  (Weakest — still leaves the truncation landmine for user-authored grids.)

## Workaround in this repo (active)

`g1_primitives/motion/collision_world.py: kernel_safe_dims(dims, voxel_size)` re-authors
every `VoxelGrid`'s dims a **quarter-voxel high** (`(k + 0.25) * voxel_size`) before the planner
loads it — cuRobo's Python-side `round()` (`VoxelGrid.get_grid_shape`) still recovers `k`, while
the kernel's float32 ratio (`k + 0.25`) truncates to `k`. Applied to both the Mapper output
(`esdf_from_depth`) and the hand-built empty build grid. GPU-free regression tests replicating
cuRobo's exact float32 arithmetic live in `tests/test_collision_world.py`
(`test_kernel_safe_dims_*`); the `..._regression_mapper_dims_are_unsafe` test will start failing
if an upstream cuRobo update fixes the arithmetic — at which point `kernel_safe_dims` can retire.

## Debugging trail (what it looked like from the outside)

Every `plan_grasp` with the live ESDF loaded failed `"Planning to approach pose failed"` for
every candidate; the truth-test (`world_check`, which queries the checker itself and therefore
faithfully reproduced the bug) reported the start config's palm spheres up to 28.9 mm "inside"
the world. Meanwhile the fused world was demonstrably clean at the palm (direct reads of the
feature tensor: +64…+78 mm free), penetrations were byte-identical across self-filter margins
0.02/0.04 and seed settings 32/4 vs 128/16, and real obstacles (ball/truck) queried free. The
contradiction between the checker's answer and a direct read of its own buffer at the same
coordinates is the diagnostic fingerprint of this bug.
