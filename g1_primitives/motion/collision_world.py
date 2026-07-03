"""Depth -> cuRobo ESDF collision world (the Mapper / voxel path).

A thin wrapper over cuRobo V2's ``Mapper``: a head-camera depth frame (+ pelvis-frame extrinsic
and pinhole intrinsics) is fused into a block-sparse TSDF and turned into a Euclidean Signed
Distance Field (ESDF) ``VoxelGrid`` that the ``MotionPlanner`` queries for collision-aware
planning. The SAME path runs in sim (the Isaac front_camera depth stream) and on real (the ZED
head) -- only the depth source differs.

Why this is correct by construction (all verified against cuRobo source):
  * cuRobo's camera frame is OpenCV optical (``+x`` right, ``+y`` down, ``+z`` forward), IDENTICAL
    to ``perception/depth.deproject_depth`` and our ``Frames.T_pelvis_camera`` optical convention,
    so the camera pose plugs in directly -- no extra axis swap (the volumetric_mapping example's
    ``y_to_z`` was only adapting the Sun3D dataset convention).
  * We pass depth in METERS with ``depth_to_meter=1.0`` (no unit ambiguity; the integrator's
    depth_min/max are in meters).
  * ``compute_esdf()`` returns a ``VoxelGrid`` already placed at the grid centre in the camera-pose
    frame (= pelvis), with ``dims``/``voxel_size``/``feature_tensor`` set (ESDF: ``>0`` free,
    ``<0`` inside an obstacle). It plugs straight into ``SceneCfg(voxel=[grid])`` /
    ``planner.update_world`` (proven by curobo tests/_src/motion/test_motion_planner_esdf.py).

The grid extent should be a multiple of ``esdf_voxel_size`` so the ESDF grid shape matches the
empty grid allocated at planner build (``empty_esdf_grid``) exactly.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from g1_primitives.spatial.pose import Pose


def esdf_grid_shape(extent_m, esdf_voxel_size: float) -> Tuple[int, int, int]:
    """(nx, ny, nz) voxel counts for an ESDF of this extent -- cuRobo's ``ceil`` convention
    (Mapper.__init__: ``ceil(extent_esdf / esdf_voxel_size)`` per axis)."""
    v = float(esdf_voxel_size)
    return tuple(int(math.ceil(float(e) / v)) for e in extent_m)


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Binary-dilate an (H,W) bool mask by ``px`` pixels (Chebyshev/square kernel) -- pure numpy
    (max over shifted views), no cv2/scipy dependency. Used to grow the SAM3 object mask a few
    pixels before cutting the object out of the depth, so mask-edge depth bleed doesn't leave an
    occupied rind around the erased object. px <= 0 returns the input unchanged."""
    m = np.asarray(mask, bool)
    if px <= 0 or not m.any():
        return m
    out = m.copy()
    for _ in range(int(px)):                 # one 3x3 (8-neighbour) dilation per iteration
        d = out.copy()
        d[1:, :] |= out[:-1, :]
        d[:-1, :] |= out[1:, :]
        d[:, 1:] |= out[:, :-1]
        d[:, :-1] |= out[:, 1:]
        d[1:, 1:] |= out[:-1, :-1]
        d[1:, :-1] |= out[:-1, 1:]
        d[:-1, 1:] |= out[1:, :-1]
        d[:-1, :-1] |= out[1:, 1:]
        out = d
    return out


def kernel_safe_dims(dims, voxel_size: float):
    """Re-author a VoxelGrid's ``dims`` so cuRobo's collision kernel recovers the TRUE voxel
    counts. cuRobo BUG (V2 ``VoxelData.load_batch`` + the warp collision kernel): the kernel's
    grid shape is ``wp.int32(float32(dims) / float32(voxel_size))`` -- a TRUNCATING cast. The
    Mapper emits ``dims = grid_shape * float32(voxel_size)`` (e.g. 120 * 0.00999999977 ->
    1.1999999), whose float32 ratio lands just BELOW the integer (119.99999 -> 119). The wrong
    dims corrupt the kernel's flat-index STRIDES (119*99 vs the buffer's 120*100), so every
    sphere-vs-voxel query above the first x-slab samples a skewed location: phantom collisions
    in free space AND real obstacles reading free (verified live 2026-07-02: the left palm 'hit'
    the blue ball's cells at half its y; the ball itself read free). Fix: place dims a QUARTER
    VOXEL high -- ``round()`` (cuRobo's python-side ``get_grid_shape``) still recovers k, while
    the kernel's float32 ratio (k + 0.25) truncates to k. Idempotent."""
    v = float(voxel_size)
    return [(round(float(d) / v) + 0.25) * v for d in dims]


def empty_esdf_grid(grid_center, extent_m, esdf_voxel_size: float, device: str = "cuda:0"):
    """A free-space ESDF ``VoxelGrid`` (all distances large-positive). Built once at planner
    creation so cuRobo allocates the voxel collision channel + its capacity; ``update_world``
    later swaps in the live depth ESDF. dims = grid_shape * voxel_size so it matches exactly the
    grid ``Mapper.compute_esdf`` produces for the same extent/voxel (get_voxel_grid sets
    ``dims = grid_shape * voxel_size``)."""
    import torch
    from curobo.scene import VoxelGrid
    nx, ny, nz = esdf_grid_shape(extent_m, esdf_voxel_size)
    v = float(esdf_voxel_size)
    dims = kernel_safe_dims([nx * v, ny * v, nz * v], v)
    feat = torch.full((nx, ny, nz), 1.0, dtype=torch.float16, device=device)  # +1 m = free space
    pose = [float(grid_center[0]), float(grid_center[1]), float(grid_center[2]), 1.0, 0.0, 0.0, 0.0]
    return VoxelGrid(name="head_esdf", pose=pose, dims=dims, voxel_size=v,
                     feature_tensor=feat, feature_dtype=torch.float16)


class EsdfMapper:
    """Reusable cuRobo ``Mapper``: ``esdf_from_depth`` -> a fresh single-view ESDF ``VoxelGrid``
    in the pelvis frame. The Mapper is built lazily (first build JIT-compiles the warp kernels)
    and reused; each call clears the dynamic map and integrates the current head frame, so the
    ESDF reflects only what the camera sees right now (a static grasp scene)."""

    def __init__(self, grid_center, extent_m, esdf_voxel_size: float = 0.02,
                 tsdf_voxel_size: float = 0.01, image_hw=(480, 640),
                 depth_min_m: float = 0.1, depth_max_m: float = 2.0, device: str = "cuda:0"):
        import torch
        from curobo.perception import Mapper, MapperCfg
        self._torch = torch
        self._Mapper = Mapper
        self._device = device
        self.grid_center = [float(c) for c in grid_center]
        self.extent_m = [float(e) for e in extent_m]
        self.esdf_voxel_size = float(esdf_voxel_size)
        self._cfg = MapperCfg(
            extent_meters_xyz=tuple(self.extent_m),
            voxel_size=float(tsdf_voxel_size),
            esdf_voxel_size=self.esdf_voxel_size,
            extent_esdf_meters_xyz=tuple(self.extent_m),
            grid_center=torch.tensor(self.grid_center, dtype=torch.float32),
            depth_minimum_distance=float(depth_min_m),
            depth_maximum_distance=float(depth_max_m),
            num_cameras=1,
            image_height=int(image_hw[0]),
            image_width=int(image_hw[1]),
            device=device,
        )
        self._mapper = None

    def empty_grid(self):
        """The free-space grid matching this mapper's extent/voxel (for the planner build)."""
        return empty_esdf_grid(self.grid_center, self.extent_m, self.esdf_voxel_size, self._device)

    def _ensure(self):
        if self._mapper is None:
            self._mapper = self._Mapper(self._cfg)
        return self._mapper

    def esdf_from_depth(self, depth_mm: np.ndarray, intrinsics: dict, T_pelvis_camera: Pose,
                        robot_filter=None, exclude_mask=None):
        """Head depth (H,W float32 MILLIMETERS) -> a cuRobo ESDF ``VoxelGrid`` (pelvis frame).

        ``intrinsics``: {fx,fy,cx,cy} (the depth map's pinhole). ``T_pelvis_camera``: our
        ``spatial.pose.Pose`` = the camera OPTICAL frame in pelvis (Frames.T_pelvis_camera).
        ``robot_filter``: optional ``CameraObservation -> filtered_depth_tensor`` callback that
        REMOVES the robot's own geometry from the depth before fusing (else the head camera fuses
        the arm into the world and the grasp starts inside a copy of itself). See
        CuroboArmPlanner.robot_depth_filter (cuRobo RobotSegmenter).
        ``exclude_mask``: optional (H,W) bool, True = pixels to CUT from the depth before fusion
        (the TARGET object's SAM3 mask, pre-dilated) -- zeroed like invalid depth, so the object
        never enters the world and the gripper can reach it (the GraspGenX end2end reference's
        object-out design). Independent of and composable with ``robot_filter``."""
        torch = self._torch
        from curobo.types import CameraObservation, Pose as CuPose
        mapper = self._ensure()

        # fresh single-view scene: clear the dynamic map over the full grid AABB, then integrate.
        gc, ex = self.grid_center, self.extent_m
        bmin = torch.tensor([gc[i] - ex[i] / 2 for i in range(3)], device=self._device)
        bmax = torch.tensor([gc[i] + ex[i] / 2 for i in range(3)], device=self._device)
        try:
            mapper.clear_region(bmin, bmax)
        except Exception:                       # noqa: BLE001 - first call has nothing to clear
            pass

        d = np.asarray(depth_mm, np.float32) / 1000.0                 # mm -> meters
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)         # invalid -> 0 (< depth_min -> rejected)
        if exclude_mask is not None:
            m = np.asarray(exclude_mask, bool)
            if m.shape == d.shape:
                d[m] = 0.0                                            # cut the object: 0 < depth_min
            else:
                print(f"[EsdfMapper] exclude_mask shape {m.shape} != depth {d.shape} -- ignored")
        depth_t = torch.as_tensor(d, dtype=torch.float32, device=self._device).unsqueeze(0)  # (1,H,W)
        # The TSDF integrator REQUIRES rgb (num_cameras,H,W,3) uint8 even though colour is
        # irrelevant to the ESDF -- feed zeros (matching the camera's H,W = the Mapper's image_hw).
        rgb_t = torch.zeros((depth_t.shape[0], depth_t.shape[-2], depth_t.shape[-1], 3),
                            dtype=torch.uint8, device=self._device)
        fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
        cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
        K = torch.tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]],
                         dtype=torch.float32, device=self._device)   # (1,3,3)
        pos = torch.tensor([list(T_pelvis_camera.translation)], dtype=torch.float32, device=self._device)
        quat = torch.tensor([list(T_pelvis_camera.quaternion_wxyz())], dtype=torch.float32, device=self._device)
        obs = CameraObservation(depth_image=depth_t, rgb_image=rgb_t, intrinsics=K,
                                pose=CuPose(position=pos, quaternion=quat), depth_to_meter=1.0)
        if robot_filter is not None:                 # zero out the robot's own pixels (self-view)
            filt = robot_filter(obs)
            if filt is not None:
                obs.depth_image = filt
        mapper.integrate(obs)
        grid = mapper.compute_esdf()
        if not getattr(grid, "name", None):
            grid.name = "head_esdf"
        # the Mapper authors dims in float32 (grid_shape * float32(voxel_size)) -- exactly the
        # form that trips the cuRobo kernel's truncating int cast. See kernel_safe_dims.
        grid.dims = kernel_safe_dims(grid.dims, grid.voxel_size)
        return grid

    def occupied_points(self) -> np.ndarray:
        """(N,3) pelvis-frame centres of the currently-occupied TSDF voxels (the fused surface) --
        for INSPECTING the world independently of the planner: confirm the geometry sits where it
        should (vs the raw deproject cloud / GT object) and reveal anything that should NOT be there
        (e.g. the robot's own arm, baked in because the head camera sees it). Call after
        esdf_from_depth. Empty array if nothing has been integrated."""
        if self._mapper is None:
            return np.zeros((0, 3), np.float32)
        vox = self._mapper.integrator.extract_occupied_voxels(surface_only=False)
        if vox is None or len(vox) == 0:
            return np.zeros((0, 3), np.float32)
        return vox.centers.detach().cpu().numpy().astype(np.float32)
