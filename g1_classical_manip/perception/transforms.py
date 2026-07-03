"""The single owner of frame math (CLAUDE.md rule 2). Everything is a numpy
``spatial.pose.Pose`` in the pelvis frame; kinematics come from cuRobo (no pinocchio).

Perception frame chain:
    camera optical frame --T_pelvis_camera (cuRobo FK to d435_link + body->optical)-->
    pelvis frame (all downstream perception composes from here).

Camera extrinsics are NOT hand-measured: ``T_pelvis_camera`` is forward-kinematics to
the URDF ``d435_link`` frame (queried from the cuRobo planner, which lists it as an
FK-only tool frame), with an optional hand-eye ``extrinsic_correction`` and a
body->optical rotation from camera.yaml. The head camera is fixed to the locked torso,
so its pose is q-independent.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from g1_classical_manip.spatial.pose import Pose, from_xyz_rpy
from g1_classical_manip.motion.curobo_planner import CAMERA_FRAME

# Standard ROS camera body -> optical frame rotation (z forward, x right, y down).
BODY_TO_OPTICAL = Pose(np.array([[0.0, 0.0, 1.0],
                                 [-1.0, 0.0, 0.0],
                                 [0.0, -1.0, 0.0]]))


def _parse_source(source: Optional[str]) -> str:
    """'urdf:d435_link' -> 'd435_link'; bare names pass through; None -> CAMERA_FRAME."""
    if not source:
        return CAMERA_FRAME
    return source.split(":", 1)[1] if source.startswith("urdf:") else source


def _mount_pose(mount) -> Pose:
    """Constant parent_frame -> camera body. identity for the head."""
    if mount is None or (isinstance(mount, str) and mount == "identity"):
        return Pose.Identity()
    if isinstance(mount, dict):
        return from_xyz_rpy(mount.get("xyz", [0, 0, 0]), mount.get("rpy", [0, 0, 0]))
    return Pose.from_homogeneous(mount)                 # 4x4 row-major


class Frames:
    """Frame-math owner, cuRobo-backed. Resolves the head camera's pelvis-frame
    optical pose via the planner's ``fk_link`` and composes perception poses."""

    def __init__(self, planner, camera_cfg: Optional[dict] = None,
                 sim_base_world_pose: Optional[dict] = None):
        self.planner = planner
        ext = (camera_cfg or {}).get("extrinsics", {})
        self.parent_frame = _parse_source(ext.get("source"))
        self.body_to_optical = (BODY_TO_OPTICAL
                                if (ext.get("body_to_optical") or "ros") == "ros"
                                else Pose.Identity())
        self.mount = _mount_pose(ext.get("mount"))
        c = ext.get("extrinsic_correction")
        self.correction = Pose.from_homogeneous(c) if c is not None else Pose.Identity()
        # T_pelvis_world for the sim ground-truth path (identity unless configured).
        self._T_pelvis_world = self._resolve_sim_base(sim_base_world_pose)

    # ------------------------------------------------------------------- internals
    @staticmethod
    def _resolve_sim_base(base: Optional[dict]) -> Pose:
        """T_pelvis_world from a sim robot-base world pose {xyz, quat_wxyz}.
        Identity (no-op) when unset -> ignored on hardware."""
        if not base:
            return Pose.Identity()
        xyz = np.asarray(base.get("xyz", [0, 0, 0]), float)
        quat = base.get("quat_wxyz", [1, 0, 0, 0])
        return Pose.from_quaternion(quat, xyz).inverse()   # (T_world_pelvis)^-1

    # ------------------------------------------------------------------- API
    def T_pelvis_camera(self, q14=None) -> Pose:
        """Camera OPTICAL frame in pelvis:
        correction * FK(parent_frame) * mount * body_to_optical. q-independent for
        the head (fixed to the locked torso); pass q14 only matters for moving mounts."""
        return (self.correction * self.planner.fk_link(self.parent_frame, q14)
                * self.mount * self.body_to_optical)

    def T_pelvis_from_camera(self, T_cam_obj: Pose, q14=None) -> Pose:
        """An object pose observed in the camera OPTICAL frame -> pelvis frame."""
        return self.T_pelvis_camera(q14) * T_cam_obj

    def T_pelvis_from_world(self, T_world_obj: Pose) -> Pose:
        """Map a sim WORLD-frame object pose into the pelvis frame using the
        configured robot fixed-base world pose. Identity-passthrough when unset."""
        return self._T_pelvis_world * T_world_obj
