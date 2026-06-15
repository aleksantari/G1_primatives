"""The single owner of frame math (CLAUDE.md rule 2). Everything is pin.SE3 in
the pelvis frame.

Frame chain for perception:
    tag (optical-frame pose from AprilTag) --T_cam_tag-->
    camera optical frame --T_pelvis_cam (FK to d435_link + body->optical)-->
    pelvis --tag_to_block--> block center --grasp offset--> palm pose
    --T_palm_ee--> L_ee/R_ee IK target.

Camera extrinsics are NOT hand-measured: T_pelvis_camera is forward-kinematics to
the URDF ``d435_link`` frame (retained in the reduced model), with an optional
hand-eye ``extrinsic_correction`` and a body->optical rotation from camera.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pinocchio as pin

from g1_classical_manip.robot_control.robot_arm_ik import load_g1_reduced

# Standard ROS camera body -> optical frame rotation (z forward, x right, y down).
BODY_TO_OPTICAL = pin.SE3(
    np.array([[0.0, 0.0, 1.0],
              [-1.0, 0.0, 0.0],
              [0.0, -1.0, 0.0]]), np.zeros(3))

CAMERA_FRAME = "d435_link"
PALM_FRAME = {"left": "left_hand_palm_link", "right": "right_hand_palm_link"}
EE_FRAME = {"left": "L_ee", "right": "R_ee"}


@dataclass
class _CamExt:
    """Resolved per-camera extrinsics. ``T_pelvis_optical`` =
    correction * FK(parent_frame) * mount * body_to_optical."""
    parent_frame: str
    mount: pin.SE3            # constant parent_frame -> camera body (Identity for head)
    body_to_optical: pin.SE3
    correction: pin.SE3       # pelvis-frame hand-eye refinement (left-multiplied)


# --------------------------------------------------------------- SE3 helpers
def se3(R=None, t=None) -> pin.SE3:
    R = np.eye(3) if R is None else np.asarray(R, float)
    t = np.zeros(3) if t is None else np.asarray(t, float)
    return pin.SE3(R, t)


def from_xyz_rpy(xyz, rpy) -> pin.SE3:
    return pin.SE3(pin.rpy.rpyToMatrix(*np.asarray(rpy, float)),
                   np.asarray(xyz, float))


def from_homogeneous(M) -> pin.SE3:
    M = np.asarray(M, float)
    return pin.SE3(M[:3, :3].copy(), M[:3, 3].copy())


def to_homogeneous(T: pin.SE3) -> np.ndarray:
    return T.homogeneous


class Frames:
    """Frame-math owner. Holds the reduced G1 model (with d435/palm/EE frames)
    and resolves all pelvis-frame transforms."""

    def __init__(self, reduced_robot=None, urdf_path=None, locked_reference=None,
                 camera_cfg: Optional[dict] = None,
                 cameras_cfg: Optional[dict] = None,
                 sim_base_world_pose: Optional[dict] = None):
        self.rr = reduced_robot or load_g1_reduced(
            urdf_path or _default_urdf(), locked_reference=locked_reference)
        self.model = self.rr.model
        self.data = self.rr.data
        self.camera_cfg = camera_cfg or {}
        # Per-camera extrinsics keyed by name. Multi-camera config (cameras.yaml)
        # is preferred; falls back to the legacy single-camera camera.yaml layout
        # so a "head" camera always resolves.
        self._cameras: Dict[str, _CamExt] = self._resolve_cameras(cameras_cfg)
        head = self._cameras["head"]
        self._body_to_optical = head.body_to_optical   # legacy single-cam accessors
        self._correction = head.correction
        # T_pelvis_world for the sim ground-truth path (robot fixed-base world pose).
        # Identity unless a sim base pose is configured -> no effect on hardware.
        self._T_pelvis_world = self._resolve_sim_base(sim_base_world_pose)

    # ------------------------------------------------------------- internals
    def _resolve_cameras(self, cameras_cfg) -> Dict[str, _CamExt]:
        out: Dict[str, _CamExt] = {}
        if cameras_cfg:
            cams = cameras_cfg.get("cameras", cameras_cfg)
            for name, spec in cams.items():
                out[name] = self._parse_cam_spec(spec or {})
        if "head" not in out:
            out["head"] = self._legacy_head(self.camera_cfg)
        return out

    @staticmethod
    def _body_to_optical_se3(mode) -> pin.SE3:
        return BODY_TO_OPTICAL if (mode or "ros") == "ros" else pin.SE3.Identity()

    @staticmethod
    def _mount_se3(mount) -> pin.SE3:
        if mount is None or (isinstance(mount, str) and mount == "identity"):
            return pin.SE3.Identity()
        if isinstance(mount, dict):
            return from_xyz_rpy(mount.get("xyz", [0, 0, 0]),
                                mount.get("rpy", [0, 0, 0]))
        return from_homogeneous(mount)                 # 4x4 row-major

    def _parse_cam_spec(self, spec: dict) -> _CamExt:
        c = spec.get("extrinsic_correction")
        return _CamExt(
            parent_frame=spec.get("parent_frame", CAMERA_FRAME),
            mount=self._mount_se3(spec.get("mount")),
            body_to_optical=self._body_to_optical_se3(spec.get("body_to_optical")),
            correction=from_homogeneous(c) if c is not None else pin.SE3.Identity())

    def _legacy_head(self, camera_cfg: dict) -> _CamExt:
        ext = (camera_cfg or {}).get("extrinsics", {})
        c = ext.get("extrinsic_correction")
        return _CamExt(
            parent_frame=CAMERA_FRAME, mount=pin.SE3.Identity(),
            body_to_optical=self._body_to_optical_se3(ext.get("body_to_optical")),
            correction=from_homogeneous(c) if c is not None else pin.SE3.Identity())

    def _resolve_sim_base(self, base: Optional[dict]) -> pin.SE3:
        """Return T_pelvis_world from a sim robot-base world pose
        {xyz, quat_wxyz}. Identity (no-op) when unset."""
        if not base:
            return pin.SE3.Identity()
        xyz = np.asarray(base.get("xyz", [0, 0, 0]), float)
        w, x, y, z = base.get("quat_wxyz", [1, 0, 0, 0])
        T_world_pelvis = pin.SE3(pin.Quaternion(w, x, y, z).toRotationMatrix(), xyz)
        return T_world_pelvis.inverse()

    def _frame_pose(self, name: str, q14=None) -> pin.SE3:
        q = np.zeros(self.model.nq) if q14 is None else np.asarray(q14, float)
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.model.getFrameId(name)].copy()

    # ------------------------------------------------------------------- API
    def T_pelvis_frame(self, name: str, q14=None) -> pin.SE3:
        return self._frame_pose(name, q14)

    def T_pelvis_camera(self, q14=None, cam_name: str = "head") -> pin.SE3:
        """Camera OPTICAL frame in pelvis:
        correction * FK(parent_frame) * mount * body_to_optical.
        The head is fixed to torso (waist locked) so it is q-independent; wrist
        cameras move with the arm, so pass a live q14 for those."""
        cam = self._cameras[cam_name]
        T_pelvis_parent = self._frame_pose(cam.parent_frame, q14)
        return cam.correction * T_pelvis_parent * cam.mount * cam.body_to_optical

    def T_palm(self, side: str, q14) -> pin.SE3:
        return self._frame_pose(PALM_FRAME[side], q14)

    def T_ee(self, side: str, q14) -> pin.SE3:
        return self._frame_pose(EE_FRAME[side], q14)

    def T_ee_palm(self, side: str) -> pin.SE3:
        """Constant transform EE -> palm (both fixed on the wrist body)."""
        q0 = np.zeros(self.model.nq)
        T_ee = self._frame_pose(EE_FRAME[side], q0)
        T_palm = self._frame_pose(PALM_FRAME[side], q0)
        return T_ee.inverse() * T_palm

    # ---- perception composition ----
    def T_pelvis_tag(self, T_cam_tag: pin.SE3, q14=None,
                     cam_name: str = "head") -> pin.SE3:
        """tag pose (in camera optical frame) -> pelvis frame."""
        return self.T_pelvis_camera(q14, cam_name) * T_cam_tag

    def T_pelvis_block(self, T_cam_tag: pin.SE3, tag_to_block: pin.SE3,
                       q14=None, cam_name: str = "head") -> pin.SE3:
        return self.T_pelvis_tag(T_cam_tag, q14, cam_name) * tag_to_block

    # ---- sim ground-truth (world frame -> pelvis) ----
    def T_pelvis_from_world(self, T_world_obj: pin.SE3) -> pin.SE3:
        """Map a sim WORLD-frame object pose into the pelvis frame using the
        configured robot fixed-base world pose. Identity-passthrough when no sim
        base pose was set."""
        return self._T_pelvis_world * T_world_obj

    # ---- grasp pose composition ----
    def grasp_ee_target(self, T_pelvis_block: pin.SE3, side: str,
                        grasp_offset: pin.SE3, q14=None) -> pin.SE3:
        """Convert a desired palm grasp pose (block * grasp_offset) into the
        L_ee/R_ee target the IK consumes: T_ee = T_palm_desired * (T_ee_palm)^-1.
        """
        T_palm_desired = T_pelvis_block * grasp_offset
        return T_palm_desired * self.T_ee_palm(side).inverse()

    def grasp_ee_target_aligned(self, T_pelvis_block: pin.SE3, side: str,
                                palm_rotation, clearance: float) -> pin.SE3:
        """EE target placing the palm `clearance` above the block CENTER (pelvis
        +z) with an explicit palm orientation, DECOUPLED from the block's own
        orientation. Avoids the near-singular top-down wrist config -- pass a
        well-conditioned palm orientation (e.g. the arm's home EE rotation)."""
        T_palm = pin.SE3(np.asarray(palm_rotation, float),
                         T_pelvis_block.translation + np.array([0.0, 0.0, clearance]))
        return T_palm * self.T_ee_palm(side).inverse()


def _default_urdf():
    from g1_classical_manip.robot_control.robot_arm_ik import DEFAULT_URDF
    return DEFAULT_URDF


def top_down_grasp_offset(approach_clearance: float = 0.0) -> pin.SE3:
    """Palm offset for a top-down grasp: palm above the block center by
    ``approach_clearance``, palm z pointing down onto the block."""
    R = pin.rpy.rpyToMatrix(np.pi, 0.0, 0.0)  # flip palm to face -z
    return pin.SE3(R, np.array([0.0, 0.0, approach_clearance]))
