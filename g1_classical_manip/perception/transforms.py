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

from typing import Optional

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
                 camera_cfg: Optional[dict] = None):
        self.rr = reduced_robot or load_g1_reduced(
            urdf_path or _default_urdf(), locked_reference=locked_reference)
        self.model = self.rr.model
        self.data = self.rr.data
        self.camera_cfg = camera_cfg or {}
        self._body_to_optical = self._resolve_body_to_optical()
        self._correction = self._resolve_correction()

    # ------------------------------------------------------------- internals
    def _resolve_body_to_optical(self) -> pin.SE3:
        mode = (self.camera_cfg.get("extrinsics", {})
                .get("body_to_optical", "ros"))
        return BODY_TO_OPTICAL if mode == "ros" else pin.SE3.Identity()

    def _resolve_correction(self) -> pin.SE3:
        c = self.camera_cfg.get("extrinsics", {}).get("extrinsic_correction")
        return from_homogeneous(c) if c is not None else pin.SE3.Identity()

    def _frame_pose(self, name: str, q14=None) -> pin.SE3:
        q = np.zeros(self.model.nq) if q14 is None else np.asarray(q14, float)
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.model.getFrameId(name)].copy()

    # ------------------------------------------------------------------- API
    def T_pelvis_frame(self, name: str, q14=None) -> pin.SE3:
        return self._frame_pose(name, q14)

    def T_pelvis_camera(self, q14=None) -> pin.SE3:
        """Camera OPTICAL frame in pelvis. Independent of arm q (camera is fixed
        to torso, waist locked). correction * FK(d435_link) * body_to_optical."""
        T_pelvis_body = self._frame_pose(CAMERA_FRAME, q14)
        return self._correction * T_pelvis_body * self._body_to_optical

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
    def T_pelvis_tag(self, T_cam_tag: pin.SE3, q14=None) -> pin.SE3:
        """tag pose (in camera optical frame) -> pelvis frame."""
        return self.T_pelvis_camera(q14) * T_cam_tag

    def T_pelvis_block(self, T_cam_tag: pin.SE3, tag_to_block: pin.SE3,
                       q14=None) -> pin.SE3:
        return self.T_pelvis_tag(T_cam_tag, q14) * tag_to_block

    # ---- grasp pose composition ----
    def grasp_ee_target(self, T_pelvis_block: pin.SE3, side: str,
                        grasp_offset: pin.SE3, q14=None) -> pin.SE3:
        """Convert a desired palm grasp pose (block * grasp_offset) into the
        L_ee/R_ee target the IK consumes: T_ee = T_palm_desired * (T_ee_palm)^-1.
        """
        T_palm_desired = T_pelvis_block * grasp_offset
        return T_palm_desired * self.T_ee_palm(side).inverse()


def _default_urdf():
    from g1_classical_manip.robot_control.robot_arm_ik import DEFAULT_URDF
    return DEFAULT_URDF


def top_down_grasp_offset(approach_clearance: float = 0.0) -> pin.SE3:
    """Palm offset for a top-down grasp: palm above the block center by
    ``approach_clearance``, palm z pointing down onto the block."""
    R = pin.rpy.rpyToMatrix(np.pi, 0.0, 0.0)  # flip palm to face -z
    return pin.SE3(R, np.array([0.0, 0.0, approach_clearance]))
