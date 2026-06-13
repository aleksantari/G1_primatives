import numpy as np
import pinocchio as pin

from g1_classical_manip.perception.transforms import (
    Frames, from_xyz_rpy, se3, to_homogeneous, from_homogeneous, BODY_TO_OPTICAL)


def _frames(robot):
    return robot.frames


def test_camera_extrinsic_from_fk(robot):
    Tc = robot.frames.T_pelvis_camera()
    # head camera sits up and forward of the pelvis, optical z points forward/down
    assert Tc.translation[2] > 0.3
    assert Tc.rotation[0, 2] > 0.3      # +x component of optical-z (forward)
    assert Tc.rotation[2, 2] < 0.0      # -z component of optical-z (down)


def test_block_roundtrip(robot):
    F = robot.frames
    T_pelvis_block = from_xyz_rpy([0.35, 0.05, 0.05], [0, 0, 0.3])
    T_cam_block = F.T_pelvis_camera().inverse() * T_pelvis_block
    back = F.T_pelvis_block(T_cam_block, se3())
    assert np.linalg.norm(back.translation - T_pelvis_block.translation) < 1e-9


def test_grasp_ee_palm_consistency(robot):
    F = robot.frames
    block = from_xyz_rpy([0.34, 0.16, 0.06], [0, 0, 0])
    offset = se3()
    ee_target = F.grasp_ee_target(block, "left", offset)
    palm = ee_target * F.T_ee_palm("left")
    assert np.linalg.norm(palm.translation - (block * offset).translation) < 1e-9


def test_homogeneous_roundtrip():
    T = from_xyz_rpy([0.1, -0.2, 0.3], [0.1, 0.2, -0.3])
    assert np.allclose(to_homogeneous(from_homogeneous(to_homogeneous(T))), T.homogeneous)


def test_body_to_optical_is_rotation():
    R = BODY_TO_OPTICAL.rotation
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)
