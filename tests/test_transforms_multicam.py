import numpy as np


def test_head_default_matches_named(robot):
    a = robot.frames.T_pelvis_camera()
    b = robot.frames.T_pelvis_camera(cam_name="head")
    assert np.allclose(a.homogeneous, b.homogeneous)


def test_head_is_q_independent(robot):
    q0 = np.zeros(14)
    q1 = q0.copy(); q1[0] = 0.5
    T0 = robot.frames.T_pelvis_camera(q0, "head")
    T1 = robot.frames.T_pelvis_camera(q1, "head")
    assert np.allclose(T0.homogeneous, T1.homogeneous)


def test_wrist_camera_moves_with_arm(robot):
    # left_wrist parent_frame is left_wrist_yaw_link -> FK depends on left arm q
    q0 = np.zeros(14)
    q1 = q0.copy(); q1[0] = 0.5
    T0 = robot.frames.T_pelvis_camera(q0, "left_wrist")
    T1 = robot.frames.T_pelvis_camera(q1, "left_wrist")
    assert not np.allclose(T0.translation, T1.translation)


def test_block_pose_accepts_cam_name(robot):
    # cam_name threads through T_pelvis_block without error; head == default
    from g1_classical_manip.perception.transforms import se3, from_xyz_rpy
    T_pelvis_block = from_xyz_rpy([0.35, 0.05, 0.05], [0, 0, 0.3])
    T_cam_block = robot.frames.T_pelvis_camera().inverse() * T_pelvis_block
    back = robot.frames.T_pelvis_block(T_cam_block, se3(), cam_name="head")
    assert np.linalg.norm(back.translation - T_pelvis_block.translation) < 1e-9
