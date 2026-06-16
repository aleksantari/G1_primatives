"""Self-consistent tests for the numpy Pose util (no pinocchio; runs in the
cuRobo env). Convention fidelity vs pinocchio is cross-checked separately."""
import numpy as np

from g1_classical_manip.spatial.pose import (
    Pose, rpy_to_matrix, quat_wxyz_to_matrix, matrix_to_quat_wxyz, from_xyz_rpy)


def _rand_pose(rng):
    q = rng.normal(size=4)
    return Pose(quat_wxyz_to_matrix(q / np.linalg.norm(q)), rng.normal(size=3))


def test_identity():
    assert np.allclose(Pose.Identity().homogeneous, np.eye(4))


def test_inverse_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        T = _rand_pose(rng)
        assert np.allclose((T * T.inverse()).homogeneous, np.eye(4), atol=1e-12)
        assert np.allclose((T.inverse() * T).homogeneous, np.eye(4), atol=1e-12)


def test_associativity():
    rng = np.random.default_rng(2)
    A, B, C = _rand_pose(rng), _rand_pose(rng), _rand_pose(rng)
    assert np.allclose(((A * B) * C).homogeneous, (A * (B * C)).homogeneous, atol=1e-12)


def test_homogeneous_roundtrip():
    rng = np.random.default_rng(3)
    T = _rand_pose(rng)
    assert np.allclose(Pose.from_homogeneous(T.homogeneous).homogeneous, T.homogeneous)


def test_quat_roundtrip():
    rng = np.random.default_rng(4)
    for _ in range(50):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q  # canonical (w>=0)
        q2 = matrix_to_quat_wxyz(quat_wxyz_to_matrix(q))
        if q2[0] < 0:
            q2 = -q2
        assert np.allclose(q, q2, atol=1e-9)


def test_from_xyz_rpy():
    T = from_xyz_rpy([0.3, -0.1, 0.5], [0.0, 0.0, 0.0])
    assert np.allclose(T.translation, [0.3, -0.1, 0.5])
    assert np.allclose(T.rotation, np.eye(3))
    # rotation is orthonormal for arbitrary rpy
    T2 = from_xyz_rpy([0, 0, 0], [0.3, -0.7, 1.1])
    R = T2.rotation
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_translation_setter():
    T = Pose.Identity()
    T.translation = np.array([1.0, 2.0, 3.0])
    assert np.allclose(T.translation, [1, 2, 3])
    M = T.copy()
    M.translation = T.translation + np.array([0.05, 0, 0])
    assert np.allclose(M.translation, [1.05, 2, 3])
    assert np.allclose(T.translation, [1, 2, 3])  # copy is independent
