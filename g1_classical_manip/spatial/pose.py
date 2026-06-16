"""SE(3) pose type (numpy-only) — the repo's pose currency, replacing pin.SE3.

Conventions match pinocchio exactly so the migration off pinocchio preserves
behavior:
  * rpy_to_matrix(r,p,y) == pin.rpy.rpyToMatrix(r,p,y) == Rz(y) @ Ry(p) @ Rx(r)
  * quaternions are (w, x, y, z), matching pin.Quaternion(w, x, y, z)

A Pose wraps a 4x4 homogeneous matrix. `.rotation` / `.translation` are
assignable (the codebase mutates them, e.g. `M = T.copy(); M.translation = ...`).
Composition is `A * B`; `A.inverse()`, `.homogeneous` as for pin.SE3.
"""
from __future__ import annotations

import numpy as np


def rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    """Roll-pitch-yaw -> rotation matrix, identical to pin.rpy.rpyToMatrix."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_wxyz_to_matrix(quat) -> np.ndarray:
    """(w,x,y,z) quaternion -> rotation matrix (matches pin.Quaternion(w,x,y,z))."""
    w, x, y, z = np.asarray(quat, float)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])


def matrix_to_quat_wxyz(R) -> np.ndarray:
    """Rotation matrix -> (w,x,y,z) quaternion."""
    R = np.asarray(R, float).reshape(3, 3)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


class Pose:
    """A rigid transform (4x4 homogeneous). Drop-in for pin.SE3's used surface."""
    __slots__ = ("_m",)

    def __init__(self, rotation=None, translation=None, *, homogeneous=None):
        if homogeneous is not None:
            self._m = np.asarray(homogeneous, float).reshape(4, 4).copy()
            return
        m = np.eye(4)
        if rotation is not None:
            m[:3, :3] = np.asarray(rotation, float).reshape(3, 3)
        if translation is not None:
            m[:3, 3] = np.asarray(translation, float).reshape(3)
        self._m = m

    # --- constructors ---
    @classmethod
    def Identity(cls) -> "Pose":
        return cls()

    @classmethod
    def from_homogeneous(cls, M) -> "Pose":
        return cls(homogeneous=M)

    @classmethod
    def from_quaternion(cls, quat_wxyz, translation=None) -> "Pose":
        return cls(quat_wxyz_to_matrix(quat_wxyz), translation)

    @classmethod
    def from_xyz_rpy(cls, xyz, rpy) -> "Pose":
        r, p, y = np.asarray(rpy, float)
        return cls(rpy_to_matrix(r, p, y), np.asarray(xyz, float))

    # --- accessors (assignable, like pin.SE3) ---
    @property
    def rotation(self) -> np.ndarray:
        return self._m[:3, :3]

    @rotation.setter
    def rotation(self, R):
        self._m[:3, :3] = np.asarray(R, float).reshape(3, 3)

    @property
    def translation(self) -> np.ndarray:
        return self._m[:3, 3]

    @translation.setter
    def translation(self, t):
        self._m[:3, 3] = np.asarray(t, float).reshape(3)

    @property
    def homogeneous(self) -> np.ndarray:
        return self._m.copy()

    def quaternion_wxyz(self) -> np.ndarray:
        return matrix_to_quat_wxyz(self._m[:3, :3])

    # --- ops ---
    def inverse(self) -> "Pose":
        Rt = self._m[:3, :3].T
        m = np.eye(4)
        m[:3, :3] = Rt
        m[:3, 3] = -Rt @ self._m[:3, 3]
        return Pose(homogeneous=m)

    def __mul__(self, other: "Pose") -> "Pose":
        if isinstance(other, Pose):
            return Pose(homogeneous=self._m @ other._m)
        return NotImplemented

    def copy(self) -> "Pose":
        return Pose(homogeneous=self._m)

    def __repr__(self) -> str:
        return f"Pose(t={np.round(self.translation, 4)})"


def from_xyz_rpy(xyz, rpy) -> Pose:
    """Module-level helper mirroring the old transforms.from_xyz_rpy."""
    return Pose.from_xyz_rpy(xyz, rpy)
