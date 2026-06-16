"""Ground-truth block detector for the simulator (no camera / no marker).

Returns a constant pelvis-frame block pose from a configured WORLD-frame block pose
(mirroring the sim scene) and the robot's fixed-base world pose -- ``transforms.Frames``
owns the world->pelvis conversion. Implements the same ``Detector`` seam as
``AprilTagDetector``, so ``detect()`` (and a later ``detect -> move`` pick) runs in the
current sim before AprilTags are added to the scene.

Later: swap the constant world pose for a live ``rt/sim_state`` DDS read; the
world->pelvis transform is reused as-is.
"""
from __future__ import annotations

from typing import Optional, Dict

import numpy as np

from g1_classical_manip.spatial.pose import Pose
from g1_classical_manip.perception.transforms import Frames
from g1_classical_manip.perception.base import Detector, Detection


class GroundTruthDetector(Detector):
    def __init__(self, frames: Frames, block_world_xyz,
                 block_world_quat_wxyz=(1.0, 0.0, 0.0, 0.0), label: str = "block"):
        T_world_block = Pose.from_quaternion(block_world_quat_wxyz,
                                             np.asarray(block_world_xyz, float))
        self._pose = frames.T_pelvis_from_world(T_world_block)
        self._label = label

    @classmethod
    def from_config(cls, frames: Frames, cfg: dict,
                    label: str = "block") -> "GroundTruthDetector":
        """cfg: {xyz: [...], quat_wxyz: [...]} -- the block's sim WORLD pose."""
        return cls(frames, cfg["xyz"], cfg.get("quat_wxyz", (1.0, 0.0, 0.0, 0.0)),
                   label=label)

    def detect(self, gray=None, q14=None) -> Dict[str, Detection]:
        return {self._label: Detection(pose=self._pose, label=self._label, score=1.0)}

    def block_pose(self, label: str = "block") -> Optional[Pose]:
        return self._pose if label == self._label else None
