"""Ground-truth block source for the simulator (no perception).

Returns a constant pelvis-frame block pose computed from a configured WORLD-frame
block pose (mirroring the sim scene) and the robot's fixed-base world pose --
``transforms.Frames`` owns the world->pelvis conversion. Exposes ``block_pose``
with the same signature the pick-place FSM expects, so it drops into
``robot.perception`` unchanged (no FSM/hub edit).

Later: swap the constant world pose for a live ``rt/sim_state`` DDS read; the
world->pelvis transform is reused as-is.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pinocchio as pin

from g1_classical_manip.perception.transforms import Frames


class GroundTruthBlockSource:
    def __init__(self, frames: Frames, block_world_xyz,
                 block_world_quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
        w, x, y, z = block_world_quat_wxyz
        T_world_block = pin.SE3(pin.Quaternion(w, x, y, z).toRotationMatrix(),
                                np.asarray(block_world_xyz, float))
        self._T_pelvis_block = frames.T_pelvis_from_world(T_world_block)

    @classmethod
    def from_config(cls, frames: Frames, cfg: dict) -> "GroundTruthBlockSource":
        """cfg: {xyz: [...], quat_wxyz: [...]} -- the block's sim WORLD pose."""
        return cls(frames, cfg["xyz"], cfg.get("quat_wxyz", (1.0, 0.0, 0.0, 0.0)))

    def block_pose(self, tag_id: int = 0) -> Optional[pin.SE3]:
        return self._T_pelvis_block
