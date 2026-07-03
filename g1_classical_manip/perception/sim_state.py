"""Live ground-truth detector from the Isaac sim's `rt/sim_state` topic.

The sim publishes a `String_` JSON every step:
    {"init_state": "<json>", "task_name": ..., "_timestamp": ...}
where the inner JSON holds `rigid_object.<key>.root_pose = [[x,y,z, qw,qx,qy,qz]]`
in the sim WORLD frame (Isaac wxyz quat). We read the manipuland's live world pose
and map it to the pelvis frame via `transforms.Frames` (same world->pelvis the static
static-config detectors used). Unlike a hardcoded block pose, this tracks
the block as it settles / is nudged -- the marker-free oracle for sim.

Requires DDS to be initialised by the caller (ChannelFactoryInitialize), i.e. used
with make_robot(connect_dds=True) or a script that inits DDS itself. The subscriber is
created lazily on first use.
"""
from __future__ import annotations

import json
from typing import Optional, Dict

from g1_classical_manip.spatial.pose import Pose
from g1_classical_manip.perception.transforms import Frames
from g1_classical_manip.perception.base import Detector, Detection


class SimStateDetector(Detector):
    def __init__(self, frames: Frames, object_key: str = "object",
                 label: str = "block", topic: str = "rt/sim_state"):
        self.frames = frames
        self.object_key = object_key
        self.label = label
        self.topic = topic
        self._sub = None
        self._last = None          # last good pelvis-frame Pose (Read() may return None)

    @classmethod
    def from_config(cls, frames: Frames, cfg: dict) -> "SimStateDetector":
        cfg = cfg or {}
        return cls(frames, object_key=cfg.get("object_key", "object"),
                   label=cfg.get("label", "block"),
                   topic=cfg.get("topic", "rt/sim_state"))

    def _ensure_sub(self):
        if self._sub is None:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
            self._sub = ChannelSubscriber(self.topic, String_)
            self._sub.Init()
        return self._sub

    def _world_pose(self) -> Optional[Pose]:
        msg = self._ensure_sub().Read()
        if msg is None or not msg.data:
            return None
        d = json.loads(msg.data)
        inner = d.get("init_state", d)
        if isinstance(inner, str):
            inner = json.loads(inner)
        obj = (inner or {}).get("rigid_object", {}).get(self.object_key)
        if not obj or "root_pose" not in obj:
            return None
        rp = obj["root_pose"]
        if rp and isinstance(rp[0], list):     # [[x,y,z,qw,qx,qy,qz]] -> [...]
            rp = rp[0]
        pos, quat_wxyz = rp[:3], rp[3:7]
        return Pose.from_quaternion(quat_wxyz, pos)

    # ----- Detector interface -----
    def block_pose(self, label: str = "block") -> Optional[Pose]:
        if label != self.label:
            return None
        w = self._world_pose()
        if w is not None:
            self._last = self.frames.T_pelvis_from_world(w)
        return self._last

    def detect(self, gray=None, q14=None) -> Dict[str, Detection]:
        p = self.block_pose(self.label)
        return {self.label: Detection(pose=p, label=self.label, score=1.0)} if p else {}
