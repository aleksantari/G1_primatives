"""Dex3 hand-local forward kinematics (wrist_yaw -> fingertips), URDF-derived.

This is NOT a planner and NOT a pelvis-frame frame builder (cuRobo + transforms.Frames
own those, and cuRobo locks the hand joints in the arms-only config so it can't FK the
fingers). It is a small, self-contained FK of the Dex3 finger chain off the wrist_yaw
link, used by exactly two consumers:

  * `scripts/derive_graspgenx_tool_transform.py` -- to DERIVE the grasp->wrist transform
    (where our thumb/finger contact lands, expressed in the wrist_yaw frame), and
  * `09_graspgen --source sim_cloud` -- to VERIFY, after a candidate is chosen, that our
    fingertips actually straddle the object the model picked (the check the viser mesh
    overlay cannot do, since that draws GraspGenX's gripper, not ours).

Everything is parsed live from the committed URDF (`assets/g1/g1_29dof_mode_16_dex3.urdf`)
so it stays correct if the hand is recalibrated. Positions are in the side's
``{side}_wrist_yaw_link`` frame (meters)."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List

import numpy as np

from g1_primitives.spatial.pose import Pose, rpy_to_matrix
from g1_primitives.ee.hand_base import LEFT, RIGHT

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_URDF = os.path.join(_REPO_ROOT, "assets", "g1", "g1_29dof_mode_16_dex3.urdf")

# The 7-vector hand-command order differs per side (middle/index swap between hands), per
# configs/hands.yaml: [thumb_0, thumb_1, thumb_2, <finger A 0,1>, <finger B 0,1>] where
# A,B = (index, middle) on the RIGHT and (middle, index) on the LEFT.
_Q7_JOINTS = {
    RIGHT: ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"],
    LEFT:  ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
}
# Distal pad reach beyond the last finger/thumb link ORIGIN, along that link's own distal
# axis. The direction is read from the link's inertial origin (the COM lies along the
# phalanx toward the tip), so it is correct per-side automatically -- the left thumb mirrors
# the right without a hand-coded sign. The magnitude ~the distal phalanx half-length is
# approximate, and the one soft number in the contact estimate (the sim FK check on the
# object is what confirms it).
_PAD_M = 0.035


def _joint_T(joint: ET.Element, angle: float) -> Pose:
    """Homogeneous transform across one URDF joint: fixed origin then rotation about its
    axis by `angle` (0 for fixed joints)."""
    o = joint.find("origin")
    xyz = [float(v) for v in (o.get("xyz", "0 0 0").split())] if o is not None else [0, 0, 0]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0").split())] if o is not None else [0, 0, 0]
    T = Pose(rpy_to_matrix(*rpy), xyz)
    ax = joint.find("axis")
    if joint.get("type") in ("revolute", "continuous") and ax is not None and angle:
        a = np.asarray([float(v) for v in ax.get("xyz").split()], float)
        a = a / (np.linalg.norm(a) or 1.0)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)   # Rodrigues
        T = T * Pose(R, [0, 0, 0])
    return T


class Dex3Kinematics:
    """Finger FK off the wrist_yaw link, parsed from the Dex3 URDF (both hands)."""

    def __init__(self, urdf_path: str = _DEFAULT_URDF):
        root = ET.parse(urdf_path).getroot()
        self._joints = {j.get("name"): j for j in root.findall("joint")}
        # child_link -> joint that produces it (to walk a chain back to wrist_yaw)
        self._by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        self._links = {l.get("name"): l for l in root.findall("link")}

    def _distal_dir(self, link_name: str) -> np.ndarray:
        """Unit direction the distal phalanx points, in the tip link's local frame -- taken
        from its inertial origin (COM lies along the phalanx toward the tip). Side-correct by
        construction (the left thumb's local axis mirrors the right's in the URDF)."""
        link = self._links[link_name]
        io = link.find("inertial/origin")
        v = np.asarray([float(x) for x in io.get("xyz").split()], float) if io is not None else None
        n = np.linalg.norm(v) if v is not None else 0.0
        return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    def _chain(self, side: str, tip_link: str) -> List[ET.Element]:
        """Ordered joints from `{side}_wrist_yaw_link` down to `tip_link`."""
        root = f"{side}_wrist_yaw_link"
        chain: List[ET.Element] = []
        link = tip_link
        while link != root:
            j = self._by_child.get(link)
            if j is None:
                raise KeyError(f"no joint produces {link!r} (chain to {root!r} broken)")
            chain.append(j)
            link = j.find("parent").get("link")
        return list(reversed(chain))

    def _tip(self, side: str, finger: str, angles: Dict[str, float], pad_m: float) -> np.ndarray:
        """Position of one fingertip (last link origin + distal pad) in the wrist_yaw frame.
        `finger` in {thumb, index, middle}; `angles` keyed by short joint name (e.g. 'thumb_1')."""
        last = "thumb_2" if finger == "thumb" else f"{finger}_1"
        tip_link = f"{side}_hand_{last}_link"
        T = Pose.Identity()
        for j in self._chain(side, tip_link):
            short = j.get("name").replace(f"{side}_hand_", "").replace("_joint", "")
            T = T * _joint_T(j, float(angles.get(short, 0.0)))
        local = pad_m * self._distal_dir(tip_link)         # distal pad along the phalanx
        return (T * Pose(np.eye(3), local)).translation

    def fingertips(self, side: str, q7, pad_m: float = _PAD_M) -> Dict[str, np.ndarray]:
        """{thumb, index, middle: (3,) wrist_yaw-frame position} at hand config `q7`
        (7-vector in the configs/hands.yaml per-side order)."""
        q7 = np.asarray(q7, float).reshape(7)
        angles = {name: q7[i] for i, name in enumerate(_Q7_JOINTS[side])}
        return {f: self._tip(side, f, angles, pad_m) for f in ("thumb", "index", "middle")}

    def contact_point(self, side: str, q7, pad_m: float = _PAD_M) -> np.ndarray:
        """Grasp contact midpoint in the wrist_yaw frame: halfway between the thumb tip and
        the index/middle finger-pair midpoint (the two opposing 'jaws' of the Dex3)."""
        tips = self.fingertips(side, q7, pad_m)
        fingers_mid = 0.5 * (tips["index"] + tips["middle"])
        return 0.5 * (tips["thumb"] + fingers_mid)
