"""cuRobo arm planner — the repo's only motion planner.

Wraps a warm cuRobo MotionPlanner built from a G1 dex3 robot config (arms-only:
legs/waist/hands locked). `plan_to_pose` moves ONE arm's wrist to a goal pose
(idle arm held at its current FK pose), returns geometry that the Ruckig retimer
times into a JointTrajectory the executor streams. Runs in the cuRobo env
(numpy2/py3.11/CUDA); no pinocchio.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from g1_classical_manip.motion.planner_base import JointPath, JointTrajectory, DOF
from g1_classical_manip.motion.retimer import retime_from_config
from g1_classical_manip.spatial.pose import Pose
from g1_classical_manip.ee.hand_base import LEFT, RIGHT

# wrist-yaw tool frames (the MVP goal frame). 5cm L_ee/palm offset is a later refinement.
WRIST_FRAME = {LEFT: "left_wrist_yaw_link", RIGHT: "right_wrist_yaw_link"}
# repo dual-arm joint order: left 7 then right 7 (G1_29 arm order)
REPO_ARM = [f"{s}_{j}_joint" for s in ("left", "right")
            for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                      "wrist_roll", "wrist_pitch", "wrist_yaw")]


class PlanningError(RuntimeError):
    pass


class CuroboArmPlanner:
    """One warm cuRobo MotionPlanner; plans single-arm wrist-to-pose moves."""

    def __init__(self, robot_cfg_path: str, planner_cfg: Optional[dict] = None):
        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        self._torch = torch
        self.planner_cfg = planner_cfg or {}
        self._mp = MotionPlanner(MotionPlannerCfg.create(robot=robot_cfg_path))
        self._mp.warmup(enable_graph=True, num_warmup_iterations=5)
        self.active = list(self._mp.joint_names)          # 14 arm joints, cuRobo order
        self.tool_frames = list(self._mp.tool_frames)
        assert len(self.active) == DOF, f"expected {DOF} active joints, got {len(self.active)}"
        assert set(self.active) == set(REPO_ARM), "active joints != repo arm set"

    # --- helpers ---
    def _start_state(self, q_repo14):
        from curobo.types import JointState
        q = np.asarray(q_repo14, float).reshape(DOF)
        q_active = q[[REPO_ARM.index(j) for j in self.active]]   # repo -> cuRobo order
        t = self._torch.tensor(q_active, dtype=self._torch.float32, device="cuda").unsqueeze(0)
        return JointState.from_position(t, joint_names=self.active)

    def _tensor(self, arr):
        return self._torch.tensor(np.asarray(arr, float), dtype=self._torch.float32,
                                  device="cuda").unsqueeze(0)

    def _link_pose(self, kin_state, frame) -> Pose:
        lp = kin_state.tool_poses.get_link_pose(frame)
        pos = lp.position[0].detach().cpu().numpy()
        quat = lp.quaternion[0].detach().cpu().numpy()   # wxyz
        return Pose.from_quaternion(quat, pos)

    def fk(self, side: str, q_repo14) -> Pose:
        """Wrist-yaw pose (pelvis frame) of `side` at the given dual-arm config."""
        ks = self._mp.compute_kinematics(self._start_state(q_repo14))
        return self._link_pose(ks, WRIST_FRAME[side])

    def default_q(self) -> np.ndarray:
        """cuRobo's collision-free default ('ready') arm config, in repo order."""
        q_active = self._mp.default_joint_state.position.detach().cpu().numpy().reshape(-1)
        return q_active[[self.active.index(j) for j in REPO_ARM]]

    def plan_joint(self, start_q_repo14, goal_q_repo14) -> JointTrajectory:
        """Direct joint-space move (Ruckig-retimed). No collision check — use only
        for known-safe moves (e.g. un-tucking from the at-rest pose to ready)."""
        q = np.vstack([np.asarray(start_q_repo14, float).reshape(DOF),
                       np.asarray(goal_q_repo14, float).reshape(DOF)])
        path = JointPath(q, meta={"waypoint_index": [(1, "joint_goal")]})
        return retime_from_config(path, self.planner_cfg)

    # --- planning ---
    def plan_to_pose(self, start_q_repo14, side: str, goal_pose: Pose) -> JointTrajectory:
        """Move `side` wrist to goal_pose (pelvis frame); hold the other arm.
        Returns a Ruckig-retimed JointTrajectory (14, repo order)."""
        from curobo.types import GoalToolPose, Pose as CuPose
        start = self._start_state(start_q_repo14)
        ks = self._mp.compute_kinematics(start)

        pose_dict = {}
        for f in self.tool_frames:
            if f == WRIST_FRAME[side]:
                pos, quat = goal_pose.translation, goal_pose.quaternion_wxyz()
            else:  # hold idle arm at its current FK pose
                cur = self._link_pose(ks, f)
                pos, quat = cur.translation, cur.quaternion_wxyz()
            pose_dict[f] = CuPose(position=self._tensor(pos), quaternion=self._tensor(quat))
        goal = GoalToolPose.from_poses(pose_dict, ordered_tool_frames=self.tool_frames, num_goalset=1)

        result = self._mp.plan_pose(goal, start)
        if result is None or not bool(result.success.any()):
            raise PlanningError(
                f"cuRobo plan_pose failed: {side} -> {np.round(goal_pose.translation, 3)}")

        interp = result.get_interpolated_plan()
        names = list(interp.joint_names)
        nj = interp.position.shape[-1]
        cols = [names.index(j) for j in REPO_ARM]                 # cuRobo -> repo order
        q = interp.position.detach().cpu().numpy().reshape(-1, nj)[:, cols]   # (N, 14)
        path = JointPath(q, meta={"waypoint_index": [(q.shape[0] - 1, f"{side}_goal")]})
        return retime_from_config(path, self.planner_cfg)
