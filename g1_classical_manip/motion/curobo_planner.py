"""cuRobo arm planner — the repo's only motion planner.

Wraps a warm cuRobo MotionPlanner built from a G1 dex3 robot config (arms-only:
legs/waist/hands locked). cuRobo's trajectory optimizer emits a fully
time-parameterized, dynamically-feasible plan (position/velocity/acceleration at
a fixed dt), so we build the executor's JointTrajectory directly from it -- no
separate retiming step. Speed is governed by the cuRobo robot config's joint
velocity/accel limits. Runs in the cuRobo env (numpy2/py3.11/CUDA); no pinocchio.

  plan_to_pose(start_q14, side, goal_pose) -> JointTrajectory   (cuRobo plan_pose)
  plan_joint (start_q14, goal_q14)         -> JointTrajectory   (cuRobo plan_cspace)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from g1_classical_manip.motion.planner_base import JointTrajectory, DOF
from g1_classical_manip.spatial.pose import Pose
from g1_classical_manip.ee.hand_base import LEFT, RIGHT

# wrist-yaw tool frames (the MVP goal frame). 5cm L_ee/palm offset is a later refinement.
WRIST_FRAME = {LEFT: "left_wrist_yaw_link", RIGHT: "right_wrist_yaw_link"}
# Head-camera link. Listed in the cuRobo config's tool_frames ONLY so its pelvis-frame
# pose is FK-queryable (perception). It is fixed to the locked torso, so plan_to_pose
# pins it to its constant current FK pose -- it never constrains the arms.
CAMERA_FRAME = "d435_link"
# repo dual-arm joint order: left 7 then right 7 (G1_29 arm order)
REPO_ARM = [f"{s}_{j}_joint" for s in ("left", "right")
            for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                      "wrist_roll", "wrist_pitch", "wrist_yaw")]


class PlanningError(RuntimeError):
    pass


class CuroboArmPlanner:
    """One warm cuRobo MotionPlanner; plans single-arm wrist-to-pose and
    joint-space moves, returning executor-ready JointTrajectory objects."""

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
    def _joint_state(self, q_repo14):
        """repo-order (14,) config -> cuRobo JointState in active (cuRobo) order."""
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

    def _to_trajectory(self, result, label: str) -> JointTrajectory:
        """cuRobo TrajOptSolverResult -> JointTrajectory (14, repo order), using
        cuRobo's own interpolated timing (position/velocity/acceleration at dt)."""
        interp = result.get_interpolated_plan()
        names = list(interp.joint_names)
        nj = len(names)
        cols = [names.index(j) for j in REPO_ARM]            # cuRobo -> repo order

        def _col(x):
            return x.detach().cpu().numpy().reshape(-1, nj)[:, cols]

        q = _col(interp.position)
        qd = _col(interp.velocity) if interp.velocity is not None else np.zeros_like(q)
        qdd = _col(interp.acceleration) if interp.acceleration is not None else np.zeros_like(q)
        dt = float(interp.dt)
        t = np.arange(q.shape[0]) * dt
        return JointTrajectory(t, q, qd, qdd, meta={
            "source": "curobo", "label": label, "dt": dt,
            "duration": float(t[-1]) if t.size else 0.0,
            "max_qd": float(np.max(np.abs(qd))) if qd.size else 0.0,
        })

    def fk_link(self, link_name: str, q_repo14=None) -> Pose:
        """Pelvis-frame pose of any tool_frame link at the given dual-arm config
        (defaults to zeros/home). CAMERA_FRAME (head camera) is rigidly fixed to the
        locked torso, so its pose is q-invariant -- callers may omit q for it."""
        q = np.zeros(DOF) if q_repo14 is None else q_repo14
        ks = self._mp.compute_kinematics(self._joint_state(q))
        return self._link_pose(ks, link_name)

    def fk(self, side: str, q_repo14) -> Pose:
        """Wrist-yaw pose (pelvis frame) of `side` at the given dual-arm config."""
        return self.fk_link(WRIST_FRAME[side], q_repo14)

    def default_q(self) -> np.ndarray:
        """cuRobo's collision-free default ('ready') arm config, in repo order."""
        q_active = self._mp.default_joint_state.position.detach().cpu().numpy().reshape(-1)
        return q_active[[self.active.index(j) for j in REPO_ARM]]

    # --- planning ---
    def plan_joint(self, start_q_repo14, goal_q_repo14) -> JointTrajectory:
        """Collision-aware joint-space move to a configuration (cuRobo plan_cspace).
        Native cuRobo timing; both arms move together."""
        start = self._joint_state(start_q_repo14)
        goal = self._joint_state(goal_q_repo14)
        result = self._mp.plan_cspace(goal, start)
        if result is None or not bool(result.success.any()):
            raise PlanningError(
                f"cuRobo plan_cspace failed -> {np.round(np.asarray(goal_q_repo14, float), 3)}")
        return self._to_trajectory(result, "joint_goal")

    def plan_to_pose(self, start_q_repo14, side: str, goal_pose: Pose) -> JointTrajectory:
        """Move `side` wrist to goal_pose (pelvis frame); hold the other arm.
        Returns a JointTrajectory (14, repo order) with cuRobo's native timing."""
        from curobo.types import GoalToolPose, Pose as CuPose
        start = self._joint_state(start_q_repo14)
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
        return self._to_trajectory(result, f"{side}_goal")
