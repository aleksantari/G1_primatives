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

from dataclasses import dataclass
from typing import List, Optional

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
# cuRobo base_link (kinematic root). cuRobo only exposes FK for tool_frames, not the
# base, so fk_link short-circuits this to identity (the base relative to itself) -- lets
# perception use a base-frame extrinsic (source: urdf:pelvis) from a hand-eye calibration.
BASE_FRAME = "pelvis"
# repo dual-arm joint order: left 7 then right 7 (G1_29 arm order)
REPO_ARM = [f"{s}_{j}_joint" for s in ("left", "right")
            for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                      "wrist_roll", "wrist_pitch", "wrist_yaw")]


class PlanningError(RuntimeError):
    pass


@dataclass
class GraspPlanOutcome:
    """Result of a native cuRobo plan_grasp goalset solve: which candidate cuRobo selected
    (chosen_index, into the candidate list as fed) and the approach/grasp/lift segments as
    executor-ready JointTrajectory objects (any may be None if that phase was not planned)."""
    success: bool
    chosen_index: int
    approach: Optional[JointTrajectory]
    grasp: Optional[JointTrajectory]
    lift: Optional[JointTrajectory]
    approach_success: bool
    grasp_success: bool
    lift_success: bool
    status: str


class CuroboArmPlanner:
    """One warm cuRobo MotionPlanner; plans single-arm wrist-to-pose and
    joint-space moves, returning executor-ready JointTrajectory objects."""

    def __init__(self, robot_cfg_path: str, planner_cfg: Optional[dict] = None):
        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        self._torch = torch
        self.planner_cfg = planner_cfg or {}
        self._robot_cfg_path = robot_cfg_path
        # max goalset size for native plan_grasp (>= graspgenx.topk); warmup auto-warms the
        # goalset path at this size. Default 1 in cuRobo would forbid multi-candidate solves.
        self._max_goalset = int((self.planner_cfg.get("grasp") or {}).get("max_goalset", 128))
        self._mp = MotionPlanner(MotionPlannerCfg.create(
            robot=robot_cfg_path, max_goalset=self._max_goalset))
        self._mp.warmup(enable_graph=True, num_warmup_iterations=5)
        self._grasp_mp = {}       # lazy per-side single-tool-frame planners for plan_grasp
        self.active = list(self._mp.joint_names)          # 14 arm joints, cuRobo order
        self.tool_frames = list(self._mp.tool_frames)
        assert len(self.active) == DOF, f"expected {DOF} active joints, got {len(self.active)}"
        assert set(self.active) == set(REPO_ARM), "active joints != repo arm set"
        self._dyn = None          # lazy cuRobo RNEA Dynamics (gravity-comp; built on 1st use)

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

    def _tensor_k(self, arr):
        """Like _tensor but WITHOUT the leading unsqueeze: the batch dim IS the K goalset rows
        (a (K, 3)/(K, 4) stack of goal positions/quaternions for GoalToolPose.from_poses)."""
        return self._torch.tensor(np.asarray(arr, float), dtype=self._torch.float32,
                                  device="cuda")

    def _link_pose(self, kin_state, frame) -> Pose:
        lp = kin_state.tool_poses.get_link_pose(frame)
        pos = lp.position[0].detach().cpu().numpy()
        quat = lp.quaternion[0].detach().cpu().numpy()   # wxyz
        return Pose.from_quaternion(quat, pos)

    def _jointstate_to_trajectory(self, js, last_tstep, label: str,
                                  dt: Optional[float] = None) -> JointTrajectory:
        """A cuRobo interpolated JointState (position/velocity/acceleration over a horizon,
        with .dt) -> JointTrajectory (14, repo order). `last_tstep` (a length-1 tensor / int,
        or None) trims to the valid horizon length exactly as cuRobo's get_interpolated_plan
        does (trim_joint_state_trajectory(js, 0, last_tstep[0])); `dt` overrides js.dt when a
        segment carries its dt separately. Shared by plan_pose/plan_cspace and plan_grasp."""
        if last_tstep is not None:
            from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
            end = last_tstep[0] if hasattr(last_tstep, "__getitem__") else last_tstep
            js = trim_joint_state_trajectory(js, 0, end)
        names = list(js.joint_names)
        nj = len(names)
        cols = [names.index(j) for j in REPO_ARM]            # cuRobo -> repo order

        def _col(x):
            return x.detach().cpu().numpy().reshape(-1, nj)[:, cols]

        q = _col(js.position)
        qd = _col(js.velocity) if js.velocity is not None else np.zeros_like(q)
        qdd = _col(js.acceleration) if js.acceleration is not None else np.zeros_like(q)
        d = float(dt if dt is not None else js.dt)
        t = np.arange(q.shape[0]) * d
        return JointTrajectory(t, q, qd, qdd, meta={
            "source": "curobo", "label": label, "dt": d,
            "duration": float(t[-1]) if t.size else 0.0,
            "max_qd": float(np.max(np.abs(qd))) if qd.size else 0.0,
        })

    def _to_trajectory(self, result, label: str) -> JointTrajectory:
        """cuRobo TrajOptSolverResult -> JointTrajectory (14, repo order), using cuRobo's own
        interpolated timing. get_interpolated_plan() already trims, so no last_tstep here."""
        return self._jointstate_to_trajectory(result.get_interpolated_plan(), None, label)

    def fk_link(self, link_name: str, q_repo14=None) -> Pose:
        """Pelvis-frame pose of any tool_frame link at the given dual-arm config
        (defaults to zeros/home). CAMERA_FRAME (head camera) is rigidly fixed to the
        locked torso, so its pose is q-invariant -- callers may omit q for it. The
        BASE_FRAME (pelvis) is the kinematic root: cuRobo doesn't expose its FK, and it
        is identity relative to itself, so we short-circuit it (lets a base-frame camera
        extrinsic be used via source: urdf:pelvis)."""
        if link_name == BASE_FRAME:
            return Pose.Identity()
        q = np.zeros(DOF) if q_repo14 is None else q_repo14
        ks = self._mp.compute_kinematics(self._joint_state(q))
        return self._link_pose(ks, link_name)

    def fk(self, side: str, q_repo14) -> Pose:
        """Wrist-yaw pose (pelvis frame) of `side` at the given dual-arm config."""
        return self.fk_link(WRIST_FRAME[side], q_repo14)

    # --- dynamics: gravity-comp feed-forward (see docs/gravity_comp.md) ---
    def _ensure_dynamics(self):
        """Lazily build cuRobo's RNEA Dynamics from the warm planner's own
        KinematicsParams (14-DoF arms, locked base). Built on first use so the
        default (gravity_comp off) path pays nothing."""
        if self._dyn is None:
            from curobo._src.robot.dynamics.dynamics import Dynamics
            from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
            from curobo._src.types.device_cfg import DeviceCfg
            kin_cfg = self._mp.kinematics.kinematics_config
            self._dyn = Dynamics(DynamicsCfg(
                kinematics_config=kin_cfg,
                device_cfg=DeviceCfg(device=self._torch.device("cuda:0"))))
            self._dyn.setup_batch_size(batch_size=1)
            self._dyn_zero = self._torch.zeros(
                (1, DOF), dtype=self._torch.float32, device="cuda")
            # cuRobo active order -> repo (left7+right7) order for the torque vector
            self._repo_from_active = [self.active.index(j) for j in REPO_ARM]
        return self._dyn

    def gravity_torque(self, q_repo14) -> np.ndarray:
        """Gravity-comp feed-forward G(q) = RNEA(q, q̇=0, q̈=0), Nm, repo order
        (left7+right7). cuRobo-native (no pinocchio). The SIGN is hardware-validated
        (matches the pinocchio convention proven on the real robot); the MAGNITUDE
        runs ~15-20% above pinocchio's. See docs/gravity_comp.md."""
        from curobo._src.state.state_joint import JointState
        dyn = self._ensure_dynamics()
        pos = self._joint_state(q_repo14).position.to(self._torch.float32).contiguous()
        js = JointState(position=pos, velocity=self._dyn_zero,
                        acceleration=self._dyn_zero)
        tau_active = dyn.compute_inverse_dynamics(js).detach().cpu().numpy().reshape(-1)
        return tau_active[self._repo_from_active]

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

    def plan_to_pose_set(self, start_q_repo14, side: str,
                         goal_poses) -> JointTrajectory:
        """Plan to the FIRST reachable goal in a ranked list of candidate wrist poses
        (e.g. grasp candidates, best-first). Tries each via plan_to_pose and returns the
        first that solves; raises PlanningError only if ALL fail. This is the multi-
        candidate path -- the planner picks a reachable/collision-free grasp instead of
        assuming one. (FUTURE: a single native cuRobo goalset solve with num_goalset=K
        would let the optimizer choose globally; deferred -- needs GPU/hardware to
        validate the goalset tensor shapes + per-goalset success mask.)"""
        goals = list(goal_poses)
        if not goals:
            raise PlanningError("plan_to_pose_set: no candidate goals")
        last = None
        for gp in goals:
            try:
                return self.plan_to_pose(start_q_repo14, side, gp)
            except PlanningError as e:
                last = e
        raise PlanningError(
            f"plan_to_pose_set: all {len(goals)} candidates failed; last: {last}")

    def _grasp_planner(self, side: str):
        """Lazily build + cache a single-tool-frame cuRobo MotionPlanner (tool_frames =
        [active wrist]) for plan_grasp, per side. The MAIN planner has THREE tool frames (both
        wrists + the torso-locked d435 camera) for FK + dual-arm holding, but cuRobo requires a
        plan_grasp goal to cover ALL the planner's tool frames AND offsets every one to form the
        approach/lift pose -- offsetting the immovable camera (or the idle arm) makes the IK
        infeasible. GraspGenX plans grasps with a single-tool-frame planner; we mirror that by
        reducing tool_frames to the active wrist (same arms/locks -> self.active + FK joint order
        unchanged; the idle arm is seed-held by trajopt). First grasp per side pays a warmup."""
        mp = self._grasp_mp.get(side)
        if mp is None:
            import yaml
            from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
            with open(self._robot_cfg_path) as f:
                rcfg = yaml.safe_load(f)
            rcfg["kinematics"]["tool_frames"] = [WRIST_FRAME[side]]
            mp = MotionPlanner(MotionPlannerCfg.create(robot=rcfg, max_goalset=self._max_goalset))
            mp.warmup(enable_graph=False, num_warmup_iterations=1)
            self._grasp_mp[side] = mp
        return mp

    def plan_grasp_set(self, start_q_repo14, side: str, wrist_goals,
                       approach_axis: str, approach_offset: float,
                       lift_axis: str, lift_offset: float,
                       approach_in_tool_frame: bool = True,
                       lift_in_tool_frame: bool = False,
                       plan_approach: bool = True, plan_lift: bool = True,
                       disable_collision_links: Optional[List[str]] = None) -> GraspPlanOutcome:
        """Native cuRobo goalset grasp solve over K candidate wrist-yaw goals (cuRobo
        plan_grasp). Builds a K-goalset GoalToolPose on ONLY the active wrist link, calls
        plan_grasp, and returns the chosen candidate index + the approach/grasp/lift segments
        as JointTrajectory. cuRobo selects the feasible grasp and emits the segments already
        consistent with that choice. The idle arm + camera are deliberately NOT goal links:
        plan_grasp offsets EVERY goal frame to form the approach/lift pose, so including the
        torso-locked camera (which cannot move) -- or the held idle arm -- makes the
        approach/lift IK infeasible. cuRobo seeds trajopt from current_state, so the idle arm
        stays at its start config without an explicit hold goal. Axis args are UNSIGNED
        ('x'|'y'|'z'); the SIGN lives in the offset. Approach defaults to the tool frame (back
        off along the grasp approach axis, -wrist-X for our derived transform); lift defaults to
        the WORLD/pelvis frame (+Z up) -- tool-frame +Z is the spread axis, not up."""
        from curobo.types import GoalToolPose, Pose as CuPose
        goals = list(wrist_goals)
        if not goals:
            raise PlanningError("plan_grasp_set: no candidate goals")
        goals = goals[:self._max_goalset]                  # clamp to the warmed goalset size
        # cuRobo's warmup only primes the path matching max_goalset (>1 -> the GOALSET path), so
        # the single-goal (num_goalset=1) path is cold: a lone goal makes plan_grasp's first solve
        # fail ("Goalset planning returned None"). The goalset path (num_goalset>1) is robust AND
        # primes the single-goal path used internally for approach/grasp/lift -- so never feed a
        # lone goal: duplicate it (both entries are the same grasp; clamp the chosen index back).
        n_real = len(goals)
        if n_real == 1:
            goals = goals + goals
        K = len(goals)
        active = WRIST_FRAME[side]
        mp = self._grasp_planner(side)                     # single-tool-frame planner (see above)
        start = self._joint_state(start_q_repo14)

        pos = np.stack([g.translation for g in goals])           # (K, 3)
        quat = np.stack([g.quaternion_wxyz() for g in goals])    # (K, 4)
        pose_dict = {active: CuPose(position=self._tensor_k(pos),
                                    quaternion=self._tensor_k(quat))}
        goal = GoalToolPose.from_poses(pose_dict, ordered_tool_frames=[active], num_goalset=K)

        if disable_collision_links is None:               # config has grasp_contact_link_names: null
            disable_collision_links = [active]
        res = mp.plan_grasp(
            goal, start,
            grasp_approach_axis=approach_axis, grasp_approach_offset=float(approach_offset),
            grasp_approach_in_tool_frame=bool(approach_in_tool_frame),
            grasp_lift_axis=lift_axis, grasp_lift_offset=float(lift_offset),
            grasp_lift_in_tool_frame=bool(lift_in_tool_frame),
            plan_approach_to_grasp=bool(plan_approach), plan_grasp_to_lift=bool(plan_lift),
            disable_collision_links=disable_collision_links)

        gi = getattr(res, "goalset_index", None)
        idx = int(gi.view(-1)[0].item()) if gi is not None else -1
        if idx >= n_real:                                  # a duplicated lone goal -> the real one
            idx = n_real - 1

        def _ok(x):
            return bool(x is not None and x.any())

        def _seg(js, last, lbl):
            return self._jointstate_to_trajectory(js, last, lbl) if js is not None else None

        return GraspPlanOutcome(
            success=_ok(res.success), chosen_index=idx,
            approach=_seg(res.approach_interpolated_trajectory,
                          res.approach_interpolated_last_tstep, f"{side}_approach"),
            grasp=_seg(res.grasp_interpolated_trajectory,
                       res.grasp_interpolated_last_tstep, f"{side}_grasp"),
            lift=_seg(res.lift_interpolated_trajectory,
                      res.lift_interpolated_last_tstep, f"{side}_lift"),
            approach_success=_ok(res.approach_success), grasp_success=_ok(res.grasp_success),
            lift_success=_ok(res.lift_success), status=getattr(res, "status", "") or "")

    def plan_grasp_set_sweep(self, start_q_repo14, side: str, wrist_goals, strategies,
                             approach_axis: str, lift_axis: str,
                             approach_in_tool_frame: bool = True,
                             lift_in_tool_frame: bool = False,
                             disable_collision_links: Optional[List[str]] = None) -> GraspPlanOutcome:
        """Try each strategy dict ({approach_offset, lift_offset?, plan_approach?, plan_lift?})
        in order via plan_grasp_set; return the first whose outcome.success, else the last
        failing outcome (its status says why). Mirrors GraspGenX's approach/lift offset sweep."""
        if not list(wrist_goals):
            raise PlanningError("plan_grasp_set_sweep: no candidate goals")
        last = None
        for s in strategies:
            out = self.plan_grasp_set(
                start_q_repo14, side, wrist_goals,
                approach_axis=approach_axis, approach_offset=s.get("approach_offset", -0.10),
                lift_axis=lift_axis, lift_offset=s.get("lift_offset", 0.0),
                approach_in_tool_frame=approach_in_tool_frame,
                lift_in_tool_frame=lift_in_tool_frame,
                plan_approach=s.get("plan_approach", True), plan_lift=s.get("plan_lift", True),
                disable_collision_links=disable_collision_links)
            if out.success:
                return out
            last = out
        return last if last is not None else GraspPlanOutcome(
            False, -1, None, None, None, False, False, False, "no strategies")
