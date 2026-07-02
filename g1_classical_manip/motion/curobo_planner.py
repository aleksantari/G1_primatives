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

from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np

from g1_classical_manip.motion.planner_base import JointTrajectory, DOF
from g1_classical_manip.spatial.pose import Pose
from g1_classical_manip.ee.hand_base import LEFT, RIGHT

# wrist-yaw tool frames (the MVP goal frame). 5cm L_ee/palm offset is a later refinement.
WRIST_FRAME = {LEFT: "left_wrist_yaw_link", RIGHT: "right_wrist_yaw_link"}
# Active-hand collision links (palm + Dex3 fingers, from the cuRobo config's collision_link_names).
# Disabled during plan_grasp when a depth ESDF world is on, so the open hand sitting in the object/
# table ESDF at the grasp pose isn't flagged as a collision (the grasp is meant to CONTACT the object).
_HAND_PARTS = ["palm", "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
HAND_LINKS = {s: [f"{s}_hand_{p}_link" for p in _HAND_PARTS] for s in (LEFT, RIGHT)}
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

# Dex3 hand joint order AS RETURNED BY robot.hand.get_q(side) (the Dex3_1_*_JointIndex enums).
# NOTE the asymmetry: the LEFT hand is thumb,thumb,thumb,MIDDLE,MIDDLE,INDEX,INDEX but the RIGHT
# hand is thumb,thumb,thumb,INDEX,INDEX,MIDDLE,MIDDLE -- so live hand q MUST be mapped to cuRobo
# joint names by THIS table, never by raw index. Feeds live finger angles into the self-filter
# (the arms-only planning model locks all hand joints at 0 = open, so it can't mask a bent finger).
_DEX3_GETQ_JOINTS = {
    LEFT:  [f"left_hand_{p}_joint"
            for p in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")],
    RIGHT: [f"right_hand_{p}_joint"
            for p in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")],
}


def _seg_positions(seg_names, q_repo14, hand_q=None):
    """Ordered position list matching `seg_names` (the hand-active segmenter's active joints):
    arm joints from `q_repo14` (repo order, matched BY NAME) + hand joints from
    `hand_q={LEFT:(7,), RIGHT:(7,)}` (each mapped by the per-side Dex3 get_q order in
    _DEX3_GETQ_JOINTS). Hand joints with no live value default to 0.0 (open). Pure / GPU-free, so
    the index<->middle name mapping is unit-testable without building cuRobo kinematics."""
    q = np.asarray(q_repo14, float).reshape(DOF)
    vals = {n: float(q[i]) for i, n in enumerate(REPO_ARM)}
    for n in seg_names:
        vals.setdefault(n, 0.0)                        # hands (and any non-arm) default open
    if hand_q is not None:
        for side, names in _DEX3_GETQ_JOINTS.items():
            hq = hand_q.get(side)
            if hq is None:
                continue
            for n, v in zip(names, np.asarray(hq, float).reshape(7)):
                vals[n] = float(v)
    return [vals[n] for n in seg_names]


def set_curobo_log_level(level: str = "debug") -> None:
    """Turn up cuRobo's OWN logger so plan_grasp's internal failure reasons get printed instead of
    swallowed: the graph planner's 'Start or End state in collision' (graph_planner_prm), the IK
    stage's 'No grasp in goal set was reachable' (motion_planner), and per-stage trajopt warnings.
    `level` in {debug, info, warning, error}; cuRobo defaults to 'warning'. Call once before planning
    (09/12 expose it as --debug-planner). Best-effort: never blocks if the logging API moves."""
    try:
        from curobo.logging import setup_logger
        setup_logger(level)
        print(f"[curobo] log level -> {level}")
    except Exception as e:                       # noqa: BLE001 - diagnostics only, never block a run
        print(f"set_curobo_log_level: could not set cuRobo log level ({e})")


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
            robot=robot_cfg_path, max_goalset=self._max_goalset, **self._solver_kwargs()))
        self._mp.warmup(enable_graph=True, num_warmup_iterations=5)
        self._grasp_mp = {}       # lazy per-side single-tool-frame planners for plan_grasp
        self._cw_loaded = set()   # sides whose grasp planner has a LIVE ESDF loaded (vs the empty
                                  # build grid) -- world_check refuses to "pass" an unloaded world
        # Depth-ESDF collision world (planner.yaml grasp.collision_world). OFF by default: when on,
        # the grasp planner is built voxel-capable and the head depth ESDF is loaded before plan_grasp
        # so the approach routes around the object/table. See motion/collision_world.py.
        self._cw_cfg = (self.planner_cfg.get("grasp") or {}).get("collision_world") or {}
        self._cw_enabled = bool(self._cw_cfg.get("enabled", False))
        self._esdf_mapper = None  # lazy EsdfMapper (depth -> ESDF), built on first depth frame
        self._segmenter = None    # lazy cuRobo RobotSegmenter (self-filter the robot out of depth)
        self._seg_names = None     # its active joint order (arm + hands; live finger tracking)
        self.active = list(self._mp.joint_names)          # 14 arm joints, cuRobo order
        self.tool_frames = list(self._mp.tool_frames)
        assert len(self.active) == DOF, f"expected {DOF} active joints, got {len(self.active)}"
        assert set(self.active) == set(REPO_ARM), "active joints != repo arm set"
        self._dyn = None          # lazy cuRobo RNEA Dynamics (gravity-comp; built on 1st use)

    # --- helpers ---
    def _solver_kwargs(self) -> dict:
        """Extra MotionPlannerCfg.create() kwargs from planner.yaml `solver` -- solver seed counts +
        collision margin/tolerances. Empty -> cuRobo defaults (num_ik_seeds 32, num_trajopt_seeds 4,
        optimizer_collision_activation_distance 0.01). MORE seeds explore the arm's 7-DoF redundancy
        (and the K-goal set) for a collision-free grasp IK -- the real lever for 'No grasp in goal set
        was reachable' with the collision world on (plan_grasp has no max_attempts and a FIXED random
        seed, so re-tries are deterministic; seeds are how you 'try harder' in one solve). Applied to
        BOTH the main planner and every per-side grasp planner."""
        s = self.planner_cfg.get("solver") or {}
        kw = {}
        if s.get("num_ik_seeds") is not None:
            kw["num_ik_seeds"] = int(s["num_ik_seeds"])
        if s.get("num_trajopt_seeds") is not None:
            kw["num_trajopt_seeds"] = int(s["num_trajopt_seeds"])
        if s.get("collision_activation_distance") is not None:
            kw["optimizer_collision_activation_distance"] = float(s["collision_activation_distance"])
        if s.get("position_tolerance") is not None:
            kw["position_tolerance"] = float(s["position_tolerance"])
        if s.get("orientation_tolerance") is not None:
            kw["orientation_tolerance"] = float(s["orientation_tolerance"])
        return kw

    def _joint_state(self, q_repo14, names: Optional[List[str]] = None):
        """repo-order (14,) config -> cuRobo JointState in `names` order (default: the MAIN planner's
        active order). Pass a specific planner's ``mp.joint_names`` when feeding ITS low-level
        compute_kinematics -- cuRobo does NOT reorder by name there (plan-level entry points do)."""
        from curobo.types import JointState
        names = list(self.active if names is None else names)
        q = np.asarray(q_repo14, float).reshape(DOF)
        q_active = q[[REPO_ARM.index(j) for j in names]]         # repo -> cuRobo order
        t = self._torch.tensor(q_active, dtype=self._torch.float32, device="cuda").unsqueeze(0)
        return JointState.from_position(t, joint_names=names)

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

    def collision_spheres(self, q_repo14=None):
        """The cuRobo collision spheres -- the geometry cuRobo ACTUALLY collision-checks (self-
        collision + vs the world) -- at the given dual-arm config, in the PELVIS frame. Returns
        ``(centers (N,3), radii (N,))`` float32 for the ACTIVE spheres (radius > 0). Defaults to
        home (zeros). Visualize these to see WHY a 'Start or End state in collision' fires: a sphere
        sitting inside the ESDF voxels (world collision) or two links' spheres overlapping (self-
        collision). NOTE these come from the MAIN 14-DoF model (both arms, hands locked open), which
        is the same sphere set the self-collision check uses."""
        q = np.zeros(DOF) if q_repo14 is None else q_repo14
        ks = self._mp.compute_kinematics(self._joint_state(q))
        sph = ks.robot_spheres
        arr = (sph.detach().cpu().numpy() if hasattr(sph, "detach") else np.asarray(sph))
        arr = arr.reshape(-1, 4)
        keep = arr[:, 3] > 0.0                       # cuRobo marks disabled spheres with radius <= 0
        return arr[keep, :3].astype(np.float32), arr[keep, 3].astype(np.float32)

    def _sphere_link_names(self, mp=None):
        """(443,) list: the link NAME each collision sphere belongs to, so a self-collision pair
        can be labelled by link (right_hand_thumb_2_link <-> right_wrist_yaw_link) rather than raw
        sphere index. From the cuRobo kinematics (link_sphere_idx_map + link_name_to_idx_map).
        Default: the MAIN planner; pass a grasp planner to label ITS sphere layout (its joint --
        and possibly sphere -- ordering differs; only the main-planner result is cached)."""
        if mp is not None and mp is not self._mp:
            kc = mp.kinematics.kinematics_config
            idx_map = kc.link_sphere_idx_map.detach().cpu().numpy().reshape(-1).astype(int)
            name_of = {int(v): k for k, v in kc.link_name_to_idx_map.items()}
            return [name_of.get(int(li), f"link_{int(li)}") for li in idx_map]
        if getattr(self, "_sphere_links", None) is None:
            kc = self._mp.kinematics.kinematics_config
            idx_map = kc.link_sphere_idx_map.detach().cpu().numpy().reshape(-1).astype(int)
            name_of = {int(v): k for k, v in kc.link_name_to_idx_map.items()}
            self._sphere_links = [name_of.get(int(li), f"link_{int(li)}") for li in idx_map]
        return self._sphere_links

    def _self_collision_pairs(self):
        """(P,2) int sphere-index pairs cuRobo ACTUALLY self-collision-checks -- adjacent links and
        every self_collision_ignore entry (incl. build_g1_dex3.patch()'s torso<->shoulder and
        thumb_1<->wrist_yaw) are ALREADY removed by cuRobo, so iterating these is ignore-matrix
        faithful for free. Also returns cuRobo's PER-SPHERE (443,) sphere_padding (the activation
        buffer it adds to each radius) so the overlap test matches what the solver rejects on."""
        if getattr(self, "_self_pairs", None) is None:
            scc = self._mp.kinematics.get_self_collision_config()
            self._self_pairs = scc.collision_pairs.detach().cpu().numpy().astype(int).reshape(-1, 2)
            pad = getattr(scc, "sphere_padding", None)
            if pad is None:
                self._self_pad = np.zeros(self._mp.kinematics.kinematics_config.total_spheres)
            else:
                pad = pad.detach().cpu().numpy() if hasattr(pad, "detach") else np.asarray(pad)
                self._self_pad = pad.reshape(-1).astype(float)
        return self._self_pairs, self._self_pad

    @staticmethod
    def _quat_to_R(wxyz):
        w, x, y, z = [float(v) for v in wxyz]
        return np.array([
            [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
            [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)]])

    def _wrist_jacobian_fd(self, q_repo14, side: str, eps: float = 1e-5):
        """The 6x7 geometric Jacobian (linear 3 + angular 3, pelvis frame) of `side`'s wrist w.r.t.
        that arm's 7 joints, by FINITE DIFFERENCE over cuRobo FK. cuRobo's KinematicsState.tool_jacobians
        is a zero placeholder unless the model is built with compute_jacobian=True, so we FD it off the
        reliable tool_poses (validated stable for eps 1e-3..1e-6). Used for the singularity measure."""
        q = np.asarray(q_repo14, float).reshape(DOF)
        cols = [REPO_ARM.index(n) for n in (REPO_ARM[0:7] if side == LEFT else REPO_ARM[7:DOF])]
        p0 = self.fk(side, q)
        R0 = self._quat_to_R(p0.quaternion_wxyz())
        J = np.zeros((6, 7))
        for n, k in enumerate(cols):
            qp = q.copy(); qp[k] += eps
            pp = self.fk(side, qp)
            J[0:3, n] = (pp.translation - p0.translation) / eps
            Rrel = self._quat_to_R(pp.quaternion_wxyz()) @ R0.T
            v = np.array([Rrel[2, 1]-Rrel[1, 2], Rrel[0, 2]-Rrel[2, 0], Rrel[1, 0]-Rrel[0, 1]])
            ang = np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))
            J[3:6, n] = (v / 2 if ang < 1e-9 else (ang / (2*np.sin(ang))) * v) / eps
        return J

    def diagnose(self, q_repo14, side: str, world_points=None, voxel_size: float = 0.01,
                 near_limit_deg: float = 5.0, sing_eps: float = 0.01, top: int = 8, show: bool = True):
        """Explain WHY a config is rejected by cuRobo -- the per-element breakdown behind a generic
        'Start or End state in collision' / 'No grasp in goal set was reachable'. At `q_repo14` (repo
        14-vector) for arm `side`, reports:
          * SELF-collision: which link/sphere PAIRS overlap + penetration mm (ignore-matrix faithful).
          * WORLD/ESDF: which spheres sit inside the depth voxels + depth mm (uses the SAME occupied
            points 09/12 overlay in red; pass `world_points` or defaults to collision_world_points()).
          * JOINT LIMITS: the side arm's joints closest to a limit (deg), flags any within near_limit_deg.
          * SINGULARITY: condition number + Yoshikawa manipulability of the side's 6x7 wrist Jacobian
            (flags cond > cond_warn) -- the near-singular configs seen at the task-space edges.
        Returns a dict (incl. offending sphere centers/radii for a red overlay). `show` prints a report.
        Runs on the MAIN 14-DoF model (self._mp) -- the same sphere set + limits the grasp solve uses."""
        q = np.asarray(q_repo14, float).reshape(DOF)
        ks = self._mp.compute_kinematics(self._joint_state(q))
        sph = ks.robot_spheres.detach().cpu().numpy().reshape(-1, 4)   # (443, 4): xyz + radius
        links = self._sphere_link_names()
        c, r = sph[:, :3], sph[:, 3]

        # --- (a) self-collision: pairwise overlap over cuRobo's checked pairs (ignores pre-removed)
        pairs, pad = self._self_collision_pairs()
        pi, pj = pairs[:, 0], pairs[:, 1]
        valid = (r[pi] > 0) & (r[pj] > 0)
        dist = np.linalg.norm(c[pi] - c[pj], axis=1)
        # padded overlap (>0 => cuRobo's self-collision would fire): radii + per-sphere padding - gap
        overlap = (r[pi] + r[pj] + pad[pi] + pad[pj]) - dist
        hit = valid & (overlap > 0) & (np.asarray(links)[pi] != np.asarray(links)[pj])
        self_rows = [{"i": int(pi[k]), "j": int(pj[k]), "link_i": links[pi[k]], "link_j": links[pj[k]],
                      "penetration_mm": float(overlap[k] * 1000.0)} for k in np.nonzero(hit)[0]]
        self_rows.sort(key=lambda d: -d["penetration_mm"])

        # --- (b) world/ESDF: robot spheres inside the occupied voxels (sphere-vs-nearest-voxel)
        wp = world_points if world_points is not None else self.collision_world_points()
        world_rows = []
        if wp is not None and len(wp):
            wp = np.asarray(wp, float).reshape(-1, 3)
            act = np.nonzero(r > 0)[0]
            for s in act:
                d = np.linalg.norm(wp - c[s], axis=1).min()           # nearest occupied voxel centre
                depth = (r[s] + 0.5 * voxel_size) - d                 # >0 => sphere overlaps a voxel
                if depth > 0:
                    world_rows.append({"i": int(s), "link": links[s], "depth_mm": float(depth * 1000.0)})
            world_rows.sort(key=lambda d: -d["depth_mm"])

        # --- (c) joint limits: this side's 7 joints, closest to a bound first
        jl = self._mp.kinematics.get_joint_limits()
        lo = jl.position[0].detach().cpu().numpy(); hi = jl.position[1].detach().cpu().numpy()
        side_names = set(REPO_ARM[0:7]) if side == LEFT else set(REPO_ARM[7:DOF])
        cols = [k for k, n in enumerate(self.active) if n in side_names]
        deg = 180.0 / np.pi
        qa = q[[REPO_ARM.index(self.active[k]) for k in cols]]
        limit_rows = []
        for k, qk in zip(cols, qa):
            m = min(qk - lo[k], hi[k] - qk)                            # rad to nearest bound
            limit_rows.append({"joint": self.active[k], "q_deg": float(qk * deg),
                               "margin_deg": float(m * deg), "lo_deg": float(lo[k] * deg),
                               "hi_deg": float(hi[k] * deg)})
        limit_rows.sort(key=lambda d: d["margin_deg"])

        # --- (d) singularity: side's 6x7 wrist Jacobian (FD -- cuRobo tool_jacobians is a zero
        # placeholder). sigma_min = distance-to-rank-loss (the flag; healthy configs run ~0.02-0.08),
        # + condition number + Yoshikawa manipulability sqrt(det(J J^T)).
        J = self._wrist_jacobian_fd(q, side)
        sv = np.linalg.svd(J, compute_uv=False)
        smin = float(sv.min()); smax = float(sv.max())
        cond = float(smax / smin) if smin > 1e-9 else float("inf")
        manip = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
        sing = {"cond": cond, "manip": manip, "sigma_min": smin}

        off_idx = sorted({d["i"] for d in self_rows} | {d["j"] for d in self_rows}
                         | {d["i"] for d in world_rows})
        out = {"self": self_rows, "world": world_rows, "limits": limit_rows, "singularity": sing,
               "offending_centers": c[off_idx].astype(np.float32) if off_idx else np.zeros((0, 3), np.float32),
               "offending_radii": r[off_idx].astype(np.float32) if off_idx else np.zeros((0,), np.float32)}
        if show:
            self._print_diagnose(out, side, near_limit_deg, sing_eps, top)
        return out

    @staticmethod
    def _print_diagnose(out, side, near_limit_deg, sing_eps, top):
        print(f"[diagnose] side={side}  (self-collision faithful to the ignore matrix; "
              f"world = sphere vs the red ESDF voxels)")
        sr = out["self"]
        print(f"  SELF-COLLISION: {len(sr)} overlapping pair(s)"
              + (":" if sr else "  -- none (start/end self-collision-free)"))
        for d in sr[:top]:
            print(f"    {d['link_i']}[{d['i']}] <-> {d['link_j']}[{d['j']}]  overlap {d['penetration_mm']:.1f} mm")
        if len(sr) > top:
            print(f"    ... +{len(sr) - top} more")
        wr = out["world"]
        if out["world"] is not None:
            print(f"  WORLD/ESDF: {len(wr)} sphere(s) inside the depth voxels"
                  + (":" if wr else "  -- none (or no collision world loaded)"))
            for d in wr[:top]:
                print(f"    {d['link']}[{d['i']}]  {d['depth_mm']:.1f} mm inside")
            if len(wr) > top:
                print(f"    ... +{len(wr) - top} more")
        lr = out["limits"]
        flagged = [d for d in lr if d["margin_deg"] < near_limit_deg]
        print(f"  JOINT LIMITS ({side} arm): closest {min(3, len(lr))}"
              + (f"  [{len(flagged)} within {near_limit_deg:g} deg!]" if flagged else ""))
        for d in lr[:3]:
            bang = "  <-- near limit" if d["margin_deg"] < near_limit_deg else ""
            print(f"    {d['joint']}: {d['margin_deg']:.1f} deg to bound "
                  f"(q={d['q_deg']:.1f}, [{d['lo_deg']:.0f}, {d['hi_deg']:.0f}]){bang}")
        s = out["singularity"]
        near = "  <-- NEAR-SINGULAR" if s["sigma_min"] < sing_eps else ""
        print(f"  SINGULARITY ({side} wrist 6x7): sigma_min={s['sigma_min']:.4f}  "
              f"cond={s['cond']:.1f}  manip={s['manip']:.4f}{near}")

    def diagnose_pose(self, side: str, wrist_pose, world_points=None, start_q_repo14=None,
                      label: str = "pose", **kw):
        """Diagnose the CONFIG that reaches a target wrist pose (e.g. a pre-grasp) -- the END-state
        counterpart to diagnose() for 'Planning to approach pose failed'. plan_grasp rejects the
        pre-grasp but does NOT return its config, so we re-solve one on the MAIN planner, which has
        NO world (only the grasp planners are voxel-capable) -> a WORLD-IGNORING config for the pose,
        then diagnose() it against `world_points` (the ESDF). Interpreting the result:
          * reachable + WORLD non-empty -> the pre-grasp genuinely puts arm spheres in the ESDF
            (real clutter collision; that grasp's approach is blocked -- not a sphere-model issue).
          * reachable + WORLD empty     -> a collision-free pre-grasp config EXISTS; plan_grasp's
            goalset/seeds just didn't find it -> more num_ik_seeds/num_trajopt_seeds may fix it.
          * unreachable                 -> the pose is infeasible even ignoring the world (self-
            collision / no IK) -> the grasp candidate itself is bad, not the world.
        NOTE plan_to_pose returns ONE IK branch (elbow up/down); a WORLD hit doesn't prove EVERY
        branch collides, but a WORLD miss proves a free one exists. Returns the diagnose() dict
        (empty sections if unreachable) + {'reachable', 'config'}."""
        start = np.zeros(DOF) if start_q_repo14 is None else start_q_repo14
        empty = {"self": [], "world": [], "limits": [], "singularity": {},
                 "offending_centers": np.zeros((0, 3), np.float32),
                 "offending_radii": np.zeros((0,), np.float32)}
        try:
            traj = self.plan_to_pose(start, side, wrist_pose)      # main planner: self-only, NO world
            cfg = traj.q[-1]
        except PlanningError as e:
            print(f"[diagnose_pose] '{label}' ({side}) UNREACHABLE ignoring the world "
                  f"(self-collision / no IK) -> the POSE itself is infeasible, not a world issue\n"
                  f"    {e}")
            return {**empty, "reachable": False, "config": None}
        print(f"[diagnose_pose] '{label}' ({side}) reached at "
              f"{np.round(np.asarray(wrist_pose.translation), 3)} (world-free IK); "
              f"checking that config vs the ESDF:")
        out = self.diagnose(cfg, side, world_points=world_points, **kw)
        out["reachable"] = True
        out["config"] = np.asarray(cfg, float)
        return out

    def _reach(self, side, pose, start_q):
        """World-free config reaching `pose` (main planner, no ESDF), or None if plan_to_pose can't.
        NOTE trajopt: None folds no-IK + no-collision-free-path; for a short home->pose move it is
        IK-bound in practice."""
        try:
            return self.plan_to_pose(start_q, side, pose).q[-1]
        except PlanningError:
            return None

    def diagnose_candidates(self, side, grasp_goals, pregrasp_goals, world_points=None,
                            k: int = 0, start_q_repo14=None, show: bool = True):
        """Sweep the top-`k` grasp candidates (k<=0 == ALL) to localize a plan_grasp approach failure
        across the goalset (plan_grasp picks ONE of K -- the top-confidence one being bad doesn't mean
        all are). Per candidate: is the GRASP pose reachable, is the PRE-GRASP reachable, and if so
        does that config penetrate the world ESDF? All reachability on the world-free main planner.
        The grasp-vs-pre-grasp comparison is the confound remover: a grasp reachable from home but its
        pre-grasp (a short back-off) not is a genuine reach/geometry boundary, not a path artifact.
        Reading the summary: some PRE-GRASP reachable AND world-free -> a collision-free approach
        EXISTS and plan_grasp didn't converge to it (raise solver.num_ik_seeds / num_trajopt_seeds --
        THE 'more resources' case); reachable pre-grasps ALL world-blocked -> real clutter (adjust
        approach / crop-or-declutter the ESDF); grasp reachable but NO pre-grasp -> the back-off leaves
        reach or hits a wrist limit (shrink approach_dist / change axis); no grasp reachable ->
        candidates are out of reach for this arm (bad grasps / wrong side). ~2 world-free plan solves
        per candidate (~0.4s each) -- a full sweep of a large goalset takes a minute+."""
        start = np.zeros(DOF) if start_q_repo14 is None else start_q_repo14
        n = min(len(grasp_goals), len(pregrasp_goals))
        if int(k) > 0:
            n = min(n, int(k))
        # world predicate: prefer the TRUTH-TEST (the grasp planner's own ESDF checker -- the exact
        # 'Start or End state in collision' gate) over the legacy geometric voxel-centre test, which
        # under-counts (no eta semantics) and produced the unverified '74/200 free' readings.
        truth = self._cw_enabled and side in self._cw_loaded
        print(f"[diagnose_candidates] sweeping {n} candidate(s) "
              f"(~{n * 0.8:.0f}s; 2 world-free plan solves each; world predicate: "
              f"{'cuRobo ESDF gate (truth)' if truth else 'geometric (no live ESDF loaded)'}) ...")
        rows = []
        for i in range(n):
            gcfg = self._reach(side, grasp_goals[i], start)
            pcfg = self._reach(side, pregrasp_goals[i], start)
            wh = None
            if pcfg is not None:
                if truth:
                    wh = len(self.world_check(side, pcfg, show=False)["violations"])
                elif world_points is not None and len(world_points):
                    wh = len(self.diagnose(pcfg, side, world_points=world_points, show=False)["world"])
            rows.append({"i": i, "grasp": gcfg is not None, "pregrasp": pcfg is not None,
                         "world_hits": wh})
            if n > 20 and (i + 1) % 20 == 0:
                print(f"    ... {i + 1}/{n}")
        s = {"n": n, "rows": rows,
             "grasp_reach": sum(r["grasp"] for r in rows),
             "pregrasp_reach": sum(r["pregrasp"] for r in rows),
             "pregrasp_free": sum(1 for r in rows if r["pregrasp"] and r["world_hits"] == 0),
             "pregrasp_blocked": sum(1 for r in rows if r["pregrasp"] and (r["world_hits"] or 0) > 0)}
        if show:
            self._print_candidates(s, side)
        return s

    @staticmethod
    def _print_candidates(s, side):
        rows = s["rows"]
        free = [r for r in rows if r["pregrasp"] and r["world_hits"] == 0]
        print(f"[diagnose_candidates] {side}: swept {s['n']} candidate(s) "
              f"(reachability on the world-free planner)")
        print(f"  summary: grasp-reachable {s['grasp_reach']}/{s['n']}, "
              f"pre-grasp-reachable {s['pregrasp_reach']}/{s['n']} "
              f"(world-free {s['pregrasp_free']}, ESDF-blocked {s['pregrasp_blocked']})")
        if free:                                      # the actionable ones: plan_grasp should hit these
            idx = ", ".join(str(r["i"]) for r in free[:20])
            print(f"  world-FREE reachable pre-grasp candidate idx: [{idx}"
                  + (f", +{len(free) - 20} more" if len(free) > 20 else "") + "]")
        show_rows = rows if s["n"] <= 20 else (       # full list when small; else free + a sample
            free[:10] + [r for r in rows if not (r["pregrasp"] and r["world_hits"] == 0)][:10])
        for r in show_rows:
            wh = "n/a" if r["world_hits"] is None else (
                f"{r['world_hits']} in ESDF" if r["world_hits"] else "world-free")
            print(f"  cand {r['i']:3d}: grasp {'OK ' if r['grasp'] else 'NO '} | "
                  f"pre-grasp {'OK ' if r['pregrasp'] else 'NO '} | {wh}")
        if s["n"] > 20:
            print(f"  (showing {len(show_rows)} of {s['n']}; summary above is the full count)")
        if s["pregrasp_free"] > 0:
            print("  => a reachable, world-FREE pre-grasp EXISTS. plan_grasp commits to ONE goalset "
                  "winner (no internal retry) -> the candidate-retry loop should reach these; if it "
                  "still fails, world_check the START config (a start in the ESDF fails EVERY plan) "
                  "and check collision_world.exclude_object.")
        elif s["pregrasp_reach"] > 0 and s["pregrasp_blocked"] == s["pregrasp_reach"]:
            print("  => every reachable pre-grasp is INSIDE the ESDF -> real clutter collision: "
                  "adjust the approach (offset/axis), crop the ESDF, or de-clutter.")
        elif s["grasp_reach"] > 0 and s["pregrasp_reach"] == 0:
            print("  => grasps reachable but NO pre-grasp is -> the back-off leaves the arm's reach "
                  "or hits a wrist limit: shrink approach_dist or change approach axis. "
                  "(If EVERY pre-grasp fails while grasps pass, also sanity-check the approach "
                  "reconstruction direction.)")
        elif s["grasp_reach"] == 0:
            print("  => no grasp pose is even reachable for this arm -> candidates are out of reach "
                  "(bad grasps / wrong side / object out of the arm's workspace).")

    def world_check(self, side: str, q_repo14, activation_distance: float = 0.0,
                    near_distance: Optional[float] = None, disable_links: Optional[List[str]] = None,
                    top: int = 8, show: bool = True):
        """TRUTH-TEST a config against the GRASP PLANNER'S OWN loaded ESDF world -- the exact same
        checker + predicate that rejects plan_grasp with 'Start or End state in collision'. Unlike
        diagnose()'s geometric sphere-vs-occupied-voxel-centre approximation, this queries cuRobo's
        scene_collision_checker: per-sphere cost = (radius + eta) - esdf(center), > 0 == collision.
        The graph-planner/IK feasibility gate runs at eta = ``activation_distance`` = 0.0 (cuRobo
        curobo/content/configs/task/metrics_base.yml: scene_collision activation_distance 0.0) --
        planner.yaml's solver.collision_activation_distance shapes only the OPTIMIZER cost. A second
        query at ``near_distance`` (default: that optimizer eta) reports the spheres the optimizer is
        actively pushing on. ``disable_links`` mimics plan_grasp Step-1/3 (spheres zeroed for those
        links); Step-2 (the approach plan -- where the failures happen) runs with ALL links enabled,
        the default here. Returns {violations, near, in_collision, max_penetration_mm,
        offending_centers/radii, world_loaded, n_spheres}; empty + world_loaded=False when this
        side's grasp planner has no LIVE ESDF loaded (never trust the empty build grid)."""
        mp = self._grasp_planner(side)
        out = {"violations": [], "near": [], "in_collision": False, "max_penetration_mm": 0.0,
               "offending_centers": np.zeros((0, 3), np.float32),
               "offending_radii": np.zeros((0,), np.float32),
               "world_loaded": side in self._cw_loaded, "n_spheres": 0}
        if mp.scene_collision_checker is None or not out["world_loaded"]:
            if show:
                print(f"[world_check] {side}: NO live ESDF loaded in this side's grasp planner "
                      f"(collision world {'enabled but not built yet' if self._cw_enabled else 'OFF'})"
                      f" -- nothing to check against.")
            return out
        assert set(mp.joint_names) == set(self.active), "grasp planner joints != main planner set"
        from curobo._src.geom.collision.buffer_collision import CollisionBuffer
        if disable_links:
            mp.disable_link_collision(list(disable_links))
        try:
            # NOTE: the grasp planner's joint ORDER differs from the main planner's, and low-level
            # compute_kinematics does NOT reorder by name -> build the state in ITS order.
            ks = mp.compute_kinematics(self._joint_state(q_repo14, names=mp.joint_names))
        finally:
            if disable_links:
                mp.enable_link_collision(list(disable_links))
        sph = ks.robot_spheres
        out["n_spheres"] = int(sph.shape[-2])

        def _query(eta: float):
            buf = CollisionBuffer.from_shape(sph.shape, mp.device_cfg)
            d = mp.scene_collision_checker.get_sphere_distance(
                ks, buf, mp.device_cfg.to_device([1.0]), mp.device_cfg.to_device([float(eta)]))
            return d.detach().float().cpu().numpy().reshape(-1)      # (N,) cost; >0 == collision

        links = self._sphere_link_names(mp)                          # THIS planner's sphere layout
        arr = sph.detach().float().cpu().numpy().reshape(-1, 4)
        radii = arr[:, 3]
        cost0 = _query(activation_distance)                          # the FAILURE predicate (eta 0)
        near_eta = (float((self.planner_cfg.get("solver") or {}).get(
            "collision_activation_distance", 0.01)) if near_distance is None else float(near_distance))
        cost_n = _query(near_eta) if near_eta > activation_distance else cost0

        def _rows(cost):
            idx = np.nonzero((cost > 0.0) & (radii > 0.0))[0]        # r<=0 = disabled spheres
            rows = [{"i": int(i), "link": links[i], "penetration_mm": float(cost[i] * 1000.0)}
                    for i in idx]
            rows.sort(key=lambda r: -r["penetration_mm"])
            return rows

        out["violations"] = _rows(cost0)
        hit = {r["i"] for r in out["violations"]}
        out["near"] = [r for r in _rows(cost_n) if r["i"] not in hit]
        out["in_collision"] = bool(out["violations"])
        out["max_penetration_mm"] = out["violations"][0]["penetration_mm"] if out["violations"] else 0.0
        off = sorted(hit)
        out["offending_centers"] = arr[off, :3].astype(np.float32) if off else out["offending_centers"]
        out["offending_radii"] = radii[off].astype(np.float32) if off else out["offending_radii"]
        if show:
            self._print_world_check(out, side, activation_distance, near_eta, top)
        return out

    @staticmethod
    def _print_world_check(out, side, eta, near_eta, top):
        v, n = out["violations"], out["near"]
        print(f"[world_check] {side}: cuRobo's OWN ESDF query ({out['n_spheres']} spheres, "
              f"gate eta={eta:g})")
        print(f"  IN COLLISION (the 'Start or End state in collision' predicate): {len(v)} sphere(s)"
              + (":" if v else "  -- config PASSES the real gate"))
        for r in v[:top]:
            print(f"    {r['link']}[{r['i']}]  {r['penetration_mm']:.1f} mm inside")
        if len(v) > top:
            print(f"    ... +{len(v) - top} more")
        if n:
            print(f"  NEAR (within optimizer eta={near_eta:g}, pushed but not gate-failing): "
                  f"{len(n)} sphere(s)")
            for r in n[:min(top, 4)]:
                print(f"    {r['link']}[{r['i']}]  {r['penetration_mm']:.1f} mm into the margin")
            if len(n) > min(top, 4):
                print(f"    ... +{len(n) - min(top, 4)} more")

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
            scene_model = None
            if self._cw_enabled:                 # voxel-capable: allocate the ESDF channel at build
                from curobo._src.geom.types import SceneCfg
                from g1_classical_manip.motion.collision_world import empty_esdf_grid
                p = self._cw_params()
                scene_model = SceneCfg(voxel=[empty_esdf_grid(
                    p["grid_center"], p["extent_m"], p["esdf_voxel_size"])])
            mp = MotionPlanner(MotionPlannerCfg.create(
                robot=rcfg, max_goalset=self._max_goalset, scene_model=scene_model,
                **self._solver_kwargs()))
            mp.warmup(enable_graph=False, num_warmup_iterations=1)
            self._grasp_mp[side] = mp
        return mp

    @property
    def collision_world_enabled(self) -> bool:
        """True when the depth-ESDF collision world is on (planner.yaml grasp.collision_world)."""
        return self._cw_enabled

    def set_collision_world(self, enabled: bool, cfg: Optional[dict] = None) -> None:
        """Toggle the depth-ESDF collision world at runtime (e.g. a `--collision-world` flag).
        Call BEFORE the first grasp: the grasp planner is built voxel-capable lazily, so any
        already-built (non-voxel) grasp planner is dropped here to rebuild with the new setting."""
        self._cw_enabled = bool(enabled)
        if cfg is not None:
            self._cw_cfg = cfg
        self._grasp_mp = {}        # force rebuild with/without the voxel channel
        self._cw_loaded = set()    # rebuilt planners start with the EMPTY grid until update_grasp_world
        self._esdf_mapper = None
        self._segmenter = None
        self._seg_names = None

    def _cw_params(self) -> dict:
        """Depth-ESDF collision-world params (planner.yaml grasp.collision_world) with defaults
        sized for the G1 tabletop workspace in the pelvis frame. extent is a multiple of
        esdf_voxel_size so the empty (build) grid and the live ESDF have identical voxel shape."""
        c = self._cw_cfg
        return dict(
            grid_center=list(c.get("grid_center", [0.4, 0.0, 0.2])),
            extent_m=list(c.get("extent_m", [1.2, 1.2, 1.0])),
            esdf_voxel_size=float(c.get("esdf_voxel_size", 0.02)),
            tsdf_voxel_size=float(c.get("tsdf_voxel_size", 0.01)),
            depth_min_m=float(c.get("depth_min_m", 0.1)),
            depth_max_m=float(c.get("depth_max_m", 2.0)),
            self_filter=bool(c.get("self_filter", True)),
            robot_mask_margin=float(c.get("robot_mask_margin", 0.02)),
            exclude_object=bool(c.get("exclude_object", True)),
            exclude_object_dilate_px=int(c.get("exclude_object_dilate_px", 4)),
        )

    def _build_hand_segmenter(self, margin: float):
        """Build a RobotSegmenter whose kinematics has the HAND joints ACTIVE (unlocked from the
        arms-only planning config), so the self-filter masks the fingers at their LIVE pose. The
        planner's own model locks every hand joint at 0 (open) and so can't mask a bent finger;
        this dedicated model leaves the planner untouched. Returns (segmenter, active_joint_names).
        Mirrors RobotSegmenter.from_robot_file but forces ops_dtype=float32 (the default bfloat16
        trips cuRobo's own check_float32_tensors on the cdist path)."""
        import copy
        from curobo._src.util_file import load_yaml
        from curobo._src.types.robot import RobotCfg
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.robot.kinematics.kinematics import Kinematics
        from curobo._src.perception.robot_segmenter import RobotSegmenter
        cfg = copy.deepcopy(load_yaml(self._robot_cfg_path))
        lj = cfg["kinematics"].get("lock_joints") or {}
        cfg["kinematics"]["lock_joints"] = {k: v for k, v in lj.items() if "hand" not in k}
        kin = Kinematics(RobotCfg.create(cfg, device_cfg=DeviceCfg()).kinematics)
        seg = RobotSegmenter(kin, distance_threshold=margin, use_cuda_graph=False,
                             ops_dtype=self._torch.float32)
        return seg, list(kin.joint_names)

    def _seg_joint_state(self, q_repo14, hand_q=None):
        """JointState over the segmenter's active joints (arm + hands), ordered to its kinematics."""
        from curobo.types import JointState
        pos = _seg_positions(self._seg_names, q_repo14, hand_q)
        t = self._torch.tensor(pos, dtype=self._torch.float32, device="cuda").unsqueeze(0)
        return JointState.from_position(t, joint_names=list(self._seg_names))

    def robot_depth_filter(self, q_repo14, hand_q=None, margin: Optional[float] = None):
        """A ``CameraObservation -> robot-removed depth`` callback (cuRobo RobotSegmenter): zeros
        the depth pixels within `margin` of the robot's collision spheres at the given config, so
        the head camera doesn't fuse the robot's own arm/hands into the collision world. Masks the
        arm at `q_repo14` (repo order) AND the fingers at their LIVE pose when `hand_q`
        ({LEFT:(7,), RIGHT:(7,)} from robot.hand.get_q) is passed -- else the hands default to
        open (0). Returns None if q is None."""
        if q_repo14 is None:
            return None
        if self._segmenter is None:
            m = self._cw_params()["robot_mask_margin"] if margin is None else float(margin)
            self._segmenter, self._seg_names = self._build_hand_segmenter(m)
        js = self._seg_joint_state(q_repo14, hand_q)

        def _filter(obs):
            _, filtered = self._segmenter.get_robot_mask_from_active_js(obs, js)
            return filtered

        return _filter

    def update_grasp_world(self, side: str, depth_mm, intrinsics: dict, T_pelvis_camera,
                           q_repo14=None, hand_q=None, object_mask=None) -> bool:
        """Build a fresh ESDF from the head depth and load it into `side`'s grasp planner, so the
        next plan_grasp avoids the table/clutter. No-op (returns False) when the collision world is
        disabled or there is no depth. `depth_mm` (H,W float32 mm), `intrinsics` {fx,fy,cx,cy},
        `T_pelvis_camera` = the camera optical pose in pelvis. `q_repo14` = the arm config at which
        the depth was captured + `hand_q` ({LEFT:(7,),RIGHT:(7,)} live finger angles) -> the robot
        (arm AND fingers) is self-filtered out of the depth (essential: else the arm/hand is fused
        into the world and plan_grasp starts the arm inside a copy of itself). `object_mask`
        ((H,W) bool, depth-aligned -- the TARGET's SAM3 mask) cuts the object OUT of the world
        (gated by collision_world.exclude_object, dilated exclude_object_dilate_px): the reference
        end2end design -- the gripper has to REACH the object, and plan_grasp's Step-2 approach
        plans with ALL links enabled (hand included), so an in-world target blocks its own grasp."""
        if not self._cw_enabled or depth_mm is None:
            return False
        import numpy as _np
        from curobo._src.geom.types import SceneCfg
        p = self._cw_params()
        if self._esdf_mapper is None:
            from g1_classical_manip.motion.collision_world import EsdfMapper
            hw = _np.asarray(depth_mm).shape
            self._esdf_mapper = EsdfMapper(
                grid_center=p["grid_center"], extent_m=p["extent_m"],
                esdf_voxel_size=p["esdf_voxel_size"], tsdf_voxel_size=p["tsdf_voxel_size"],
                image_hw=(int(hw[0]), int(hw[1])),
                depth_min_m=p["depth_min_m"], depth_max_m=p["depth_max_m"])
        ex = None
        if object_mask is not None and p["exclude_object"]:
            from g1_classical_manip.motion.collision_world import dilate_mask
            m = _np.asarray(object_mask, bool)
            if m.shape != _np.asarray(depth_mm).shape:
                print(f"update_grasp_world: object_mask {m.shape} != depth "
                      f"{_np.asarray(depth_mm).shape} -- object NOT excluded")
            elif m.any():
                ex = dilate_mask(m, p["exclude_object_dilate_px"])
                print(f"update_grasp_world: target object CUT from the world "
                      f"({int(m.sum())} px, dilated {p['exclude_object_dilate_px']}px)")
        rf = self.robot_depth_filter(q_repo14, hand_q=hand_q) if p["self_filter"] else None
        grid = self._esdf_mapper.esdf_from_depth(depth_mm, intrinsics, T_pelvis_camera,
                                                 robot_filter=rf, exclude_mask=ex)
        self._grasp_planner(side).update_world(SceneCfg(voxel=[grid]))
        self._cw_loaded.add(side)              # this side now queries the LIVE ESDF, not the build grid
        return True

    def collision_world_points(self):
        """(N,3) pelvis-frame occupied-voxel centres of the current depth-ESDF collision world, or
        None if the world is off / not built yet. For viz -- 09 overlays these (red) on the grasp
        scene so the operator sees the obstacles the approach routes around."""
        return self._esdf_mapper.occupied_points() if self._esdf_mapper is not None else None

    def _hold_idle(self, traj, side: str, hold_q_repo14):
        """Pin the IDLE (non-`side`) arm's 7 joints to `hold_q_repo14` across every waypoint of a
        planned grasp segment (and zero their vel/accel). The single-tool-frame grasp planner only
        constrains the active wrist -- the idle arm is an unconstrained DoF, so trajopt drifts it,
        and the drift accumulates across the approach/grasp/lift solves. Overriding it to the start
        (home) config holds it still. The arms share no joints, so the active-arm trajectory is
        untouched. NOTE: cuRobo collision-checked the active arm against the *drifted* idle arm;
        pinning to home is the safe rest pose for a single-side grasp, but for a cross-body reach
        re-verify (the rigorous alternative is to LOCK the idle joints in the grasp planner so the
        solve is collision-consistent)."""
        if traj is None:
            return traj
        idle = LEFT if side == RIGHT else RIGHT
        cols = slice(0, 7) if idle == LEFT else slice(7, DOF)   # repo order: left 0:7, right 7:14
        hold = np.asarray(hold_q_repo14, float).reshape(DOF)[cols]
        traj.q[:, cols] = hold
        traj.qd[:, cols] = 0.0
        traj.qdd[:, cols] = 0.0
        traj.meta["idle_held"] = True
        return traj

    def plan_grasp_set(self, start_q_repo14, side: str, wrist_goals,
                       approach_axis: str, approach_offset: float,
                       lift_axis: str, lift_offset: float,
                       approach_in_tool_frame: bool = True,
                       lift_in_tool_frame: bool = False,
                       plan_approach: bool = True, plan_lift: bool = True,
                       hold_idle: bool = True,
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
        off along the grasp approach axis, -wrist-Y for our derived transform); lift defaults to
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
            if self._cw_enabled:                          # let the open hand sit in the object ESDF
                disable_collision_links = disable_collision_links + HAND_LINKS[side]
        # LEFT-hand approach mirror: the derived transform maps the grasp approach to wrist +Y on
        # the RIGHT but wrist -Y on the LEFT (tool_transform._MIRROR_Y), so the tool-frame "y"
        # offset must flip sign for the left or the pre-grasp backs INTO the object. The policy
        # lives with the mirror itself (tool_transform.approach_offset_for_side) and is numerically
        # locked by tests/test_tool_transform.py.
        from g1_classical_manip.grasp.tool_transform import approach_offset_for_side
        approach_offset = approach_offset_for_side(
            side, approach_axis, approach_in_tool_frame, approach_offset)
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
            traj = self._jointstate_to_trajectory(js, last, lbl) if js is not None else None
            return self._hold_idle(traj, side, start_q_repo14) if hold_idle else traj

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
                             hold_idle: bool = True,
                             disable_collision_links: Optional[List[str]] = None,
                             max_candidate_retries: int = 12) -> GraspPlanOutcome:
        """Try each strategy dict ({approach_offset, lift_offset?, plan_approach?, plan_lift?}) in
        order; return the first success, else the last failing outcome (its status says why). Mirrors
        GraspGenX's approach/lift offset sweep. `hold_idle` pins the idle arm (see _hold_idle).

        CANDIDATE RETRY: cuRobo's native plan_grasp commits to ONE goalset winner -- Step 1 goalset
        IK picks goal_index[0] (by GRASP-pose IK only), Step 2 plans THAT grasp's approach alone
        (num_goalset=1), and if it fails it returns 'Planning to approach pose failed' WITHOUT trying
        any other candidate (motion_planner.py plan_grasp). So a scene with dozens of feasible grasps
        still fails when the single goalset winner's pre-grasp is blocked -- and MORE SEEDS can't help
        (the winner is chosen fine; only its lone approach fails). We overcome it here: on any failure
        with a valid winner, EXCLUDE that candidate and re-call plan_grasp so cuRobo picks a different
        grasp, up to `max_candidate_retries`.

        LIMITS OF THE RETRY (audit 2026-07-02): it only helps once START/GOAL validity holds --
        Step 2 plans the approach with ALL collision links RE-ENABLED (hand included;
        disable_collision_links applies to Steps 1/3 only), and its graph/IK gate checks the start
        AND goal configs against the world at activation 0. A start config inside the ESDF, or a
        target object fused into the world blocking its own pre-grasp region, fails EVERY retry
        identically. world_check(side, q) tests the exact gate; collision_world.exclude_object cuts
        the target from the world (the end2end reference design). Diagnose the set with
        diagnose_candidates."""
        goals_all = list(wrist_goals)
        if not goals_all:
            raise PlanningError("plan_grasp_set_sweep: no candidate goals")
        last = None
        for s in strategies:
            excluded = set()
            for _ in range(max(1, int(max_candidate_retries))):
                active = [(i, g) for i, g in enumerate(goals_all) if i not in excluded]
                if not active:
                    break
                active_idx = [i for i, _ in active]
                out = self.plan_grasp_set(
                    start_q_repo14, side, [g for _, g in active],
                    approach_axis=approach_axis, approach_offset=s.get("approach_offset", -0.10),
                    lift_axis=lift_axis, lift_offset=s.get("lift_offset", 0.0),
                    approach_in_tool_frame=approach_in_tool_frame,
                    lift_in_tool_frame=lift_in_tool_frame,
                    plan_approach=s.get("plan_approach", True), plan_lift=s.get("plan_lift", True),
                    hold_idle=hold_idle, disable_collision_links=disable_collision_links)
                if out.success:                        # remap chosen_index into the ORIGINAL list
                    if 0 <= out.chosen_index < len(active_idx):
                        out = replace(out, chosen_index=active_idx[out.chosen_index])
                    return out
                last = out
                if not (0 <= out.chosen_index < len(active_idx)):
                    break                              # Step-1 goalset IK failed: no winner to exclude
                orig = active_idx[out.chosen_index]    # the winner whose approach/grasp/lift failed
                excluded.add(orig)
                print(f"plan_grasp: cand {orig} failed ({out.status}); excluding + retrying a "
                      f"different grasp ({len(excluded)}/{len(goals_all)} excluded)")
        return last if last is not None else GraspPlanOutcome(
            False, -1, None, None, None, False, False, False, "no strategies")
