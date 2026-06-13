# Adapted from unitreerobotics/xr_teleoperate (Apache-2.0) and the user's
# G1_teleop_dex fork: teleop/robot_control/robot_arm_ik.py -> G1_29_ArmIK.
# See NOTICE for attribution.
#
# Changes for g1_classical_manip:
#   * Single G1_29_ArmIK class (G1_23/H1/H1_2 variants dropped).
#   * Default model = assets/g1/g1_body29_dex3.urdf (full Dex3 + head d435 +
#     palm frames). Lock list = legs + 3 waist + 14 Dex3 hand joints -> 14-DoF
#     dual-arm reduced model. EE frames L_ee/R_ee at wrist_yaw + [0.05,0,0].
#     The reduced model RETAINS the d435_link / *_hand_palm_link frames (as
#     fixed frames on their parent bodies) -- perception/transforms.py uses them.
#   * Model build factored into the free function load_g1_reduced(), shared by
#     the IK and transforms.py. Reduced model cached under
#     ~/.cache/g1_classical_manip (rebuilt if the URDF is newer than the cache).
#   * Cross-call WeightedMovingFilter OFF by default (raw per-waypoint IK for
#     planning); pass smooth=True for the teleop streaming behaviour.
#   * Added fk(q14) -> (pin.SE3, pin.SE3) EE forward kinematics.
#
# Targets are 4x4 homogeneous transforms of the L_ee / R_ee frames in the pelvis
# (model root) frame. The robot is suspended with the waist locked, so pelvis ==
# world up to the mount.

import os
import pickle
import hashlib

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

try:
    import logging_mp

    logger_mp = logging_mp.getLogger(__name__)
except Exception:
    import logging

    logger_mp = logging.getLogger(__name__)

from g1_classical_manip.utils.weighted_moving_filter import WeightedMovingFilter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
ASSETS_G1_DIR = os.path.join(_REPO_ROOT, "assets", "g1")
DEFAULT_URDF = os.path.join(ASSETS_G1_DIR, "g1_body29_dex3.urdf")
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "g1_classical_manip")

# Joints locked to reduce the 29-body + 14-hand model to the 14 arm DoF.
LOCKED_JOINTS = [
    # legs
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # waist
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # left Dex3 hand
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    # right Dex3 hand
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]

EE_OFFSET_X = 0.05  # L_ee/R_ee sit 0.05 m along +x of the wrist-yaw joint.


def _cache_path(urdf_path, locked_reference, add_ee):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tag = hashlib.md5(
        (urdf_path + str(np.asarray(locked_reference).tolist()) + str(add_ee))
        .encode()).hexdigest()[:8]
    stem = os.path.splitext(os.path.basename(urdf_path))[0]
    return os.path.join(_CACHE_DIR, f"{stem}_{tag}_reduced.pkl")


def load_g1_reduced(urdf_path=DEFAULT_URDF, locked_reference=None,
                    add_ee=True, use_cache=True, verbose=False):
    """Build (or load) the 14-DoF dual-arm reduced model from a G1 URDF.

    Locks legs + waist + Dex3 hand joints at ``locked_reference`` (full-model
    config, default zeros). Optionally adds L_ee/R_ee operational frames. The
    returned reduced model also retains the URDF's fixed frames (d435_link,
    *_hand_palm_link, torso_link, ...). Returns a pin.RobotWrapper.
    """
    model_dir = os.path.dirname(urdf_path)
    cache = _cache_path(urdf_path, locked_reference, add_ee)
    if use_cache and os.path.exists(cache):
        try:
            if os.path.getmtime(cache) >= os.path.getmtime(urdf_path):
                with open(cache, "rb") as f:
                    model = pickle.load(f)["reduced_model"]
                rr = pin.RobotWrapper(model=model)
                rr.data = rr.model.createData()
                if verbose:
                    logger_mp.info(f"[g1_model] loaded cache {cache}")
                return rr
        except Exception as e:
            logger_mp.warning(f"[g1_model] cache load failed ({e}); rebuilding")

    if verbose:
        logger_mp.info(f"[g1_model] building reduced model from {urdf_path}")
    robot = pin.RobotWrapper.BuildFromURDF(urdf_path, model_dir)
    ref = (np.asarray(locked_reference, dtype=float)
           if locked_reference is not None else np.zeros(robot.model.nq))
    lock = [robot.model.getJointId(j)
            for j in LOCKED_JOINTS if robot.model.existJointName(j)]
    reduced = robot.buildReducedRobot(list_of_joints_to_lock=lock,
                                      reference_configuration=ref)
    if reduced.model.nq != 14:
        raise RuntimeError(
            f"reduced model has {reduced.model.nq} DoF, expected 14 "
            f"(URDF={urdf_path}); check the lock list")
    if add_ee:
        for name, joint in (("L_ee", "left_wrist_yaw_joint"),
                            ("R_ee", "right_wrist_yaw_joint")):
            if not reduced.model.existFrame(name):
                reduced.model.addFrame(pin.Frame(
                    name, reduced.model.getJointId(joint),
                    pin.SE3(np.eye(3), np.array([EE_OFFSET_X, 0.0, 0.0])),
                    pin.FrameType.OP_FRAME))
        reduced.data = reduced.model.createData()
    try:
        with open(cache, "wb") as f:
            pickle.dump({"reduced_model": reduced.model}, f)
    except Exception as e:
        logger_mp.warning(f"[g1_model] cache save failed: {e}")
    return reduced


class G1_29_ArmIK:
    """Weighted dual-arm IK (Pinocchio + CasADi + Ipopt), warm-started.

    cost = 200*translation + 1*rotation + 0.001*||q|| + 0.01*smoothness
    """

    def __init__(self, urdf_path=None, Visualization=False, smooth=False,
                 locked_reference=None, verbose=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.urdf_path = urdf_path or DEFAULT_URDF
        self.Visualization = Visualization
        self.verbose = verbose

        self.reduced_robot = load_g1_reduced(
            self.urdf_path, locked_reference=locked_reference, add_ee=True,
            use_cache=not Visualization, verbose=verbose)

        self.nq = self.reduced_robot.model.nq
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self._build_opti()

        self.init_data = np.zeros(self.nq)
        self.smooth_filter = (
            WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), self.nq)
            if smooth else None)
        self.vis = None
        if self.Visualization:
            self._init_viz()

    # ------------------------------------------------------------------- opti
    def _build_opti(self):
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.translational_error = casadi.Function(
            "translational_error", [self.cq, self.cTf_l, self.cTf_r],
            [casadi.vertcat(
                self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3])])
        self.rotational_error = casadi.Function(
            "rotational_error", [self.cq, self.cTf_l, self.cTf_r],
            [casadi.vertcat(
                cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T))])

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.nq)
        self.var_q_last = self.opti.parameter(self.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        regularization_cost = casadi.sumsqr(self.var_q)
        smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit))
        self.opti.minimize(200 * translational_cost + rotation_cost
                           + 0.001 * regularization_cost + 0.01 * smooth_cost)
        self.opti.solver("ipopt", {
            "expand": True, "detect_simple_bounds": True, "calc_lam_p": False,
            "print_time": False, "ipopt.sb": "yes", "ipopt.print_level": 0,
            "ipopt.max_iter": 100, "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-5,
            "ipopt.acceptable_iter": 5, "ipopt.warm_start_init_point": "yes",
            "ipopt.derivative_test": "none", "ipopt.jacobian_approximation": "exact",
        })

    # -------------------------------------------------------------------- API
    def fk(self, q14):
        """EE forward kinematics. Returns (T_L, T_R) as pin.SE3 in pelvis frame."""
        q14 = np.asarray(q14, dtype=float)
        pin.framesForwardKinematics(self.reduced_robot.model,
                                    self.reduced_robot.data, q14)
        return (self.reduced_robot.data.oMf[self.L_hand_id].copy(),
                self.reduced_robot.data.oMf[self.R_hand_id].copy())

    def gravity_torque(self, q14):
        m, d = self.reduced_robot.model, self.reduced_robot.data
        return pin.rnea(m, d, np.asarray(q14, float), np.zeros(m.nv), np.zeros(m.nv))

    def solve_ik(self, left_wrist, right_wrist,
                 current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        """Solve dual-arm IK. left_wrist/right_wrist are 4x4 homogeneous EE
        targets in the pelvis frame. Returns (sol_q14, sol_tauff14)."""
        left_wrist = np.asarray(left_wrist, dtype=float)
        right_wrist = np.asarray(right_wrist, dtype=float)
        if current_lr_arm_motor_q is not None:
            self.init_data = np.asarray(current_lr_arm_motor_q, dtype=float)
        self.opti.set_initial(self.var_q, self.init_data)
        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(left_wrist)
            self.vis.viewer["R_ee_target"].set_transform(right_wrist)
        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            self.opti.solve()
            sol_q = self.opti.value(self.var_q)
        except Exception as e:
            logger_mp.error(f"[G1_29_ArmIK] IK did not converge: {e}")
            sol_q = self.opti.debug.value(self.var_q)

        if self.smooth_filter is not None:
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data
        self.init_data = sol_q
        sol_tauff = self.gravity_torque(sol_q)
        if self.Visualization:
            self.vis.display(sol_q)
        return sol_q, sol_tauff

    # -------------------------------------------------------------------- viz
    def _init_viz(self):
        import meshcat.geometry as mg
        from pinocchio.visualize import MeshcatVisualizer
        self.vis = MeshcatVisualizer(self.reduced_robot.model,
                                     self.reduced_robot.collision_model,
                                     self.reduced_robot.visual_model)
        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")
        self.vis.display(pin.neutral(self.reduced_robot.model))
        axis = (np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1]]).astype(np.float32).T)
        colors = (np.array([[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0],
                            [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T)
        for name in ("L_ee_target", "R_ee_target"):
            self.vis.viewer[name].set_object(mg.LineSegments(
                mg.PointsGeometry(position=0.1 * axis, color=colors),
                mg.LineBasicMaterial(linewidth=20, vertexColors=True)))


if __name__ == "__main__":
    ik = G1_29_ArmIK(verbose=True)
    L = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.25, 0.25, 0.1]))
    R = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.25, -0.25, 0.1]))
    q, tau = ik.solve_ik(L.homogeneous, R.homogeneous)
    TL, TR = ik.fk(q)
    print("L translation err (m):", np.linalg.norm(TL.translation - L.translation))
    print("R translation err (m):", np.linalg.norm(TR.translation - R.translation))
