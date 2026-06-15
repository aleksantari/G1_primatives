import numpy as np
import pinocchio as pin

from g1_classical_manip.perception.ground_truth import GroundTruthBlockSource
from g1_classical_manip.perception.transforms import Frames, from_xyz_rpy
from g1_classical_manip.tasks import primitives as P


def test_t_pelvis_from_world_block(robot):
    # robot.yaml sim.base_world_pose + the configured block world pose must map to
    # the validated pelvis-frame block location.
    gt = GroundTruthBlockSource.from_config(
        robot.frames, robot.cfg["pick_place"]["ground_truth_block_world"])
    t = gt.block_pose(0).translation
    assert np.allclose(t, [0.33, -0.05, 0.08], atol=1e-3)


def test_t_pelvis_from_world_identity_default(ik):
    # No sim base pose -> world == pelvis (pure passthrough).
    F = Frames(reduced_robot=ik.reduced_robot)
    T = from_xyz_rpy([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert np.allclose(F.T_pelvis_from_world(T).homogeneous, T.homogeneous)


def test_ground_truth_block_pose_constant_ignores_tag(robot):
    gt = GroundTruthBlockSource(robot.frames, [-4.25, -4.03, 0.84], [1, 0, 0, 0])
    assert np.allclose(gt.block_pose(0).homogeneous, gt.block_pose(7).homogeneous)


def test_grasp_aligned_decoupled_from_block_yaw(robot):
    # home_aligned grasp target must NOT inherit the block's yaw (the singular case).
    side = "right"
    R_palm = robot.frames.T_palm(side, P.home_q(robot)).rotation
    pos = np.array([0.33, -0.05, 0.08])
    blk0 = pin.SE3(np.eye(3), pos)
    blk90 = pin.SE3(pin.rpy.rpyToMatrix(0, 0, np.pi / 2), pos)
    t0 = robot.frames.grasp_ee_target_aligned(blk0, side, R_palm, 0.1)
    t90 = robot.frames.grasp_ee_target_aligned(blk90, side, R_palm, 0.1)
    assert np.allclose(t0.homogeneous, t90.homogeneous)


def test_grasp_target_approach_modes_differ(robot):
    cfg = robot.cfg["pick_place"]["grasp"]
    blk = pin.SE3(np.eye(3), np.array([0.33, -0.05, 0.08]))
    old = cfg.get("approach")
    try:
        cfg["approach"] = "home_aligned"
        ha = P.grasp_target(robot, blk, "right", 0.1)
        cfg["approach"] = "top_down"
        td = P.grasp_target(robot, blk, "right", 0.1)
    finally:
        cfg["approach"] = old
    assert not np.allclose(ha.homogeneous, td.homogeneous)
