import numpy as np

from g1_classical_manip.perception.transforms import from_xyz_rpy
from g1_classical_manip.perception.apriltag_block import AprilTagBlockDetector


def _detector(robot, clock):
    perc = dict(robot.cfg["perception"])
    cam = dict(robot.cfg["camera"])
    cam["intrinsics"] = {**cam["intrinsics"], "fx": 600, "fy": 600, "cx": 320, "cy": 240}
    return AprilTagBlockDetector(robot.frames, perc, cam, clock=clock)


def test_median_block_recovery(robot):
    t = [0.0]
    det = _detector(robot, lambda: t[0])
    gt = from_xyz_rpy([0.36, 0.04, 0.06], [0, 0, 0.2])
    T_pelvis_tag = gt * det.tag_to_block.inverse()
    T_cam_tag = robot.frames.T_pelvis_camera().inverse() * T_pelvis_tag
    rng = np.random.default_rng(0)
    for i in range(6):
        t[0] = i * 0.05
        det.ingest_detection(0, T_cam_tag.rotation,
                             T_cam_tag.translation + rng.normal(0, 0.002, 3), 50.0)
    rec = det.block_pose(0)
    assert rec is not None
    assert np.linalg.norm(rec.translation - gt.translation) < 0.005


def test_low_margin_rejected(robot):
    det = _detector(robot, lambda: 0.0)
    assert det.ingest_detection(0, np.eye(3), np.zeros(3), decision_margin=5.0) is None


def test_staleness_rejected(robot):
    t = [0.0]
    det = _detector(robot, lambda: t[0])
    det.ingest_detection(0, np.eye(3), np.array([0, 0, 0.5]), 50.0)
    t[0] = 100.0
    assert det.block_pose(0) is None
