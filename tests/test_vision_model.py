import numpy as np
import pinocchio as pin

from g1_classical_manip.perception.apriltag_block import (
    AprilTagVisionModel, TagReading)


def _model(robot):
    cam = dict(robot.cfg["camera"])
    cam["intrinsics"] = {**cam["intrinsics"], "fx": 600, "fy": 600,
                         "cx": 320, "cy": 240}
    return AprilTagVisionModel(robot.frames, robot.cfg["perception"], cam,
                               spec={"camera": "head"})


def test_cameras_declared(robot):
    assert _model(robot).cameras() == ["head"]


def test_maps_readings_to_pose_estimates(robot, monkeypatch):
    m = _model(robot)
    reading = TagReading(0, pin.SE3.Identity(), 42.0, 0.0)
    monkeypatch.setattr(m.det, "detect", lambda img, q14=None: {0: reading})
    out = m.process({"head": np.zeros((4, 4, 3), np.uint8)})
    assert out.model == "apriltag"
    assert len(out.poses) == 1
    p = out.poses[0]
    assert p.label == "block"            # id 0 -> "block" (perception.yaml ids)
    assert p.score == 42.0
    assert p.source_cam == "head"
    assert p.extra["tag_id"] == 0
    assert out.best_pose().score == 42.0
    assert out.source_cams == ["head"]


def test_empty_when_no_head_frame(robot):
    out = _model(robot).process({})      # no head frame in the dict
    assert out.poses == []
    assert out.model == "apriltag"


def test_block_pose_delegates(robot):
    m = _model(robot)
    for _ in range(m.median_frames):
        m.det.ingest_detection(0, np.eye(3), np.array([0.3, 0.0, 0.1]), 50.0)
    assert m.block_pose(0) is not None
    assert m.detector is m.det.detector
