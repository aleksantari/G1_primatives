"""Offline perception tests: the frame-math + filtering of the detectors, with a
fake planner FK so no camera / cuRobo / pupil_apriltags is needed."""
import numpy as np

from g1_classical_manip.spatial.pose import Pose
from g1_classical_manip.perception.transforms import Frames, BODY_TO_OPTICAL
from g1_classical_manip.perception.apriltag_block import AprilTagDetector, _pose_median
from g1_classical_manip.perception.ground_truth import GroundTruthDetector


class FakePlanner:
    """Stands in for CuroboArmPlanner: fk_link returns a fixed camera body pose."""
    def __init__(self, cam_body_pose):
        self._cam = cam_body_pose

    def fk_link(self, link_name, q_repo14=None):
        return self._cam


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


PERC = {
    "tag": {"family": "tag36h11", "size_m": 0.03,
            "tag_to_block": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
            "ids": {0: "block"}},
    "filter": {"median_frames": 5, "max_stale_s": 0.5, "min_decision_margin": 30.0},
}


def _cam_cfg(body_to_optical="identity"):
    return {"extrinsics": {"source": "urdf:d435_link", "body_to_optical": body_to_optical},
            "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}}


def test_pelvis_block_identity_chain():
    # camera body at [0,0,0.4], identity rotation, body_to_optical=identity.
    cam = Pose(np.eye(3), [0.0, 0.0, 0.4])
    frames = Frames(FakePlanner(cam), camera_cfg=_cam_cfg("identity"))
    det = AprilTagDetector(frames, PERC, _cam_cfg("identity"))
    T_cam_tag = det.cam_tag_pose(np.eye(3), [0.1, 0.2, 0.5])
    out = det.pelvis_block_pose(T_cam_tag)
    assert np.allclose(out.translation, [0.1, 0.2, 0.9], atol=1e-9)


def test_body_to_optical_rotation_applied():
    # camera body at origin; body_to_optical=ros rotates optical z -> pelvis x.
    frames = Frames(FakePlanner(Pose()), camera_cfg=_cam_cfg("ros"))
    det = AprilTagDetector(frames, PERC, _cam_cfg("ros"))
    assert np.allclose(frames.T_pelvis_camera().rotation, BODY_TO_OPTICAL.rotation)
    T_cam_tag = det.cam_tag_pose(np.eye(3), [0.0, 0.0, 1.0])  # 1 m along optical z
    out = det.pelvis_block_pose(T_cam_tag)
    assert np.allclose(out.translation, [1.0, 0.0, 0.0], atol=1e-9)


def test_pose_median_picks_median_translation():
    poses = [Pose(np.eye(3), [x, 0.0, 0.5]) for x in [0.0, 0.1, 0.2, 0.3, 0.4]]
    med = _pose_median(poses)
    assert np.isclose(med.translation[0], 0.2)


def test_block_pose_median_and_staleness():
    clock = Clock()
    frames = Frames(FakePlanner(Pose()), camera_cfg=_cam_cfg("identity"))
    det = AprilTagDetector(frames, PERC, _cam_cfg("identity"), clock=clock)
    for x in [0.0, 0.1, 0.2, 0.3, 0.4]:
        det.ingest_detection(0, np.eye(3), [x, 0.0, 0.5], decision_margin=50.0)
    p = det.block_pose("block")
    assert p is not None and np.isclose(p.translation[0], 0.2)
    # advance past max_stale_s -> all readings stale -> None
    clock.t = 1.0
    assert det.block_pose("block") is None


def test_decision_margin_gate():
    frames = Frames(FakePlanner(Pose()), camera_cfg=_cam_cfg("identity"))
    det = AprilTagDetector(frames, PERC, _cam_cfg("identity"), clock=Clock())
    assert det.ingest_detection(0, np.eye(3), [0, 0, 0.5], decision_margin=10.0) is None
    assert det.block_pose("block") is None


def test_ground_truth_detector():
    # no sim base -> world == pelvis; block world pose passes straight through.
    frames = Frames(FakePlanner(Pose()), camera_cfg=_cam_cfg("identity"))
    gt = GroundTruthDetector.from_config(frames, {"xyz": [0.3, 0.0, 0.1]})
    out = gt.detect()
    assert "block" in out and np.allclose(out["block"].pose.translation, [0.3, 0.0, 0.1])
    assert np.allclose(gt.block_pose("block").translation, [0.3, 0.0, 0.1])
    assert gt.block_pose("nope") is None
