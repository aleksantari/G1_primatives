import numpy as np

from g1_classical_manip.perception.hub import PerceptionHub


def _hub(robot):
    return PerceptionHub(robot.frames, robot.cfg["perception"],
                         cameras_cfg=robot.cfg["cameras"],
                         camera_cfg=robot.cfg["camera"])


def test_builds_configured_models(robot):
    hub = _hub(robot)
    assert "apriltag" in hub.models
    assert hub.required_cameras() == {"head"}


def test_process_populates_latest(robot, monkeypatch):
    hub = _hub(robot)
    monkeypatch.setattr(hub.models["apriltag"].det, "detect",
                        lambda img, q14=None: {})
    assert hub.latest("apriltag") is None
    hub.process({"head": np.zeros((4, 4, 3), np.uint8)})
    out = hub.latest("apriltag")
    assert out is not None and out.model == "apriltag"


def test_delegates_apriltag_surface(robot):
    hub = _hub(robot)
    am = hub.models["apriltag"]
    assert hub.median_frames == am.median_frames
    assert hub.detector is am.detector
    for _ in range(hub.median_frames):
        am.det.ingest_detection(0, np.eye(3), np.array([0.3, 0.0, 0.1]), 50.0)
    assert hub.block_pose(0) is not None


def test_min_period_throttles(robot, monkeypatch):
    # a model with min_period_s should not re-run within the window
    clock = [0.0]
    cfg = {**robot.cfg["perception"],
           "models": [{"name": "apriltag", "camera": "head",
                       "min_period_s": 10.0}]}
    hub = PerceptionHub(robot.frames, cfg, cameras_cfg=robot.cfg["cameras"],
                        camera_cfg=robot.cfg["camera"], clock=lambda: clock[0])
    calls = [0]

    def fake_detect(img, q14=None):
        calls[0] += 1
        return {}
    monkeypatch.setattr(hub.models["apriltag"].det, "detect", fake_detect)
    frame = {"head": np.zeros((4, 4, 3), np.uint8)}
    hub.process(frame)          # t=0 runs
    hub.process(frame)          # t=0 within window -> skipped
    assert calls[0] == 1
    clock[0] = 11.0
    hub.process(frame)          # window elapsed -> runs again
    assert calls[0] == 2
