"""Robot facade (GPU/DDS-free: Robot(cfg, fake_planner) + fakes/monkeypatch).

Covers the set_* reconfiguration contract (config mutation + component rebuild stay
in lockstep), the camera warm-up, grasp() orchestration, and the curated top-level
exports."""
import pytest

import g1_primitives
from g1_primitives.api.robot import Robot, connect
from g1_primitives.api import robot as robot_mod
from g1_primitives.api import _builders
from g1_primitives.api import primitives as P
from g1_primitives.api.results import GraspResult
from g1_primitives.motion.executor import Executor


# ------------------------------------------------------------- reconfiguration
def test_set_grasp_source_mutates_cfg_and_rebuilds(monkeypatch):
    r = Robot({"grasp": {"source": "graspgenx",
                         "graspgenx": {"port": 5556},
                         "sim_cloud": {"density": 2}}}, None)
    r.frames = "FRAMES"
    sentinel = object()
    seen = {}

    def fake_build(frames, cfg):
        seen["frames"], seen["cfg"] = frames, cfg
        return sentinel

    monkeypatch.setattr(_builders, "build_grasp_source", fake_build)
    r.set_grasp_source("sim_cloud", density=5)
    assert r.grasp_source is sentinel
    assert r.cfg["grasp"]["source"] == "sim_cloud"
    assert r.cfg["grasp"]["sim_cloud"]["density"] == 5       # override merged in place
    assert seen["frames"] == "FRAMES" and seen["cfg"] is r.cfg
    assert r.grasp_source_kind == "sim_cloud"


def test_set_grasp_source_overrides_fall_back_to_graspgenx_block(monkeypatch):
    r = Robot({"grasp": {"graspgenx": {"planner": "diffusion"}}}, None)
    monkeypatch.setattr(_builders, "build_grasp_source", lambda f, c: object())
    r.set_grasp_source("graspgenx", planner="topdown")
    assert r.cfg["grasp"]["graspgenx"]["planner"] == "topdown"


def test_set_segmenter_rebuilds_source(monkeypatch):
    r = Robot({"grasp": {"segment": {"mode": None, "port": 5557}}}, None)
    monkeypatch.setattr(_builders, "build_grasp_source", lambda f, c: "REBUILT")
    r.set_segmenter("auto", top_k=2)
    assert r.cfg["grasp"]["segment"]["mode"] == "auto"
    assert r.cfg["grasp"]["segment"]["top_k"] == 2
    assert r.cfg["grasp"]["segment"]["port"] == 5557         # existing keys survive
    assert r.grasp_source == "REBUILT"


def test_set_visualize(monkeypatch):
    r = Robot({"grasp": {"graspgenx": {}}}, None)
    monkeypatch.setattr(_builders, "build_grasp_source", lambda f, c: "V")
    r.set_visualize(True, port=8081)
    v = r.cfg["grasp"]["graspgenx"]["visualize"]
    assert v["enabled"] is True and v["port"] == 8081
    assert r.grasp_source == "V"


def test_set_executor_maps_attrs():
    r = Robot({}, None)
    r.executor = Executor(arm_controller=object(), planner=None)
    r.set_executor(speed=0.5, abort_thresh_rad=0.33, gravity_scale=0.8)
    assert r.executor.time_dilation == 0.5
    assert r.executor.abort_thresh == 0.33
    assert r.executor.gravity_scale == 0.8
    # gravity-comp needs the planner's RNEA -- guarded off when there is none
    r.set_executor(gravity_comp=True)
    assert r.executor.gravity_comp is False


def test_set_executor_offline_raises():
    r = Robot({}, None)
    with pytest.raises(RuntimeError, match="offline"):
        r.set_executor(speed=0.5)


def test_set_collision_world_applies_to_planner():
    class FakePlannerCW:
        def __init__(self):
            self.calls = []

        def set_collision_world(self, enabled, cfg=None):
            self.calls.append((enabled, cfg))

    r = Robot({"planner": {"grasp": {"collision_world": {"enabled": False,
                                                         "voxel_size": 0.01}}}},
              FakePlannerCW())
    r.set_collision_world(True, exclude_object_dilate_px=8)
    cw = r.cfg["planner"]["grasp"]["collision_world"]
    assert cw["enabled"] is True
    assert cw["voxel_size"] == 0.01                          # existing keys survive
    assert cw["exclude_object_dilate_px"] == 8
    assert r.planner.calls == [(True, cw)]


def test_set_grasp_strategies():
    r = Robot({"planner": {}}, None)
    r.set_grasp_strategies([{"approach_offset": -0.08, "lift_offset": 0.1}])
    assert r.cfg["planner"]["grasp"]["strategies"] == [
        {"approach_offset": -0.08, "lift_offset": 0.1}]


# ------------------------------------------------------------------ perception
class FakeCam:
    """Yields None for the first N polls of each stream, then a frame."""

    def __init__(self, rgb_after=0, depth_after=None):
        self._rgb_left = rgb_after
        self._depth_left = depth_after       # None => stream has no depth

    @property
    def has_depth(self):
        return self._depth_left is not None

    def get_rgb_frame(self):
        if self._rgb_left > 0:
            self._rgb_left -= 1
            return None
        return "RGB"

    def get_depth_frame(self):
        if self._depth_left is None:
            return None
        if self._depth_left > 0:
            self._depth_left -= 1
            return None
        return "DEPTH"


def test_wait_for_frames_rgb_only():
    r = Robot({}, None)
    r.camera = FakeCam(rgb_after=2)
    assert r.wait_for_frames(rgb=True, depth=False, timeout_s=2.0)


def test_wait_for_frames_rgb_and_depth():
    r = Robot({}, None)
    r.camera = FakeCam(rgb_after=1, depth_after=2)
    assert r.wait_for_frames(rgb=True, depth=True, timeout_s=2.0)


def test_wait_for_frames_depth_missing_times_out():
    r = Robot({}, None)
    r.camera = FakeCam(rgb_after=0, depth_after=None)        # no depth stream
    assert not r.wait_for_frames(rgb=True, depth=True, timeout_s=0.2)


def test_wait_for_frames_no_camera():
    r = Robot({}, None)
    assert not r.wait_for_frames()


# ---------------------------------------------------------------------- grasp
class FakeSource:
    def __init__(self, cands):
        self._cands = cands
        self.calls = []

    def grasps(self, robot, side, target):
        self.calls.append((side, target))
        return list(self._cands)


def test_grasp_no_candidates_names_the_source():
    r = Robot({"grasp": {"source": "sim_cloud"}}, None)
    r.grasp_source = FakeSource([])
    res = r.grasp("right", "block")
    assert not res
    assert "sim_cloud" in res.info and "block" in res.info
    assert res.chosen_index == -1


def test_grasp_delegates_to_grasp_motion(monkeypatch):
    r = Robot({"grasp": {"source": "sim_cloud"}}, None)
    r.grasp_source = FakeSource(["c1", "c2"])
    seen = {}

    def fake_grasp_motion(robot, side, candidates, options=None):
        seen["args"] = (robot, side, list(candidates), options)
        return GraspResult(True, "ok")

    monkeypatch.setattr(P, "grasp_motion", fake_grasp_motion)
    res = r.grasp("left", "cube")
    assert res
    robot_arg, side, cands, options = seen["args"]
    assert robot_arg is r and side == "left" and cands == ["c1", "c2"]
    assert options is None
    assert r.grasp_source.calls == [("left", "cube")]


def test_grasp_candidates_warms_depth_only_for_graspgenx(monkeypatch):
    r = Robot({"grasp": {"source": "graspgenx"}}, None)
    r.grasp_source = FakeSource(["c"])
    r.camera = FakeCam()
    seen = {}

    def fake_wait(rgb=True, depth=False, timeout_s=6.0):
        seen["depth"] = depth
        return True

    monkeypatch.setattr(r, "wait_for_frames", fake_wait)
    r.grasp_candidates("right", "block")
    assert seen["depth"] is True                             # graspgenx needs depth

    r.cfg["grasp"]["source"] = "sim_cloud"
    r.grasp_candidates("right", "block")
    assert seen["depth"] is False                            # sim_cloud does not


# -------------------------------------------------------------------- exports
def test_curated_exports():
    assert g1_primitives.connect is connect
    assert g1_primitives.Robot is Robot
    assert connect.__module__ == robot_mod.__name__
    for name in g1_primitives.__all__:
        assert getattr(g1_primitives, name, None) is not None, name


def test_connect_rejects_unknown_target():
    with pytest.raises(ValueError, match="unknown target"):
        Robot.connect("robot9000")
    with pytest.raises(ValueError, match="unknown target"):
        Robot.offline("robot9000")
