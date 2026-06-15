import numpy as np

from g1_classical_manip.perception.observation import build_observation


class _StubCam:
    def get_rgb_frames(self):
        return {"head": np.zeros((480, 640, 3), np.uint8),
                "left_wrist": np.zeros((480, 640, 3), np.uint8)}


class _StubArm:
    def get_current_dual_arm_q(self):
        return np.arange(14, dtype=float)


def test_observation_schema():
    obs = build_observation(_StubCam(), _StubArm())
    assert obs["observation.images.head"].shape == (480, 640, 3)
    assert obs["observation.images.head"].dtype == np.uint8
    assert "observation.images.left_wrist" in obs
    state = obs["observation.state"]
    assert state.shape == (14,)
    assert state.dtype == np.float32
    assert np.allclose(state, np.arange(14))


def test_observation_no_camera():
    obs = build_observation(None, _StubArm())
    assert obs["observation.state"].shape == (14,)
    assert not any(k.startswith("observation.images") for k in obs)


def test_observation_ee_state_appends():
    class _Hand:
        class ctrl:
            @staticmethod
            def get_state(side):
                return {"q": np.zeros(7)}
    obs = build_observation(None, _StubArm(), hand=_Hand(),
                            include_ee_state=True)
    assert obs["observation.state"].shape == (28,)   # 14 arm + 7 + 7 hand
