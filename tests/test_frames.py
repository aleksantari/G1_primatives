"""perception.frames.Frames with an INJECTED fk_link callable -- the layering fix makes
Frames testable without cuRobo/torch (previously it required the planner object)."""
import numpy as np

from g1_primitives.spatial.pose import Pose
from g1_primitives.perception.frames import Frames, BODY_TO_OPTICAL


def _fk(cam_body_pose):
    """A fake fk_link(link_name, q14) -> constant camera body pose."""
    return lambda link, q14=None: cam_body_pose


def _cam_cfg(body_to_optical="identity"):
    return {"extrinsics": {"source": "urdf:d435_link", "body_to_optical": body_to_optical},
            "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}}


def test_camera_chain_identity():
    # camera body at [0,0,0.4], identity rotation, body_to_optical=identity:
    # an object 0.5 m along optical z lands at pelvis z = 0.9.
    frames = Frames(_fk(Pose(np.eye(3), [0.0, 0.0, 0.4])), camera_cfg=_cam_cfg("identity"))
    out = frames.T_pelvis_from_camera(Pose(np.eye(3), [0.1, 0.2, 0.5]))
    np.testing.assert_allclose(out.translation, [0.1, 0.2, 0.9], atol=1e-9)


def test_body_to_optical_rotation_applied():
    # camera body at origin; body_to_optical=ros rotates optical z -> pelvis x.
    frames = Frames(_fk(Pose()), camera_cfg=_cam_cfg("ros"))
    np.testing.assert_allclose(frames.T_pelvis_camera().rotation, BODY_TO_OPTICAL.rotation)
    out = frames.T_pelvis_from_camera(Pose(np.eye(3), [0.0, 0.0, 1.0]))  # 1 m along optical z
    np.testing.assert_allclose(out.translation, [1.0, 0.0, 0.0], atol=1e-9)


def test_mount_and_correction_compose():
    # correction * FK * mount * body_to_optical, in that exact order.
    cfg = {"extrinsics": {"source": "urdf:d435_link", "body_to_optical": "identity",
                          "mount": {"xyz": [0.0, 0.0, 0.1], "rpy": [0.0, 0.0, 0.0]},
                          "extrinsic_correction": np.eye(4).tolist()}}
    frames = Frames(_fk(Pose(np.eye(3), [0.2, 0.0, 0.0])), camera_cfg=cfg)
    np.testing.assert_allclose(frames.T_pelvis_camera().translation, [0.2, 0.0, 0.1], atol=1e-12)


def test_sim_world_mapping():
    # T_pelvis_from_world uses the configured robot base world pose (inverse applied).
    base = {"xyz": [1.0, 2.0, 0.0], "quat_wxyz": [1, 0, 0, 0]}
    frames = Frames(_fk(Pose()), camera_cfg=_cam_cfg(), sim_base_world_pose=base)
    out = frames.T_pelvis_from_world(Pose(np.eye(3), [1.5, 2.0, 0.3]))
    np.testing.assert_allclose(out.translation, [0.5, 0.0, 0.3], atol=1e-12)
    # unset -> identity passthrough (hardware)
    frames2 = Frames(_fk(Pose()), camera_cfg=_cam_cfg())
    np.testing.assert_allclose(
        frames2.T_pelvis_from_world(Pose(np.eye(3), [0.3, 0.1, 0.2])).translation,
        [0.3, 0.1, 0.2], atol=1e-12)
