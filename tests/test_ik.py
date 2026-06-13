import numpy as np
import pinocchio as pin


def test_reduced_model_is_14dof(ik):
    assert ik.reduced_robot.model.nq == 14


def test_ik_submillimeter(ik):
    L = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.25, 0.25, 0.1]))
    R = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.25, -0.25, 0.1]))
    q, tau = ik.solve_ik(L.homogeneous, R.homogeneous)
    TL, TR = ik.fk(q)
    assert np.linalg.norm(TL.translation - L.translation) < 1e-3
    assert np.linalg.norm(TR.translation - R.translation) < 1e-3
    assert tau.shape == (14,)


def test_fk_roundtrip(ik, home_q):
    TL, TR = ik.fk(home_q)
    q, _ = ik.solve_ik(TL.homogeneous, TR.homogeneous, home_q)
    TL2, TR2 = ik.fk(q)
    assert np.linalg.norm(TL2.translation - TL.translation) < 1e-3
