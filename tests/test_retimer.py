import numpy as np

from g1_classical_manip.motion.planner_base import JointPath
from g1_classical_manip.motion.retimer import retime


def _straight_path(n=50):
    q0 = np.zeros(14)
    q1 = np.full(14, 0.3)
    q = np.linspace(q0, q1, n)
    return JointPath(q, meta={"waypoint_index": [(n - 1, "goal")]})


def test_limits_respected_straight():
    traj = retime(_straight_path(), control_hz=250, max_velocity=2.0,
                  max_acceleration=5.0, max_jerk=30.0)
    assert traj.meta["max_qd"] <= 2.0 * 1.05
    assert traj.meta["max_qdd"] <= 5.0 * 1.20
    assert np.all(np.diff(traj.t) > 0)            # monotonic time
    assert np.allclose(traj.qd[0], 0) and np.allclose(traj.qd[-1], 0)  # rest-to-rest


def test_degenerate_single_point():
    q = np.zeros((2, 14))                          # start == goal
    traj = retime(JointPath(q, meta={"waypoint_index": [(1, "g")]}))
    assert traj.q.shape[0] == 1
    assert traj.duration == 0.0


def test_curved_path_is_slowed():
    # a sharp two-segment path; check the retimer keeps accel within limits
    a = np.linspace(np.zeros(14), np.full(14, 0.3), 30)
    b = np.linspace(np.full(14, 0.3), np.full(14, -0.1), 30)[1:]
    q = np.vstack([a, b])
    path = JointPath(q, meta={"waypoint_index": [(29, "mid"), (len(q) - 1, "goal")]})
    traj = retime(path, max_velocity=2.0, max_acceleration=5.0, max_jerk=30.0)
    assert traj.meta["max_qdd"] <= 5.0 * 1.20
