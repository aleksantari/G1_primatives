import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session")
def ik():
    from g1_classical_manip.robot_control.robot_arm_ik import G1_29_ArmIK
    return G1_29_ArmIK()


@pytest.fixture(scope="session")
def robot():
    from g1_classical_manip.factory import make_robot
    return make_robot(connect_dds=False, build_perception=True)


@pytest.fixture
def home_q():
    return np.deg2rad([15, 10, 0, 30, 0, 0, 0, 15, -10, 0, 30, 0, 0, 0])
