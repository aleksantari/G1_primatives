"""Dex3 hand FK (wrist_yaw -> fingertips), parsed from the committed URDF."""
import numpy as np

from g1_classical_manip.ee.hand_base import LEFT, RIGHT
from g1_classical_manip.ee.hand_kinematics import Dex3Kinematics

# configs/hands.yaml dex3.presets.power_close (RIGHT-hand order: th0,th1,th2, idx0,idx1, mid0,mid1)
POWER_CLOSE_RIGHT = [0.0, -0.85, -1.40, 1.30, 1.40, 1.30, 1.40]


def test_right_hand_open_geometry():
    kin = Dex3Kinematics()
    tips = kin.fingertips(RIGHT, [0.0] * 7)                     # open / extended
    assert tips["index"][0] > 0.1 and tips["middle"][0] > 0.1   # fingers reach forward (wrist +x)
    assert tips["index"][2] > 0.02 and tips["middle"][2] < -0.02  # index/middle split along ±z
    assert tips["thumb"][1] > 0.0                               # thumb opposes from the +y side


def test_contact_point_matches_derivation():
    # contact midpoint at power_close == the derived graspgenx anchor + the GraspGenX depth
    # along approach: t = [0.0442, 0.0414, 0] and approach offset [0.07,0,0] -> c = [0.1142,...].
    kin = Dex3Kinematics()
    c = kin.contact_point(RIGHT, POWER_CLOSE_RIGHT)
    np.testing.assert_allclose(c, [0.1142, 0.0414, 0.0], atol=2e-3)


def test_left_hand_mirrors_right():
    kin = Dex3Kinematics()
    cr = kin.contact_point(RIGHT, POWER_CLOSE_RIGHT)
    # power_close LEFT is the sign-mirror of RIGHT (configs/hands.yaml)
    cl = kin.contact_point(LEFT, [0.0, 0.85, 1.40, -1.30, -1.40, -1.30, -1.40])
    np.testing.assert_allclose(cl, [cr[0], -cr[1], cr[2]], atol=1e-3)   # mirror across wrist y
