"""g1_primitives -- cuRobo-native action + perception primitives for the Unitree G1.

The intended surface for an agent host:

    import g1_primitives as g1

    robot = g1.connect("sim")                       # or "real"
    robot.home()
    result = robot.grasp("right", "block",
                         options=g1.GraspOptions())  # defaults = full autonomous pick
    if not result:
        print(result.info, result.report)

Everything re-exported here is import-light (no torch/CUDA/DDS at import time --
enforced by tests/test_import_hygiene.py); the heavy stacks load on connect().
"""
from g1_primitives.api.robot import Robot, connect
from g1_primitives.api.options import GraspOptions, GraspObserver
from g1_primitives.api.results import Result, GraspResult, GraspReport
from g1_primitives.perception.base import Detection
from g1_primitives.spatial.pose import Pose
from g1_primitives.spatial.pointcloud import PointCloud
from g1_primitives.ee.hand_base import LEFT, RIGHT

__version__ = "0.2.0"

__all__ = [
    "connect", "Robot",
    "GraspOptions", "GraspObserver",
    "Result", "GraspResult", "GraspReport",
    "Detection", "Pose", "PointCloud",
    "LEFT", "RIGHT",
    "__version__",
]
