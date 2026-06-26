"""Client-side grasp visualization (viser).

Renders the GraspGenX point cloud + ranked grasps + gripper mesh in a viser web
GUI, matching the GraspGenX demos -- but driven by data the client already has
(pelvis-frame cloud, ranked candidates, the cuRobo-chosen grasp). Vendors the
viser primitives + a torch-free gripper-geometry loader so it imports NOTHING
from the ``graspgenx`` package (which would pull torch + a multi-GB auto-download).

``viser`` is imported lazily (only when ``GraspViz`` is accessed), so importing
this package is cheap and non-viz runs never touch viser. The loader
``load_gripper_geom`` needs only trimesh.
"""
from g1_classical_manip.viz.gripper_geom import GripperGeom, load_gripper_geom

__all__ = ["GripperGeom", "load_gripper_geom", "GraspViz"]


def __getattr__(name):
    # Lazy: defer importing grasp_viz (-> viser) until GraspViz is actually used.
    if name == "GraspViz":
        from g1_classical_manip.viz.grasp_viz import GraspViz
        return GraspViz
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
