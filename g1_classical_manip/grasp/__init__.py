"""Grasp-pose sources — the seam between perception and motion.

A ``GraspSource`` turns a target object into a RANKED list of wrist-yaw goal ``Pose``s
(``GraspCandidate``), exactly what ``primitives.move`` / the cuRobo planner already
consume. Two implementations sit behind the one interface:

  * ``AprilTagGraspSource`` — the known-good A-B reference (one candidate from the
    detected pose + the URDF palm offset; mirrors ``07_pick_place``).
  * ``GraspGenXGraspSource`` — depth -> segmented cloud -> the GraspGenX ZMQ service ->
    ranked 6-DoF grasps mapped through the fixed grasp->tool transform.
"""
