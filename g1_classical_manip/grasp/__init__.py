"""Grasp-pose sources — the seam between perception and motion.

A ``GraspSource`` turns a target object into a RANKED list of wrist-yaw goal ``Pose``s
(``GraspCandidate``), exactly what the primitives / the cuRobo planner already consume.
Implementations behind the one interface:

  * ``GraspGenXGraspSource`` — depth -> segmented cloud -> the GraspGenX ZMQ service ->
    ranked 6-DoF grasps mapped through the fixed grasp->tool transform.
  * ``SimCloudGraspSource`` — SIM only: ground-truth cube cloud from ``rt/sim_state`` ->
    the same GraspGenX + tool-transform path (no camera/depth/SAM3).
"""
