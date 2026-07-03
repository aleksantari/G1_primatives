"""GraspGenX grasp source: depth -> segmented cloud -> GraspGenX ZMQ -> ranked grasps.

Pipeline (all pelvis-frame, meters): capture the rgb+depth pair, get a 2D object mask from
the segmenter (SAM3 / interactive / none), deproject ONLY the masked-in depth pixels into a
single-object cloud, send it to the GraspGenX service, map each returned 6-DoF grasp through
the fixed grasp->tool transform to a wrist-yaw goal, return ranked by confidence. The service
centers/uncenters internally, so the cloud is sent in the pelvis frame and grasps come back in
the pelvis frame (no client centering / normals / table plane).
"""
from __future__ import annotations

from typing import List

import numpy as np

from g1_primitives.spatial.pose import rpy_to_matrix
from g1_primitives.perception.depth import deproject_depth
from g1_primitives.perception.segment import SegmentationAborted
from g1_primitives.grasp.base import GraspSource, GraspCandidate, SourceSnapshot
from g1_primitives.grasp.tool_transform import candidates_from_grasps
from g1_primitives.latency import LOG   # latency instrumentation (no-op unless enabled)


class GraspGenXGraspSource(GraspSource):
    def __init__(self, frames, segmenter, client_factory, gcfg: dict, camera_cfg: dict,
                 viz=None):
        self.frames = frames
        self.segmenter = segmenter                     # perception.segment.Segmenter (.mask(rgb))
        self.client_factory = client_factory           # () -> GraspGenXClient (ctx manager)
        self.gcfg = gcfg
        self.intrinsics = (camera_cfg or {}).get("intrinsics", {})
        self.palm_offset_xyz = np.asarray(gcfg["palm_offset_xyz"], float)
        self.R_wristyaw_grasp = rpy_to_matrix(*gcfg["wristyaw_grasp_rpy"])
        self.viz = viz                                  # optional viz.GraspViz (None = off)
        # last_snapshot (grasp.base.SourceSnapshot): the most recent mask (collision-world
        # object exclusion reads it) + cloud (perception validation reads it). Reset per call.
        self.last_snapshot = None

    def grasps(self, robot, side: str, target: str) -> List[GraspCandidate]:
        self.last_snapshot = None                      # stale-state guard (see SourceSnapshot)
        cam = getattr(robot, "camera", None)
        if cam is None:
            return []
        with LOG.span("depth_grab"):                   # head-camera rgb+depth capture
            rgb = cam.get_rgb_frame()                  # same-instant pair (rgb then depth)
            depth = cam.get_depth_frame()
        if depth is None or rgb is None:
            return []                                  # no depth (sim/off) -> caller fails loudly
        arm = getattr(robot, "arm", None)
        q14 = arm.get_current_dual_arm_q() if arm is not None else None
        T_pc = self.frames.T_pelvis_camera(q14)

        try:
            mask = self.segmenter.mask(rgb)            # (H,W) bool / all-False / None (whole frame)
        except SegmentationAborted:
            return []                                  # operator aborted -> no grasps (loud)
        snap = SourceSnapshot(target=target, mask=mask)  # mask -> collision_world.exclude_object
        self.last_snapshot = snap

        with LOG.span("deproject"):                    # masked depth -> pelvis-frame point cloud
            cloud = deproject_depth(depth, self.intrinsics, T_pc,
                                    voxel_m=self.gcfg.get("voxel_m"), mask=mask, rgb=rgb)
        snap.cloud = cloud                             # -> perception validation (GT compare)
        if cloud.is_empty():
            return []
        assert cloud.frame == "pelvis", f"cloud must be pelvis-frame, got {cloud.frame!r}"

        with self.client_factory() as client:
            grasps, conf, tags = client.infer(
                cloud.points, gripper_name=self.gcfg.get("gripper_name", "unitree_g1"),
                num_grasps=int(self.gcfg.get("num_grasps", 200)),
                grasp_threshold=float(self.gcfg.get("grasp_threshold", -1.0)),
                topk_num_grasps=int(self.gcfg.get("topk", 100)),
                planner=self.gcfg.get("planner"),               # diffusion|graspmoe|topdown
                obb_density=self.gcfg.get("obb_density"),
                skip_obb_rule=self.gcfg.get("skip_obb_rule"))
        grasps = np.asarray(grasps, dtype=np.float32)
        conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        k = min(grasps.shape[0], conf.shape[0])        # guard a grasps/conf length mismatch
        if k == 0:
            return []
        grasps, conf = grasps[:k], conf[:k]

        if self.viz is not None:                        # cloud + all grasps (pelvis frame)
            self.viz.show_candidates(cloud.points, grasps, conf, colors=cloud.colors,
                                     branch_tags=tags)
        # candidates_from_grasps stashes branch_tags[i] on cand.extra["branch_tag"], aligned
        # to each grasp by the same sort (one code path, shared with the sim-cloud source).
        out = candidates_from_grasps(grasps, conf, side, self.palm_offset_xyz,
                                     self.R_wristyaw_grasp, branch_tags=tags)
        snap.n_candidates = len(out)
        return out
