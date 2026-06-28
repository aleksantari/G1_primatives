"""GraspViz -- client-side viser visualization of GraspGenX grasps.

A data-driven refactor of the demo viz loop (GraspGenX scripts/demo_object_pc.py
lines ~472-698) operating on what the client already has: a pelvis-frame point
cloud, ranked grasps (K,4,4) + confidences, and (later) the cuRobo-chosen grasp.
No model, no centering -- everything is rendered directly in the pelvis frame, so
the grasps sit on the cloud and the gripper mesh lands where the hand will go.

Lives entirely on the client; the GraspGenX server stays headless. Uses the
vendored viser primitives + the torch-free gripper-geometry loader, so it pulls
nothing from the graspgenx package.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from g1_classical_manip.viz.gripper_geom import GripperGeom
from g1_classical_manip.viz import viser_primitives as vp

_BEST_COLOR = [0, 100, 255]      # blue   -- model's top-confidence grasp
_CHOSEN_COLOR = [0, 255, 0]      # green  -- the grasp cuRobo actually selected
_OBB_COLOR = [255, 150, 0]       # amber  -- OBB / top-down grasp (protocol-v2 branch_tag)
_DIFF_COLOR = [180, 80, 220]     # purple -- diffusion grasp


class GraspViz:
    """Render a point cloud + ranked grasps + gripper mesh in a viser GUI."""

    def __init__(
        self,
        gripper_geom: GripperGeom,
        port: int = 8080,
        max_markers: int = 100,
        show_mesh: bool = True,
        threshold_tuner: bool = True,
    ):
        self.geom = gripper_geom
        self.max_markers = int(max_markers)
        self.show_mesh = bool(show_mesh)
        self.threshold_tuner = bool(threshold_tuner)
        self.vis = vp.create_visualizer(port=port)
        # (confidence, [line handles]) per drawn marker -- for the threshold slider.
        self._grasp_handles: List[Tuple[float, list]] = []
        self._tuner_built = False

    def reset(self):
        """Clear the scene (between objects / runs)."""
        self.vis.scene.reset()
        self._grasp_handles = []
        self._tuner_built = False

    def show_candidates(
        self,
        points_xyz: np.ndarray,
        grasps_4x4: np.ndarray,
        conf: np.ndarray,
        colors: Optional[np.ndarray] = None,
        branch_tags: Optional[list] = None,
    ):
        """Draw the cloud + all (top-``max_markers``) grasps, best one highlighted.

        Args:
            points_xyz: (N,3) pelvis-frame point cloud.
            grasps_4x4: (K,4,4) grasp poses in the pelvis frame.
            conf:       (K,) confidences in [0,1].
            colors:     optional (N,3) uint8 RGB per point; None -> flat white.
            branch_tags: optional (K,) "obb"/"diff" per grasp (GraspGenX protocol v2). When
                given, grasps are colored by branch -- amber (OBB/top-down) vs purple
                (diffusion) -- instead of the confidence gradient; the best is still blue.
        """
        points_xyz = np.asarray(points_xyz, dtype=np.float32)
        grasps = np.asarray(grasps_4x4, dtype=np.float64).reshape(-1, 4, 4)
        conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        k = min(len(grasps), len(conf))
        grasps, conf = grasps[:k], conf[:k]

        self.vis.scene.reset()
        self._grasp_handles = []
        self._tuner_built = False

        vp.make_frame(self.vis, "pelvis", h=0.1)
        if len(points_xyz):
            vp.visualize_pointcloud(self.vis, "pc", points_xyz, color=colors, size=0.003)

        if k == 0:
            print("[GraspViz] no grasps to display.")
            return

        # Keep only the top-`max_markers` by confidence (viser slows with thousands).
        order = np.argsort(-conf)
        if self.max_markers > 0:
            order = order[: self.max_markers]
        best_idx = int(np.argmax(conf))
        colors = vp.get_color_from_score(conf, use_255_scale=True)
        tags = list(branch_tags) if branch_tags is not None else None

        for rank, j in enumerate(order):
            is_best = j == best_idx
            if is_best:
                color = _BEST_COLOR
            elif tags is not None and j < len(tags):       # color by branch (OBB vs diffusion)
                color = _OBB_COLOR if tags[j] == "obb" else _DIFF_COLOR
            else:                                          # default: confidence gradient
                color = colors[j]
            lw = 5.0 if is_best else 3.0
            handles = vp.visualize_x_grasp(
                self.vis, f"grasps/grasp_{rank:03d}", grasps[j],
                color=color, gripper_info=self.geom, linewidth=lw,
            )
            self._grasp_handles.append((float(conf[j]), handles))

        # Mesh overlay at the best grasp (the demo's --plot_top_mesh look).
        if self.show_mesh and self.geom.has_mesh:
            vp.visualize_mesh(
                self.vis, "top_grasp_mesh", self.geom.mesh,
                color=_BEST_COLOR, transform=grasps[best_idx],
            )

        branch = ""
        if tags is not None:                               # obb (amber) vs diff (purple) split
            n_obb = sum(1 for t in tags[:k] if t == "obb")
            branch = f"obb={n_obb} diff={k - n_obb} | "
        print(
            f"[GraspViz] {len(order)}/{k} grasps shown | "
            f"conf [{conf.min():.3f}, {conf.max():.3f}] | {branch}"
            f"best idx {best_idx} ({conf[best_idx]:.3f}) | "
            f"mesh overlay: {self.show_mesh and self.geom.has_mesh}"
        )

        if self.threshold_tuner:
            self._build_threshold_tuner()

    def mark_chosen(self, grasp_pose_4x4: np.ndarray):
        """Overlay the cuRobo-selected grasp distinctly (green marker + mesh)."""
        T = np.asarray(grasp_pose_4x4, dtype=np.float64).reshape(4, 4)
        vp.visualize_x_grasp(
            self.vis, "chosen_grasp", T,
            color=_CHOSEN_COLOR, gripper_info=self.geom, linewidth=6.0,
        )
        if self.show_mesh and self.geom.has_mesh:
            vp.visualize_mesh(
                self.vis, "chosen_grasp_mesh", self.geom.mesh,
                color=_CHOSEN_COLOR, transform=T,
            )
        print(f"[GraspViz] chosen grasp marked at t={np.round(T[:3, 3], 3)}")

    def spin(self):
        """Block so the viser server stays up (for the standalone inspection tool)."""
        import time
        print("[GraspViz] viser running; press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

    # -- internals ----------------------------------------------------------

    def _build_threshold_tuner(self):
        """Add a confidence-threshold slider that toggles marker visibility."""
        if self._tuner_built or not self._grasp_handles:
            return
        n = len(self._grasp_handles)
        with self.vis.gui.add_folder("Threshold Tuner"):
            count_md = self.vis.gui.add_markdown(f"**Visible: {n} / {n}**")
            slider = self.vis.gui.add_slider(
                "Confidence threshold", min=0.0, max=1.0, step=0.01,
                initial_value=0.0,
            )

        @slider.on_update
        def _on_update(_):
            thresh = slider.value
            visible = 0
            for c, handles in self._grasp_handles:
                show = bool(c >= thresh)
                visible += int(show)
                for h in handles:
                    h.visible = show
            count_md.content = f"**Visible: {visible} / {n}**"

        self._tuner_built = True
