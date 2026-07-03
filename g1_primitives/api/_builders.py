"""Private component assembly for the Robot facade.

Each builder maps one config block onto one constructed component. Heavy /
optional deps (viser, SAM3 client, camera client) are imported lazily INSIDE the
builders so importing the API never pays for components a run doesn't use. The
Robot facade (api.robot) is the only intended caller; the ``set_*`` reconfigure
methods re-run individual builders after mutating the config.
"""
from __future__ import annotations

import os
from typing import Any, Dict

from g1_primitives.config import _REPO_ROOT


def build_perception(planner, cfg: Dict[str, Any]):
    """Build the frame-math owner + the configured detector (no I/O). The seam for
    new detectors is the `detector:` selector in perception.yaml."""
    from g1_primitives.perception.frames import Frames
    from g1_primitives.perception.sim_state import SimStateDetector

    sim_base = (cfg["robot"].get("sim", {}) or {}).get("base_world_pose")
    frames = Frames(planner.fk_link, camera_cfg=cfg["camera"], sim_base_world_pose=sim_base)
    perc = cfg["perception"] or {}
    kind = perc.get("detector", "sim_state")
    if kind == "sim_state":
        # live sim ground-truth via rt/sim_state; needs DDS (a connected robot).
        # Future real-camera detectors register here (the `detector:` seam).
        detector = SimStateDetector.from_config(frames, perc.get("sim_state", {}))
    else:
        raise ValueError(f"unknown detector: {kind}")
    return frames, detector


def build_segmenter(seg_cfg: Dict[str, Any]):
    """Object segmenter from grasp.yaml `segment` (mode: null|auto|interactive). null/sim ->
    whole frame; auto/interactive -> SAM3 (lazy ZMQ socket). Mirrors the `detector:` seam."""
    from g1_primitives.perception.segment import NullSegmenter, Sam3Segmenter
    mode = seg_cfg.get("mode")
    if mode in (None, "null", "none", "sim"):
        return NullSegmenter()
    from g1_primitives.perception.sam3_client import Sam3Client

    def _sam3():
        return Sam3Client(host=seg_cfg.get("host", "127.0.0.1"),
                          port=int(seg_cfg.get("port", 5557)),
                          timeout_ms=int(seg_cfg.get("timeout_ms", 60000)))
    prompt = seg_cfg.get("default_prompt") or {}
    top_k = int(seg_cfg.get("top_k", 3))
    if mode == "auto":
        return Sam3Segmenter(_sam3, prompt, top_k=1)
    if mode == "interactive":
        from g1_primitives.perception.segment_gui import InteractiveSam3Segmenter
        return InteractiveSam3Segmenter(_sam3, prompt, top_k=top_k)
    raise ValueError(f"unknown segment mode: {mode}")


def build_grasp_viz(gx: Dict[str, Any]):
    """Build an optional client-side GraspViz from the graspgenx `visualize` block.
    Returns None unless `visualize.enabled`. Lazy imports (viser) live here so
    non-viz runs never touch viser."""
    vcfg = gx.get("visualize", {}) or {}
    if not vcfg.get("enabled"):
        return None
    from g1_primitives.viz import load_gripper_geom
    from g1_primitives.viz.grasp_viz import GraspViz
    asset_dir = vcfg.get("gripper_asset_dir", "assets/grippers")
    if not os.path.isabs(asset_dir):                # anchor on the repo root, not cwd
        asset_dir = os.path.join(_REPO_ROOT, asset_dir)
    geom = load_gripper_geom(asset_dir, gx.get("gripper_name", "unitree_g1"))
    return GraspViz(geom,
                    port=int(vcfg.get("port", 8080)),
                    max_markers=int(vcfg.get("max_markers", 100)),
                    show_mesh=bool(vcfg.get("show_mesh", True)),
                    threshold_tuner=bool(vcfg.get("threshold_tuner", False)))


def build_grasp_source(frames, cfg: Dict[str, Any]):
    """Build the configured grasp source (no I/O; the GraspGenX/SAM3 ZMQ sockets open lazily
    per call). Selector: grasp.yaml `grasp_source` -- mirrors the `detector:` seam."""
    g = cfg.get("grasp", {}) or {}
    kind = g.get("grasp_source", "graspgenx")
    if kind in ("graspgenx", "sim_cloud"):
        from g1_primitives.grasp.graspgenx_client import GraspGenXClient
        gx = dict(g.get("graspgenx", {}) or {})

        def _client():
            return GraspGenXClient(host=gx.get("host", "127.0.0.1"),
                                   port=int(gx.get("port", 5556)),
                                   timeout_ms=int(gx.get("timeout_ms", 60000)))
        viz = build_grasp_viz(gx)                       # None unless visualize.enabled
        if kind == "graspgenx":
            from g1_primitives.grasp.graspgenx_source import GraspGenXGraspSource
            seg = build_segmenter(g.get("segment", {}) or {})
            return GraspGenXGraspSource(frames, seg, _client, gx, cfg["camera"], viz=viz)
        # sim_cloud: GT cube cloud from rt/sim_state (no camera / depth / SAM3)
        from g1_primitives.grasp.sim_cloud_source import SimCloudGraspSource
        from g1_primitives.perception.sim_state import SimStateDetector
        gx["sim_cloud"] = g.get("sim_cloud", {}) or {}
        pose_source = SimStateDetector.from_config(
            frames, (cfg.get("perception", {}) or {}).get("sim_state", {}))
        return SimCloudGraspSource(frames, pose_source, _client, gx, cfg["camera"], viz=viz)
    raise ValueError(f"unknown grasp_source: {kind}")


def build_camera(cfg: Dict[str, Any]):
    from g1_primitives.hardware.camera_client import HeadCamera
    st = (cfg["camera"] or {}).get("stream", {})
    backend = st.get("backend", "zmq")
    # forward the rest of the stream block (port / request_port / stereo / stereo_side
    # / recv_timeout_ms) straight to HeadCamera as kwargs.
    extra = {k: v for k, v in st.items() if k not in ("backend", "host")}
    return HeadCamera(host=st.get("host", "127.0.0.1"), backend=backend, **extra)
