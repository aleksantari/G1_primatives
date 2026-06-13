"""cuRoboV2 planner adapter (Phase 7) -- interface-conformant STUB.

cuRobo is NOT installed in this pass (heavy CUDA build; torch>=2.5). This stub
keeps the planner seam alive so `planner: curobo` in configs/planner.yaml is a
one-line switch once Phase 7 is implemented. It returns the SAME JointPath the
Cartesian planner returns.

Implementation notes for Phase 7 (see G1_CLASSICAL_MANIP_PLAN.md sec 0.6 / 7):
  * Build a fixed-base dual-arm config from the shipped unitree_g1.yml: no
    floating extra_links, legs+waist locked at measured q (verify the v2
    lock-joints field; fallback = reduced URDF + sphere subset).
  * World model: table cuboid + perceived block cuboid (World.obstacles here).
  * Map the pick phase onto cuRobo's approach/grasp/lift grasp-planning call.
  * Keep Ruckig retiming optional behind planner.yaml `curobo.use_ruckig_retime`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from g1_classical_manip.motion.planner_base import Planner, Goal, World, JointPath


class CuRoboPlanner(Planner):
    def __init__(self, ik=None, planner_cfg: Optional[dict] = None, **kwargs):
        self.ik = ik
        self.cfg = (planner_cfg or {}).get("curobo", {})

    def plan(self, start_q14: np.ndarray, goal: Goal,
             world: Optional[World] = None) -> JointPath:
        raise NotImplementedError(
            "CuRoboPlanner is a Phase-7 stub. Install cuRobo (cu12-torch extra) "
            "and implement against the same JointPath contract; see this file's "
            "docstring and G1_CLASSICAL_MANIP_PLAN.md sec 7.")
