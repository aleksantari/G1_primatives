"""GraspOptions + GraspObserver -- the grasp verb's POLICY, replacing the old
four-positional-callback signature. The agent path passes nothing (defaults grasp,
close, lift); the operator-gated examples pass ``confirm=`` and an observer for viz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


class GraspObserver:
    """Side-channel hooks for viz/diagnostics during a grasp -- all no-ops here.
    Subclass and override what you need (the examples overlay viser scenes)."""

    def on_world_built(self, occupied_points) -> None:
        """After the depth-ESDF collision world is (re)built, BEFORE planning --
        ``occupied_points`` is (N,3) pelvis-frame voxel centres or None (world off)."""

    def on_selected(self, candidate, report) -> None:
        """After cuRobo picks the goalset winner, BEFORE any motion.
        ``candidate`` is the chosen GraspCandidate; ``report`` the (partial) GraspReport."""

    def on_phase(self, label: str, ok: bool, info: str) -> None:
        """After each executed phase (approach/grasp/close/lift)."""


@dataclass
class GraspOptions:
    """Policy for one ``grasp_motion`` call. Defaults = the full autonomous pick:
    approach -> grasp -> close -> lift, no operator gating."""
    close_hand: bool = True             # settle at the grasp pose, then close the hand
    close_fraction: float = 0.65
    verify_close: bool = False          # True -> close failure fails the grasp
    approach: bool = True               # plan the pre-grasp back-off segment
    lift: bool = True                   # plan the post-grasp lift segment
    grasp_z_offset: float = 0.0         # vertical pre-shift of every grasp goal (m, +z)
    max_candidates: Optional[int] = None  # feed only the top-N candidates (None = all)
    confirm: Optional[Callable[[str], None]] = None   # operator gate, called per phase label
    observer: Optional[GraspObserver] = None
