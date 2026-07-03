"""End-effector abstraction (CLAUDE.md: thin interface over Dex3/Dex1).

A Hand wraps a low-level threaded controller (robot_control.robot_hand_unitree)
and adds named presets + grasp verification from motor stall / tau_est / press
sensors. Concrete hands: ee.dex3.Dex3Hand, ee.dex1.Dex1Hand. The factory picks
one from configs/robot.yaml `hand:`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

LEFT = "left"
RIGHT = "right"
SIDES = (LEFT, RIGHT)


class Hand(ABC):
    @abstractmethod
    def command(self, side: str, q):
        """Send a raw joint target to one hand."""

    @abstractmethod
    def get_state(self, side: str) -> dict:
        """Return {'q','dq','tau','press'} for one hand."""

    @abstractmethod
    def open(self, side: str, verify: bool = True) -> bool:
        """Command the open preset; if verify, confirm it reached open."""

    @abstractmethod
    def close(self, side: str, verify: bool = True, fraction: float = 1.0) -> bool:
        """Command the close target, interpolated open->close by `fraction` (1.0 = full
        close, 0.5 = halfway). If verify, return whether an object is held (motor stall
        short of target OR tau_est OR press sensor over threshold)."""

    @abstractmethod
    def grasped(self, side: str) -> bool:
        """Instantaneous grasp-success check against the last close target."""
