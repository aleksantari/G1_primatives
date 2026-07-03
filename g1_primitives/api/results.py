"""Result types the primitives return -- plain, serializable, planner-type-free.

``Result`` is the universal verb outcome; ``GraspResult`` adds a ``GraspReport`` -- a
JSON-friendly summary of a grasp attempt (which candidate won, per-phase outcomes, the
raw cuRobo status STRING). The planner's internal ``GraspPlanOutcome`` (trajectory
segments, tensors) never crosses this boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Result:
    """ok + a human-readable info string. Truthiness == ok, so ``if robot.home(): ...``
    reads naturally. NOTE for the hand verbs: without ``verify=True`` ok means
    COMMANDED, not confirmed -- the info string says which."""
    ok: bool
    info: str = ""

    def __bool__(self) -> bool:
        return self.ok


#: per-phase outcome vocabulary: "ok" (executed), "failed", "skipped" (not part of this
#: plan / disabled by options), "pending" (planned but never reached -- an earlier phase
#: failed first).
PHASES = ("approach", "grasp", "close", "lift")


@dataclass
class GraspReport:
    """Serializable summary of one grasp attempt (agent/telemetry-friendly)."""
    chosen_index: int = -1              # index into the FED candidate list (-1 = none)
    n_candidates: int = 0
    phases: Dict[str, str] = field(default_factory=dict)   # phase -> ok|failed|skipped|pending
    failed_phase: Optional[str] = None
    planner_status: str = ""            # raw cuRobo status text (diagnostic, not an object)
    strategy: Optional[dict] = None     # the winning {approach_offset, lift_offset, ...}


@dataclass
class GraspResult(Result):
    report: Optional[GraspReport] = None

    @property
    def chosen_index(self) -> int:
        return self.report.chosen_index if self.report is not None else -1
