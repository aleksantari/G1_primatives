"""api.results types + the GraspPlanOutcome -> GraspReport mapping (GPU-free)."""
from g1_primitives.api.results import Result, GraspResult, GraspReport
from g1_primitives.api.primitives import _report_from_outcome
from g1_primitives.motion.planner import GraspPlanOutcome


def test_result_truthiness():
    assert Result(True, "ok") and not Result(False, "nope")
    r = GraspResult(True, "ok", GraspReport(chosen_index=3))
    assert r and r.chosen_index == 3
    assert GraspResult(False, "x").chosen_index == -1        # no report -> -1


def test_report_success_mapping():
    out = GraspPlanOutcome(True, 2, "A", "G", "L", True, True, True, "ok",
                           strategy={"approach_offset": -0.1})
    rep = _report_from_outcome(out, 5)
    assert rep.chosen_index == 2 and rep.n_candidates == 5
    # planned segments start 'pending' (execution overwrites); close pending too
    assert rep.phases == {"approach": "pending", "grasp": "pending",
                          "lift": "pending", "close": "pending"}
    assert rep.failed_phase is None and rep.strategy == {"approach_offset": -0.1}


def test_report_plan_failure_mapping():
    # approach planning failed -> that's the failed phase; nothing else planned
    out = GraspPlanOutcome(False, 7, None, None, None, False, False, False,
                           "Planning to approach pose failed.")
    rep = _report_from_outcome(out, 10)
    assert rep.failed_phase == "approach"
    assert rep.phases["approach"] == "failed"
    assert rep.planner_status.startswith("Planning to approach")
    assert rep.chosen_index == 7                             # the rejected winner is surfaced


def test_report_lift_skipped_mapping():
    # a plan without a lift segment (plan_lift=False strategy) -> lift 'skipped'
    out = GraspPlanOutcome(True, 0, "A", "G", None, True, True, True, "ok")
    rep = _report_from_outcome(out, 1)
    assert rep.phases["lift"] == "skipped"
    assert rep.phases["approach"] == "pending" and rep.phases["grasp"] == "pending"
