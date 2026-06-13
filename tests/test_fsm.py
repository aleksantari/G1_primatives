from g1_classical_manip.tasks.fsm import FSM, DONE, ABORT


def test_linear_fsm_reaches_done():
    fsm = FSM()
    fsm.add("A", lambda c: "B")
    fsm.add("B", lambda c: DONE)
    visited = []
    fsm.on_transition = lambda s, c: visited.append(s)
    assert fsm.run("A", ctx={}) == DONE
    assert visited == ["A", "B", DONE]


def test_retry_then_abort():
    fsm = FSM()

    def loop(c):
        c["n"] += 1
        return ABORT if c["n"] >= 3 else "LOOP"

    fsm.add("LOOP", loop)
    ctx = {"n": 0}
    assert fsm.run("LOOP", ctx) == ABORT
    assert ctx["n"] == 3


def test_missing_handler_raises():
    fsm = FSM()
    fsm.add("A", lambda c: "NOWHERE")
    try:
        fsm.run("A", ctx={})
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_max_steps_guard():
    fsm = FSM()
    fsm.add("A", lambda c: "A")  # infinite loop
    try:
        fsm.run("A", ctx={}, max_steps=10)
        assert False
    except RuntimeError:
        pass
