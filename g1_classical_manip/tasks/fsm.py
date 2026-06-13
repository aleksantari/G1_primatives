"""A small, generic state machine. States are handlers ctx -> next_state_name.
Terminal states end the run. Every transition fires an on_transition callback
(used to log state/q/target/perception to Rerun -- CLAUDE.md rule 5).
"""
from __future__ import annotations

from typing import Callable, Dict, Any, Iterable

DONE = "DONE"
ABORT = "ABORT"


class FSM:
    def __init__(self, on_transition: Callable[[str, Any], None] = None,
                 terminal: Iterable[str] = (DONE, ABORT)):
        self.states: Dict[str, Callable[[Any], str]] = {}
        self.on_transition = on_transition
        self.terminal = set(terminal)

    def state(self, name: str):
        def deco(fn):
            self.states[name] = fn
            return fn
        return deco

    def add(self, name: str, handler: Callable[[Any], str]):
        self.states[name] = handler

    def run(self, start: str, ctx: Any, max_steps: int = 1000) -> str:
        cur = start
        steps = 0
        while cur not in self.terminal:
            if steps >= max_steps:
                raise RuntimeError(f"FSM exceeded {max_steps} steps at '{cur}'")
            if self.on_transition is not None:
                self.on_transition(cur, ctx)
            if cur not in self.states:
                raise KeyError(f"no handler for state '{cur}'")
            cur = self.states[cur](ctx)
            steps += 1
        if self.on_transition is not None:
            self.on_transition(cur, ctx)
        return cur
