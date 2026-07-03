"""Lightweight latency instrumentation for the grasp pipeline.

A process-global ``LOG`` records named timing spans so a grasp run (``tools/grasp_preview``) can report how
long each component ACTUALLY takes -- isolating real inference/compute (depth grab, SAM3,
GraspGenX, cuRobo ``plan_grasp``, collision-world) from motion execution and from the human
GUI/keyboard gates, which otherwise dominate wall-clock but are not part of the pipeline speed.

Usage:
    from g1_primitives.latency import LOG
    LOG.enable()                       # start recording (also clears prior spans)
    with LOG.span("sam3"):             # a compute span (the default category)
        ...
    with LOG.span("wait: approach", LOG.WAIT):   # a human/gate span
        ...
    print(LOG.summary())               # text table
    LOG.plot("latency.png")            # timeline (Gantt) + per-component bar chart

Zero overhead when disabled: ``span()`` becomes a no-op context manager. The module imports
only the stdlib at import time (matplotlib is imported lazily inside ``plot``).
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass

# span categories -- what kind of time this is
COMPUTE = "compute"   # the inference / CPU-GPU work we want to measure (SAM3, GraspGenX, cuRobo, depth)
EXEC = "exec"         # trajectory playback / physical motion (governed by time_dilation)
WAIT = "wait"         # human-in-the-loop: keyboard confirm gates (NOT pipeline speed)


@dataclass
class Span:
    name: str
    category: str
    t0: float
    t1: float

    @property
    def ms(self) -> float:
        return (self.t1 - self.t0) * 1e3


class LatencyLog:
    # re-export the categories so callers only need to import LOG
    COMPUTE = COMPUTE
    EXEC = EXEC
    WAIT = WAIT

    def __init__(self):
        self.enabled = False
        self._spans: list[Span] = []

    def enable(self) -> None:
        """Start recording; clears any prior spans."""
        self.enabled = True
        self._spans = []

    def disable(self) -> None:
        self.enabled = False

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    @contextmanager
    def span(self, name: str, category: str = COMPUTE):
        """Time the wrapped block under ``name``. No-op (near-zero cost) when disabled."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._spans.append(Span(name, category, t0, time.perf_counter()))

    # ---- reporting -------------------------------------------------------------------------
    def _aggregate(self):
        """name -> [total_ms, count, category] in first-occurrence order."""
        agg: dict[str, list] = {}
        for s in self._spans:
            a = agg.setdefault(s.name, [0.0, 0, s.category])
            a[0] += s.ms
            a[1] += 1
        return agg

    def summary(self) -> str:
        if not self._spans:
            return "latency: (no spans recorded)"
        agg = self._aggregate()
        rows = ["", "=== latency summary (ms) ===",
                f"{'component':24s} {'calls':>5s} {'total':>9s} {'mean':>9s}  category"]
        for name, (tot, cnt, cat) in agg.items():
            rows.append(f"{name:24s} {cnt:5d} {tot:9.1f} {tot / cnt:9.1f}  {cat}")
        wall = (max(s.t1 for s in self._spans) - min(s.t0 for s in self._spans)) * 1e3
        comp = sum(s.ms for s in self._spans if s.category == COMPUTE)
        exe = sum(s.ms for s in self._spans if s.category == EXEC)
        wait = sum(s.ms for s in self._spans if s.category == WAIT)
        rows += ["-" * 52,
                 f"{'compute (inference)':24s} {'':5s} {comp:9.1f}",
                 f"{'exec (motion)':24s} {'':5s} {exe:9.1f}",
                 f"{'wait (confirm gates)':24s} {'':5s} {wait:9.1f}",
                 f"{'untimed gap (GUI/idle)':24s} {'':5s} {wall - comp - exe - wait:9.1f}",
                 f"{'wall-clock':24s} {'':5s} {wall:9.1f}"]
        return "\n".join(rows)

    def dump_json(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump([{"name": s.name, "category": s.category,
                        "t0": s.t0, "t1": s.t1, "ms": s.ms} for s in self._spans], f, indent=2)
        return path

    def plot(self, path: str, title: str | None = None) -> str | None:
        """Render a timeline (Gantt) + per-component bar chart to ``path`` (PNG). Returns the
        path, or None if nothing was recorded. Uses the Agg backend (no display needed)."""
        if not self._spans:
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        colors = {COMPUTE: "#2b6cb0", EXEC: "#2f855a", WAIT: "#dd6b20"}
        spans = self._spans
        t0 = min(s.t0 for s in spans)

        # rows ordered by first occurrence (top -> bottom)
        names: list[str] = []
        for s in spans:
            if s.name not in names:
                names.append(s.name)
        row = {n: len(names) - 1 - i for i, n in enumerate(names)}   # top name = highest y

        fig, (ax_t, ax_b) = plt.subplots(
            2, 1, figsize=(11, max(4.0, 0.5 * len(names) + 3.5)),
            gridspec_kw={"height_ratios": [max(2, len(names)), max(2, len(self._aggregate()))]})

        # --- timeline (Gantt): each span as a bar on its component's row ---
        for s in spans:
            ax_t.broken_barh([(s.t0 - t0, max(s.ms / 1e3, 1e-3))], (row[s.name] - 0.4, 0.8),
                             facecolors=colors.get(s.category, "#888888"))
        ax_t.set_yticks([row[n] for n in names])
        ax_t.set_yticklabels(names)
        ax_t.set_xlabel("time since first event (s)")
        ax_t.set_title(title or "grasp pipeline latency timeline")
        ax_t.grid(axis="x", alpha=0.3)
        ax_t.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()],
                    loc="lower right", fontsize=8, framealpha=0.9)

        # --- per-component bar chart (ascending so the slowest is on top) ---
        agg = self._aggregate()
        order = sorted(agg, key=lambda n: agg[n][0])
        vals = [agg[n][0] for n in order]
        ax_b.barh(order, vals, color=[colors.get(agg[n][2], "#888888") for n in order])
        for i, n in enumerate(order):
            tot, cnt, _ = agg[n]
            ax_b.text(tot, i, f"  {tot:.0f} ms" + (f"  (x{cnt})" if cnt > 1 else ""),
                      va="center", fontsize=8)
        ax_b.set_xlim(0, max(vals) * 1.18 if vals else 1)
        ax_b.set_xlabel("total time (ms)")
        ax_b.set_title("per-component time (blue = inference/compute, green = motion, orange = human gate)")
        ax_b.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path


# process-global recorder
LOG = LatencyLog()
