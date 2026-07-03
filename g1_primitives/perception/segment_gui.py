"""Interactive OpenCV GUI to refine a SAM3 segmentation mask before it propagates to the
depth->cloud pipeline. The operator tries text/box/point prompts, sees the mask live, and
accepts when satisfied.

Mouse:  L-click = +point   R-click = -point   L-drag = box
Keys :  t = type a text prompt (terminal)   n/p = cycle candidate masks
        r = reset   Enter / c = accept   q / Esc = abort

Needs a display (cv2 highgui). On a headless box it raises a clear RuntimeError telling the
operator to use --segment auto|none. Not unit-tested (requires a display).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from g1_primitives.perception.segment import SegmentationAborted, Segmenter

try:
    import cv2
except Exception:
    cv2 = None


def _no_display():
    raise RuntimeError("interactive segmentation needs a display (cv2 highgui); "
                       "use --segment auto or --segment none instead.")


def refine_mask(rgb: np.ndarray, client, prompt_cfg: Optional[dict] = None, *,
                top_k: int = 3, window: str = "SAM3 refine") -> Optional[np.ndarray]:
    """Drive `client` (a connected Sam3Client) with operator prompts over `rgb`; return the
    accepted (H,W) bool mask. Raises SegmentationAborted if the operator aborts."""
    if cv2 is None:
        _no_display()
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb, np.uint8), cv2.COLOR_RGB2BGR)

    st = {
        "mode": "points", "points": [], "labels": [], "box": None,
        "drag_start": None, "drag_rect": None, "dragging": False, "text": None,
        "masks": np.zeros((0, h, w), np.uint8), "scores": np.zeros((0,), np.float32),
        "labels_out": [], "sel": 0, "dirty": False, "status": "",
    }

    def _query():
        try:
            if st["mode"] == "points" and st["points"]:
                res = client.segment(rgb, points=st["points"],
                                     point_labels=st["labels"], top_k=top_k)
            elif st["mode"] == "box" and st["box"] is not None:
                res = client.segment(rgb, box=st["box"], top_k=top_k)
            elif st["mode"] == "text" and st["text"]:
                res = client.segment(rgb, text=st["text"], top_k=top_k)
            else:
                return
            st["masks"], st["scores"], st["labels_out"], st["sel"] = (*res, 0)
            st["status"] = "no object found" if st["masks"].shape[0] == 0 else ""
        except (TimeoutError, RuntimeError) as e:
            st["status"] = f"SAM3 error: {e}"

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            st["drag_start"], st["dragging"], st["drag_rect"] = (x, y), True, None
        elif event == cv2.EVENT_MOUSEMOVE and st["dragging"]:
            sx, sy = st["drag_start"]
            st["drag_rect"] = (sx, sy, x, y)
        elif event == cv2.EVENT_LBUTTONUP and st["dragging"]:
            st["dragging"] = False
            sx, sy = st["drag_start"]
            st["drag_start"], st["drag_rect"] = None, None
            if abs(x - sx) >= 5 and abs(y - sy) >= 5:          # a box
                st["box"] = [min(sx, x), min(sy, y), max(sx, x), max(sy, y)]
                st["mode"], st["points"], st["labels"] = "box", [], []
            else:                                              # a click = +point
                st["points"].append([x, y])
                st["labels"].append(1)
                st["mode"], st["box"] = "points", None
            st["dirty"] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            st["points"].append([x, y])
            st["labels"].append(0)                             # -point (background)
            st["mode"], st["box"], st["dirty"] = "points", None, True

    def _render():
        out = bgr.copy()
        m_n = st["masks"].shape[0]
        if m_n > 0 and 0 <= st["sel"] < m_n:
            m = st["masks"][st["sel"]] > 0
            tint = np.zeros_like(out)
            tint[m] = (255, 80, 0)
            out = cv2.addWeighted(out, 1.0, tint, 0.45, 0)
        if st["box"] is not None:
            x0, y0, x1, y1 = (int(v) for v in st["box"])
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
        if st["drag_rect"] is not None:
            x0, y0, x1, y1 = (int(v) for v in st["drag_rect"])
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 200), 1)
        for (px, py), lb in zip(st["points"], st["labels"]):
            cv2.circle(out, (int(px), int(py)), 5, (0, 255, 0) if lb else (0, 0, 255), -1)
        score = f"{st['scores'][st['sel']]:.2f}" if m_n > 0 else "-"
        label = st["labels_out"][st["sel"]] if (m_n > 0 and st["labels_out"]) else ""
        cv2.putText(out, f"mode={st['mode']}  mask {st['sel'] + 1 if m_n else 0}/{m_n}  "
                         f"score={score}  {label}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.putText(out, "L+=pt R-=pt drag=box  t=text  n/p=cycle  r=reset  "
                         "Enter=accept  q=abort", (8, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if st["status"]:
            cv2.putText(out, st["status"], (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
        return out

    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, _on_mouse)
        cv2.imshow(window, bgr)
        cv2.waitKey(1)
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(window)
            _no_display()
    except cv2.error:
        _no_display()

    if prompt_cfg:                                             # seed the first query
        if "text" in prompt_cfg:
            st["mode"], st["text"] = "text", prompt_cfg["text"]
        elif "box" in prompt_cfg:
            st["mode"], st["box"] = "box", list(prompt_cfg["box"])
        elif "points" in prompt_cfg:
            st["mode"] = "points"
            st["points"] = [list(p) for p in prompt_cfg["points"]]
            st["labels"] = list(prompt_cfg.get("point_labels") or [1] * len(st["points"]))
        st["dirty"] = True

    result = None
    while True:
        if st["dirty"]:
            _query()
            st["dirty"] = False
        cv2.imshow(window, _render())
        k = cv2.waitKey(20) & 0xFF
        if k == ord("t"):
            try:
                t = input("text prompt> ").strip()
            except EOFError:
                t = ""
            if t:
                st.update(mode="text", text=t, points=[], labels=[], box=None, dirty=True)
        elif k in (ord("n"), ord("m")) and st["masks"].shape[0] > 0:
            st["sel"] = (st["sel"] + 1) % st["masks"].shape[0]
        elif k == ord("p") and st["masks"].shape[0] > 0:
            st["sel"] = (st["sel"] - 1) % st["masks"].shape[0]
        elif k == ord("r"):
            st.update(mode="points", points=[], labels=[], box=None, text=None, sel=0,
                      masks=np.zeros((0, h, w), np.uint8),
                      scores=np.zeros((0,), np.float32), labels_out=[], status="")
        elif k in (13, ord("c")) and st["masks"].shape[0] > 0:    # Enter / c = accept
            result = st["masks"][st["sel"]] > 0
            break
        elif k in (ord("q"), 27):                                 # q / Esc = abort
            cv2.destroyWindow(window)
            raise SegmentationAborted("operator aborted segmentation")
        try:                                                      # window closed via 'X'
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise SegmentationAborted("segmentation window closed")
        except cv2.error:
            raise SegmentationAborted("segmentation window closed")

    cv2.destroyWindow(window)
    return result


class InteractiveSam3Segmenter(Segmenter):
    """Operator refines the SAM3 prompt (text/box/points) in the cv2 GUI above, then accepts
    a mask. Raises SegmentationAborted if the operator aborts. Lives here (not segment.py)
    because it is display-bound -- segment.py stays importable headless with no cycle."""

    def __init__(self, client_factory, prompt_cfg=None, top_k: int = 3):
        self.client_factory = client_factory
        self.prompt_cfg = prompt_cfg
        self.top_k = int(top_k)

    def mask(self, rgb: np.ndarray):
        with self.client_factory() as c:           # one persistent client across all queries
            return refine_mask(rgb, c, self.prompt_cfg, top_k=self.top_k)
