#!/usr/bin/env python
"""examples/03 - Interactive API playground: drive every agent-facing primitive from a
viser web GUI and see each call the way an LLM AGENT would -- `tool(args) -> JSON result`.

One browser tab (default http://localhost:8080): the 3-D grasp scene (cloud, ranked
grasps, chosen marker, ESDF world, live wrist frames) + a GUI panel with verb buttons,
GraspOptions controls, a free-text TOOL-CALL box (`grasp {"side": "right"}`), and a
rolling agent-wire log. Buttons and typed tool calls go through the SAME tool registry,
so the log always shows the exact JSON an agent would receive.

Runs the STANDARD full stack by default (source graspgenx + SAM3 interactive), sim and
real alike; `--source sim_cloud` is the rarely-used cube-only sim GT fallback.

    sim : CYCLONEDDS_URI=file://$PWD/configs/cyclonedds_loopback.xml \\
          bash -ic 'use_conda g1_curobo && python scripts/examples/03_playground.py --target sim'
    real: bash -ic 'use_conda g1_curobo && python scripts/examples/03_playground.py --target real'

Safety: one tool runs at a time (buttons grey out); ABORT unwinds a grasp at the next
phase gate and HOLDS (no recovery motion) -- the hard stop on real remains the e-stop.
`--offline` runs the GUI with no DDS/robot (motion tools return an error Result).
Source/segmenter are launch flags, not GUI controls: switching them rebuilds the grasp
source, which owns this GUI's viser server.
"""
import argparse
import json
import threading
import time
import traceback
from dataclasses import asdict

import numpy as np

import g1_primitives as g1
from g1_primitives.api import console
from g1_primitives.api.results import Result

LOG_LINES = 60          # rendered tail of the tool log
GATE_POLL_S = 0.1


# --------------------------------------------------------------- serialization
def _clean(x):
    """Recursively make a tool result JSON-serializable (numpy -> python, 3 decimals)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):          # before int: bool subclasses int
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), 3)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    return x


def result_json(res) -> dict:
    """Result / GraspResult -> the agent-wire dict."""
    out = {"ok": bool(res.ok), "info": res.info}
    report = getattr(res, "report", None)
    if report is not None:
        out["report"] = asdict(report)
    return _clean(out)


def pose_json(pose) -> dict:
    return _clean({"xyz": pose.translation.tolist(),
                   "quat_wxyz": pose.quaternion_wxyz().tolist()})


# --------------------------------------------------------------------- tool log
class ToolLog:
    """Ring buffer rendered into one viser markdown pane (agent-wire format)."""

    def __init__(self):
        self._lines = []
        self._md = None
        self._lock = threading.Lock()

    def attach(self, md_handle):
        self._md = md_handle

    def add(self, line: str):
        with self._lock:
            self._lines.append(line)
            tail = self._lines[-LOG_LINES:]
        if self._md is not None:
            self._md.content = "```text\n" + "\n".join(tail) + "\n```"

    def call(self, name: str, args: dict):
        self.add(f"> {name} {json.dumps(_clean(args))}")

    def result(self, payload: dict, dt: float):
        self.add(f"  -> {json.dumps(payload)}   ({dt:.1f}s)")


# ------------------------------------------------------------------ tool runner
class ToolRunner:
    """Runs ONE tool call at a time on a worker thread; rejects calls while busy.
    `abort` is checked at grasp phase gates (soft abort: unwind + hold)."""

    def __init__(self, log: ToolLog):
        self.log = log
        self.busy = None                 # name of the running tool, or None
        self.abort = threading.Event()
        self.gate = threading.Event()    # CONTINUE button at confirm gates
        self._buttons = []               # disabled while busy (ABORT stays live)
        self._lock = threading.Lock()

    def register_buttons(self, handles):
        self._buttons.extend(handles)

    def submit(self, name: str, fn, args: dict):
        with self._lock:
            if self.busy is not None:
                self.log.add(f"! busy: '{self.busy}' still running -- '{name}' rejected")
                return
            self.busy = name
        self.abort.clear()
        for b in self._buttons:
            b.disabled = True
        self.log.call(name, args)
        threading.Thread(target=self._run, args=(name, fn, args), daemon=True).start()

    def _run(self, name, fn, args):
        t0 = time.time()
        try:
            payload = fn(**args)
        except KeyboardInterrupt as e:            # soft abort at a phase gate
            payload = {"ok": False, "info": f"aborted: {e} -- HOLDING (no recovery)"}
        except Exception as e:                    # noqa: BLE001 - agent-wire error, never crash the GUI
            payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        self.log.result(payload, time.time() - t0)
        for b in self._buttons:
            b.disabled = False
        with self._lock:
            self.busy = None

    def gate_fn(self, wait: bool):
        """The GraspOptions.confirm callback: always abort-checked; when `wait`,
        pause at each phase gate until CONTINUE (or ABORT)."""
        def confirm(label: str):
            if self.abort.is_set():
                raise KeyboardInterrupt(f"ABORT before '{label}'")
            if not wait:
                return
            self.gate.clear()
            self.log.add(f"  .. gate '{label}': CONTINUE / ABORT")
            while not self.gate.wait(timeout=GATE_POLL_S):
                if self.abort.is_set():
                    raise KeyboardInterrupt(f"ABORT at '{label}' gate")
        return confirm


# ---------------------------------------------------------------- tool registry
def build_tools(robot, runner: ToolRunner, ui):
    """The agent surface: name -> fn(**args) returning a JSON-ready dict. `ui` supplies
    GUI-widget defaults (tool-call args override them); ui is None-safe for tests."""

    def _need_motion():
        if robot.executor is None:
            return {"ok": False, "info": "offline -- no executor/arm connected"}
        return None

    def home():
        return _need_motion() or result_json(robot.home())

    def move(side, xyz, rpy):
        err = _need_motion()
        if err:
            return err
        goal = g1.Pose.from_xyz_rpy(np.asarray(xyz, float), np.asarray(rpy, float))
        return result_json(robot.move(side, goal))

    def open_hand(side, verify=False):
        return _need_motion() or result_json(robot.open_hand(side, verify=verify))

    def close_hand(side, verify=False, fraction=1.0):
        return _need_motion() or result_json(
            robot.close_hand(side, verify=verify, fraction=fraction))

    def detect(target):
        det = robot.detect(target)
        if det is None:
            return {"found": False, "target": target}
        return {"found": True, "label": det.label, "pose": pose_json(det.pose)}

    def grasp_candidates(side, target):
        cands = robot.grasp_candidates(side, target)
        return _clean({
            "n": len(cands),
            "top": [{"confidence": c.confidence,
                     "xyz": c.wrist_goal.translation.tolist(),
                     "branch": c.extra.get("branch_tag", "?")} for c in cands[:5]],
        })

    def grasp(side, target, **opt):
        err = _need_motion()
        if err:
            return err
        options = g1.GraspOptions(
            approach=opt.get("approach", True), lift=opt.get("lift", True),
            verify_close=opt.get("verify_close", False),
            close_fraction=opt.get("close_fraction", 0.65),
            grasp_z_offset=opt.get("grasp_z_offset", 0.0),
            max_candidates=opt.get("max_candidates"),
            confirm=runner.gate_fn(opt.get("gates", False)),
            observer=ui.observer if ui is not None else None)
        return result_json(robot.grasp(side, target, options=options))

    return {
        "home": (home, "move both arms to the home pose"),
        "move": (move, 'move one wrist: {"side","xyz":[..3],"rpy":[..3]}'),
        "open_hand": (open_hand, "open the hand"),
        "close_hand": (close_hand, "close the hand (fraction, verify)"),
        "detect": (detect, "object pose (sim GT detector)"),
        "grasp_candidates": (grasp_candidates, "run the grasp source once"),
        "grasp": (grasp, "full pick: candidates -> plan -> approach/grasp/close/lift"),
    }


# ------------------------------------------------------------------------- GUI
class UI:
    """Owns the viser GUI folders + the grasp observer (viz + log side-channel)."""

    def __init__(self, robot, vis, runner, log, live_cloud_default=True):
        self.robot, self.vis, self.runner, self.log = robot, vis, runner, log
        gui = vis.gui

        self.status = gui.add_markdown("connecting ...")

        with gui.add_folder("Verbs"):
            self.side = gui.add_dropdown("side", (g1.LEFT, g1.RIGHT),
                                         initial_value=g1.RIGHT)
            self.obj = gui.add_text("object", initial_value="block")
            b_home = gui.add_button("home")
            b_open = gui.add_button("open hand")
            b_close = gui.add_button("close hand")
            b_det = gui.add_button("detect")
            b_cand = gui.add_button("grasp candidates")
            b_grasp = gui.add_button("GRASP")
            b_cont = gui.add_button("CONTINUE (gate)")
            b_abort = gui.add_button("ABORT")

        with gui.add_folder("Move", expand_by_default=False):
            q0 = np.zeros(14) if robot.arm is None else robot.arm.get_current_dual_arm_q()
            t0 = robot.planner.fk(g1.RIGHT, q0).translation
            self.xyz = gui.add_vector3("xyz (pelvis, m)", initial_value=tuple(
                round(float(v), 3) for v in t0), step=0.01)
            self.rpy = gui.add_vector3("rpy (rad)", initial_value=(0.0, 0.0, 0.0),
                                       step=0.05)
            b_move = gui.add_button("move wrist")

        with gui.add_folder("Grasp options", expand_by_default=False):
            self.o_approach = gui.add_checkbox("approach", initial_value=True)
            self.o_lift = gui.add_checkbox("lift", initial_value=True)
            self.o_verify = gui.add_checkbox("verify close", initial_value=False)
            self.o_gates = gui.add_checkbox("pause at phase gates", initial_value=False)
            self.o_frac = gui.add_slider("close fraction", min=0.1, max=1.0, step=0.05,
                                         initial_value=0.65)
            self.o_maxc = gui.add_slider("max candidates (0=all)", min=0, max=20, step=1,
                                         initial_value=0)
            self.o_zoff = gui.add_slider("grasp z offset (m)", min=-0.05, max=0.05,
                                         step=0.005, initial_value=0.0)

        with gui.add_folder("Session", expand_by_default=False):
            ex = robot.executor
            self.s_speed = gui.add_slider("speed (time dilation)", min=0.1, max=1.0,
                                          step=0.05,
                                          initial_value=(ex.time_dilation if ex else 1.0))
            self.s_cw = gui.add_checkbox(
                "collision world", initial_value=robot.planner.collision_world_enabled)
            self.s_cloud = gui.add_checkbox("live camera cloud",
                                            initial_value=live_cloud_default)

        with gui.add_folder("Tool call (agent wire)"):
            gui.add_markdown("`name {json}` -- e.g. `grasp "
                             '{"side": "right", "target": "block"}`')
            self.tool_in = gui.add_text("tool call", initial_value="")
            b_run = gui.add_button("run tool call")

        with gui.add_folder("Tool log"):
            log.attach(gui.add_markdown("```text\n(ready)\n```"))

        self.tools = build_tools(robot, runner, self)
        runner.register_buttons([b_home, b_open, b_close, b_det, b_cand, b_grasp,
                                 b_move, b_run])

        # ---- button wiring (buttons collect args from the widgets) ----
        b_home.on_click(lambda _: self._submit("home", {}))
        b_open.on_click(lambda _: self._submit("open_hand", {"side": self.side.value}))
        b_close.on_click(lambda _: self._submit("close_hand", {
            "side": self.side.value, "verify": self.o_verify.value,
            "fraction": float(self.o_frac.value)}))
        b_det.on_click(lambda _: self._submit("detect", {"target": self.obj.value}))
        b_cand.on_click(lambda _: self._submit("grasp_candidates", {
            "side": self.side.value, "target": self.obj.value}))
        b_grasp.on_click(lambda _: self._submit("grasp", self.grasp_args()))
        b_move.on_click(lambda _: self._submit("move", {
            "side": self.side.value, "xyz": list(self.xyz.value),
            "rpy": list(self.rpy.value)}))
        b_cont.on_click(lambda _: runner.gate.set())
        b_abort.on_click(lambda _: (runner.abort.set(),
                                    log.add("! ABORT requested (lands at the next "
                                            "phase gate; e-stop for hard stop)")))
        b_run.on_click(lambda _: self._typed(self.tool_in.value))
        self.s_speed.on_update(lambda _: robot.executor is not None
                               and robot.set_executor(speed=float(self.s_speed.value)))
        self.s_cw.on_update(lambda _: robot.set_collision_world(bool(self.s_cw.value)))

    # the GRASP button's args, read from the option widgets
    def grasp_args(self):
        mc = int(self.o_maxc.value)
        return {"side": self.side.value, "target": self.obj.value,
                "approach": self.o_approach.value, "lift": self.o_lift.value,
                "verify_close": self.o_verify.value, "gates": self.o_gates.value,
                "close_fraction": float(self.o_frac.value),
                "grasp_z_offset": float(self.o_zoff.value),
                "max_candidates": (mc if mc > 0 else None)}

    def _submit(self, name, args):
        fn, _ = self.tools[name]
        self.runner.submit(name, fn, args)

    def _typed(self, text: str):
        """Parse `name {json}` exactly as an agent host would; errors go to the log."""
        text = (text or "").strip()
        if not text:
            return
        name, _, rest = text.partition(" ")
        try:
            args = json.loads(rest) if rest.strip() else {}
            if not isinstance(args, dict):
                raise ValueError("args must be a JSON object")
        except ValueError as e:
            self.log.call(name, {"raw": rest.strip()})
            self.log.add(f'  -> {{"ok": false, "error": "bad tool-call JSON: {e}"}}')
            return
        if name not in self.tools:
            self.log.call(name, args)
            known = ", ".join(sorted(self.tools))
            self.log.add(f'  -> {{"ok": false, "error": "unknown tool \'{name}\'; '
                         f'tools: {known}"}}')
            return
        if name == "grasp":                      # widget defaults under typed overrides
            args = {**self.grasp_args(), **args}
        self._submit(name, args)

    # ---- grasp observer: viz + log side-channel (mirrors 02_pick's _Obs) ----
    @property
    def observer(self):
        ui = self

        class _Obs(g1.GraspObserver):
            def on_world_built(self, pts):
                viz = getattr(ui.robot.grasp_source, "viz", None)
                if viz is None or pts is None:
                    return
                q = (ui.robot.arm.get_current_dual_arm_q()
                     if ui.robot.arm is not None else np.zeros(14))
                viz.show_collision_world(
                    pts, voxel_size=ui.robot.planner._cw_params()["esdf_voxel_size"],
                    wrists={s: ui.robot.planner.fk(s, q).translation
                            for s in (g1.LEFT, g1.RIGHT)})

            def on_selected(self, chosen, report):
                viz = getattr(ui.robot.grasp_source, "viz", None)
                if viz is not None and chosen.grasp_pose is not None:
                    viz.mark_chosen(chosen.grasp_pose.homogeneous)
                ui.log.add(f"  .. chosen: idx {report.chosen_index} "
                           f"confidence {chosen.confidence:.3f}")

            def on_phase(self, label, ok, info):
                ui.log.add(f"  .. phase {label}: {'ok' if ok else 'FAILED'} ({info})")

        return _Obs()


# ----------------------------------------------------------------------- ticker
def ticker(robot, vis, ui, runner, stop: threading.Event):
    """Background scene refresh: wrist frames (~5 Hz) + optional camera cloud (~1 Hz).
    SKIPS while a tool runs -- the planner must never be used from two threads."""
    from g1_primitives.perception.depth import deproject_depth
    frames = {}
    last_cloud = 0.0
    while not stop.is_set():
        time.sleep(0.2)
        if runner.busy is not None:
            continue
        try:
            q = (robot.arm.get_current_dual_arm_q()
                 if robot.arm is not None else np.zeros(14))
            for side in (g1.LEFT, g1.RIGHT):
                T = robot.planner.fk(side, q)
                h = frames.get(side)
                if h is not None:
                    try:
                        h.wxyz, h.position = (tuple(T.quaternion_wxyz()),
                                              tuple(T.translation))
                    except RuntimeError:   # GraspViz.reset() wiped the scene -> recreate
                        h = None
                if h is None:
                    frames[side] = vis.scene.add_frame(
                        f"/live/wrist_{side}", axes_length=0.08, axes_radius=0.004,
                        wxyz=tuple(T.quaternion_wxyz()), position=tuple(T.translation))
            ex = robot.executor
            busy = runner.busy or "idle"
            ui.status.content = (
                f"**{robot.target}** | source `{robot.grasp_source_kind}` | **{busy}** | "
                f"speed {ex.time_dilation if ex else '-'} | "
                f"cw {'on' if robot.planner.collision_world_enabled else 'off'}\n\n"
                f"q L {np.round(np.rad2deg(q[:7]), 0).astype(int).tolist()}\n\n"
                f"q R {np.round(np.rad2deg(q[7:]), 0).astype(int).tolist()}")
            if (ui.s_cloud.value and robot.camera is not None
                    and time.time() - last_cloud > 1.0):
                depth, rgb = robot.depth(), robot.rgb()
                if depth is not None:
                    cloud = deproject_depth(depth, robot.cfg["camera"]["intrinsics"],
                                            robot.frames.T_pelvis_camera(q),
                                            voxel_m=0.02, rgb=rgb)
                    if not cloud.is_empty():
                        from g1_primitives.viz.viser_primitives import visualize_pointcloud
                        visualize_pointcloud(vis, "/live/cloud", cloud.points,
                                             color=cloud.colors, size=0.004)
                    last_cloud = time.time()
        except Exception as e:                   # noqa: BLE001 - ticker must never die
            print(f"[ticker] {type(e).__name__}: {e}")
            time.sleep(1.0)


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    console.add_target_arg(ap)
    ap.add_argument("--source", choices=["graspgenx", "sim_cloud"], default="graspgenx",
                    help="grasp source (DEFAULT graspgenx -- the standard, sim AND real; "
                         "sim_cloud = rarely-used cube-only sim GT fallback)")
    ap.add_argument("--segment", choices=["auto", "interactive", "none"], default="interactive",
                    help="SAM3 mask mode (DEFAULT interactive: cv2 prompt GUI; "
                         "auto = headless default_prompt)")
    ap.add_argument("--collision-world", action="store_true",
                    help="start with the depth-ESDF collision world ON")
    ap.add_argument("--offline", action="store_true",
                    help="GUI with no DDS/robot (motion tools return an error Result)")
    ap.add_argument("--no-cloud", action="store_true",
                    help="start with the live camera cloud ticker OFF")
    args = ap.parse_args()

    if args.offline:
        robot = g1.Robot.offline(args.target, camera=False)
    else:
        robot = g1.connect(args.target)
    if args.source:
        robot.set_grasp_source(args.source)
    if args.segment is not None:
        robot.set_segmenter(args.segment)
    robot.set_visualize(True)                    # builds GraspViz -> OUR viser server
    if args.collision_world:
        robot.set_collision_world(True)
    vis = robot.grasp_source.viz.vis

    log = ToolLog()
    runner = ToolRunner(log)
    ui = UI(robot, vis, runner, log, live_cloud_default=not args.no_cloud)
    log.add(f"ready ({robot.target}{', OFFLINE' if args.offline else ''}) -- tools: "
            + ", ".join(sorted(ui.tools)))

    stop = threading.Event()
    threading.Thread(target=ticker, args=(robot, vis, ui, runner, stop),
                     daemon=True).start()
    print("playground up -- open the viser URL above (Ctrl-C to exit)")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nshutting down (controllers keep their hold pose; e-stop is the "
              "hard stop)")
        stop.set()
        robot.close()


if __name__ == "__main__":
    main()
