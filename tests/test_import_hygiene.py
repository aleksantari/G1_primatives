"""Layering guard: the perception layer (and the base package) must import WITHOUT pulling
torch / cuRobo / unitree_sdk2py into the process. Heavy deps live behind the motion layer
and lazy factory builders only -- an agent host importing the API should not pay a CUDA
startup, and the GPU-free test suite must stay GPU-free. Runs in a subprocess so this
test's own imports can't contaminate the measurement."""
import subprocess
import sys

HEAVY = ("torch", "curobo", "unitree_sdk2py", "cv2")

# modules that must stay heavy-free (cv2 exempted where noted: segment_gui is display-bound
# by design and imports cv2 guardedly at module scope)
CLEAN_MODULES = [
    "g1_primitives",
    "g1_primitives.spatial.pose",
    "g1_primitives.spatial.pointcloud",
    "g1_primitives.perception.base",
    "g1_primitives.perception.frames",
    "g1_primitives.perception.depth",
    "g1_primitives.perception.segment",
    "g1_primitives.perception.sam3_client",
    "g1_primitives.grasp.base",
    "g1_primitives.grasp.tool_transform",
    "g1_primitives.grasp.graspgenx_client",
    "g1_primitives.ee.hand_base",
    "g1_primitives.motion.planner_base",
]


def _probe(module: str, banned: tuple) -> str:
    code = (
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        f"hit = [m for m in {banned!r} if m in sys.modules]\n"
        "print(','.join(hit))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"import of {module} failed:\n{out.stderr}"
    return out.stdout.strip()


def test_perception_and_base_layers_import_without_heavy_deps():
    for mod in CLEAN_MODULES:
        hit = _probe(mod, HEAVY)
        assert not hit, f"{mod} pulled heavy deps: {hit}"


def test_motion_planner_module_scope_is_torch_free():
    # curobo_planner defers torch/cuRobo into methods (the FakeMP/__new__ test pattern
    # depends on this) -- importing the MODULE must stay cheap.
    hit = _probe("g1_primitives.motion.curobo_planner", ("torch", "curobo"))
    assert not hit, f"motion.curobo_planner pulled {hit} at module scope"
