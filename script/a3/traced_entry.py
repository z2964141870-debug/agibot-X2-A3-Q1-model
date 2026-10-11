"""Instrument the pinned upstream entry in memory; vendor files stay read-only."""

import ast
import faulthandler
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys

from script.a3.phase_trace import trace, wrap

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "script/vendor/sonic_for_a3"
TARGETS = {
    "isaaclab.sim.simulation_context": {"SimulationContext":{"__init__":"SIM_CONTEXT_CREATE", "step":"PHYSICS_STEP"}},
    "isaaclab.scene.interactive_scene": {"InteractiveScene":{"__init__":"SCENE_CREATE", "update":"SCENE_STATE_UPDATE", "write_data_to_sim":"SCENE_ACTION_WRITE"}},
    "isaaclab.envs.manager_based_rl_env": {"ManagerBasedRLEnv":{"__init__":"ISAAC_ENV_CREATE", "step":"ISAAC_ENV_STEP"}},
    "gear_sonic.envs.wrapper.manager_env_wrapper": {"ManagerEnvWrapper":{"__init__":"ENV_WRAPPER_CREATE", "step":"ACTION_TO_ENV", "reset_all":"ENV_RESET"}},
    "gear_sonic.utils.motion_lib.motion_lib_base": {"MotionLibBase":{"__init__":"MOTIONLIB_CREATE", "load_data":"MOTION_DATA_READ", "load_motions":"MOTION_LOAD"}},
}


class TraceLoader(importlib.abc.Loader):
    def __init__(self, loader, fullname):
        self.loader, self.fullname = loader, fullname

    def create_module(self, spec):
        method = getattr(self.loader, "create_module", None)
        return method(spec) if method else None

    def exec_module(self, module):
        with trace().span("IMPORT_" + self.fullname, cuda_sync=False):
            self.loader.exec_module(module)
        hooks = []
        for clsname, methods in TARGETS[self.fullname].items():
            cls = getattr(module, clsname)
            for method, phase in methods.items():
                setattr(cls, method, wrap(getattr(cls, method), phase))
                hooks.append(clsname + "." + method)
        trace().emit("HOOKS_READY", self.fullname, hooks=hooks)


class TraceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        # Delegate without recursing into this finder; preserve normal finder order.
        for finder in sys.meta_path:
            if finder is self:
                continue
            method = getattr(finder, "find_spec", None)
            spec = method(fullname, path, target) if method else None
            if spec is not None:
                if spec.loader is None:
                    raise RuntimeError("Target has no executable loader: " + fullname)
                spec.loader = TraceLoader(spec.loader, fullname)
                return spec
        raise ImportError("Trace target not found: " + fullname)


class EntryTransformer(ast.NodeTransformer):
    PHASES = {"launch_isaac_app_with_lock":"ISAAC_APP_INIT", "create_manager_env":"ENV_CREATE"}
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name in self.PHASES:
            node.decorator_list.append(ast.Call(func=ast.Name(id="_yuanqi_trace_decorate", ctx=ast.Load()),
                                                args=[ast.Constant(self.PHASES[node.name])], keywords=[]))
        return node


def main():
    directory = Path(os.environ["YUANQI_PHASE_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    # Native crashes may reach faulthandler; power/reset usually cannot.
    fault = (directory / f"faulthandler.{os.getpid()}.log").open("a")
    faulthandler.enable(file=fault, all_threads=True)
    finder = TraceFinder()
    sys.meta_path.insert(0, finder)
    entry = VENDOR / "gear_sonic/train_agent_trl.py"
    sys.path.insert(0, str(VENDOR.resolve()))
    sys.argv[0] = str(entry)
    with trace().span("UPSTREAM_ENTRY", cuda_sync=False):
        tree = EntryTransformer().visit(ast.parse(entry.read_text(), filename=str(entry)))
        tree.body.insert(0, ast.ImportFrom(module="script.a3.phase_trace", names=[
            ast.alias(name="decorate", asname="_yuanqi_trace_decorate")], level=0))
        ast.fix_missing_locations(tree)
        namespace = dict(__name__="__main__", __file__=str(entry), __package__=None)
        exec(compile(tree, str(entry), "exec"), namespace)


if __name__ == "__main__":
    main()
