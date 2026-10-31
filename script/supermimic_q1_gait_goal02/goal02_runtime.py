"""Bind independent GOAL02 rewards to the existing native trial entrypoints."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def install(runtime, output):
    sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_gait_goal"),
                   str(HERE.parent / "project_tools"), str(runtime / "src"), str(runtime)]
    import trial_support as native
    from coupled_rewards import CoupledMDP, CoupledReference
    from goal02_config import contract
    from ppo_diagnostics import PPODiagnostics

    native.reward_types = lambda unused: (CoupledMDP, CoupledReference)
    original_prepare, original_adapter, original_make = native.candidate_prepare, native.CandidateAdapter, native.make_ppo
    holder = {}

    def prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        native.save(Path(output) / "goal02_recipe.json", contract())
        return result

    class Adapter(original_adapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.diagnostic_steps = []
            holder["adapter"] = self

    def make_ppo(actor, obs, count):
        algorithm = original_make(actor, obs, count)
        adapter = holder["adapter"]
        adapter.foot_reference.record_diagnostics = True
        algorithm.goal02_diagnostics = PPODiagnostics(algorithm, adapter, output).attach()
        native.save(Path(output) / "goal02_recipe.json", contract())
        return algorithm

    native.candidate_prepare, native.CandidateAdapter, native.make_ppo = prepare, Adapter, make_ppo
    return native, make_ppo
