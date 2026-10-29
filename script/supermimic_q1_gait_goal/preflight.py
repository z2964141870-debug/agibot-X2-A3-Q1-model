"""New base/PPO preflights with real native gradients and zero optimizer steps."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def run(a):
    sys.path[:0] = [str(HERE), str(a.runtime / "src"), str(a.runtime)]
    from trial_support import CandidateAdapter, candidate_prepare, base_learner, reward_types, rollout
    from trial_support import check_contract, save, identity, stamp
    if a.line == "ppo":
        path = HERE.parent / "supermimic_q1_contact_comp/preflight.py"
        spec = importlib.util.spec_from_file_location("goal_new_ppo_preflight", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Adapter(CandidateAdapter):
            def __init__(self, env, teacher, thresholds, upper_alpha=.5):
                super().__init__(env, teacher, thresholds, upper_alpha, runtime=a.runtime)
        module.Adapter = Adapter
        _, Reference = reward_types(a.runtime)
        module.ContactReference = lambda mdp: Reference(mdp, a.runtime / "q1_assets/scene.xml")
        module.run(a)
        return
    import torch
    from base import preserve_hp
    from physical_gate import FIXED
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(status="INITIALIZING", line="base", envs=a.envs, optimizer_steps=0, started_utc=stamp())
    env = independent = None
    try:
        torch.set_num_threads(8)
        _, env, cfg = candidate_prepare(a, count=a.envs, base_candidate=True)
        result["physical_contract"] = check_contract(env)
        learner = base_learner(env, cfg, a.output, a.checkpoint)
        with torch.no_grad():
            learner.collect(16)
        original = {name: {key: value.clone() for key, value in getattr(learner, name).state_dict().items()}
                    for name in ("wm", "pi", "critic")}
        sample = learner._sample_valid_starts
        captured = {}
        def same_replay_start(count):
            start, history = sample(count)
            for key in start:
                torch.testing.assert_close(start[key], history[key][:, -1], atol=0, rtol=0)
            captured["inputs"] = [value.clone() for value in (*start.values(), *history.values())]
            captured["references"] = [*start.values(), *history.values()]
            return start, history
        learner._sample_valid_starts = same_replay_start
        steps = {name: 0 for name in ("wm", "pi", "v")}
        for name in steps:
            optimizer = getattr(learner, "opt_" + name)
            optimizer.step = lambda *values, _name=name, **options: steps.__setitem__(_name, steps[_name] + 1)
        learner.wm_update()
        learner.policy_update()
        for before, after in zip(captured["inputs"], captured["references"]):
            torch.testing.assert_close(before, after, atol=0, rtol=0)
        gradient_norms = {}
        for name in original:
            module = getattr(learner, name)
            gradient_norms[name] = sum(float(parameter.grad.float().square().sum())
                for parameter in module.parameters() if parameter.grad is not None)**.5
            if not gradient_norms[name] > 0:
                raise AssertionError("Native update produced no gradient: " + name)
            for key, value in module.named_parameters():
                torch.testing.assert_close(value, original[name][key], atol=0, rtol=0)
                if value.grad is not None and not bool(torch.isfinite(value.grad).all()):
                    raise AssertionError("Nonfinite native candidate gradient")
            module.load_state_dict(original[name])
        if steps != dict(wm=1, pi=1, v=20):
            raise AssertionError("Native update did not exercise the expected blocked optimizer calls")
        rows = [row for split in json.loads(a.library.read_text())["splits"].values()
                for row in split if row["name"] in FIXED]
        if len(rows) != 3:
            raise ValueError("Original fixed3 clips missing")
        subset = a.output / "evaluation_library.json"
        save(subset, dict(splits=dict(train=rows)))
        with preserve_hp(learner):
            _, independent, _ = candidate_prepare(a, count=3, evaluation=True, library=subset)
            _, Reference = reward_types(a.runtime)
            reference = Reference(independent.supermimic_mdp, a.runtime / "q1_assets/scene.xml")
            thresholds = json.loads(a.calibration.read_text())["thresholds"]
            trace = rollout(independent, learner.pi, None, thresholds, reference, mode="teacher", upper_alpha=0.)
            if not bool(torch.isfinite(torch.from_numpy(trace["states"])).all()):
                raise AssertionError("Nonfinite independent base evaluation")
        result.update(status="GOAL_BASE_NATIVE_ZERO_OPTIMIZER_PREFLIGHT_PASSED", completed_utc=stamp(),
            blocked_optimizer_calls=steps, parameters_unchanged=True, normalizers_restored=True,
            wm_batch=learner.cfg.wm_batch, actor_batch=learner.cfg.pi_batch, horizon=learner.cfg.horizon,
            compile=learner.cfg.compile, checkpoint=identity(a.checkpoint),
            h32_replay_start_exact_and_unmutated=True, finite_nonzero_gradient_norms=gradient_norms,
            cross_environment_physical_direction_established=False,
            independent_evaluation_preserved_live_state_rng=True,
            native_collection_transitions=17 * a.envs,
            independent_evaluation_transitions=int(trace["states"].shape[0] * 3),
            candidate_contract=env.supermimic_mdp.phase_contract,
            native_reward_maximum=env.supermimic_mdp.max_step_reward)
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if independent is not None:
            independent.close()
            independent.backend.close()
        if env is not None:
            env.close()
            env.backend.close()
        save(a.output / "receipt.json", result)
    print(result["status"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--line", choices=("base", "ppo"), required=True)
    parser.add_argument("--envs", type=int, choices=(16, 4096), required=True)
    parser.add_argument("--upper-alpha", type=float, default=.5, choices=(.5,))
    run(parser.parse_args())
