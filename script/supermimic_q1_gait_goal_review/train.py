"""Compose frozen GAIT_GOAL01 training with corrected, training-only reporting."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_gait_goal"), str(HERE.parent / "project_tools")]
from evidence import candidate_rank, forward_metrics, physical_decision


def fixed_metrics(trace, rows, reference, repeat, label):
    import torch
    import trial_support as native
    result = native.fixed_metrics(trace, rows, reference, repeat, label)
    for metrics in result:
        index = next(i for i, row in enumerate(rows) if row["name"] == metrics["clip"])
        count = int(trace["steps"][index])
        start = reference.motion.clip_start[index]
        full_frames = start + torch.arange(1, int(trace["reference_steps"][index]) + 1,
                                            device=reference.motion.device)
        root = reference.motion.root_p[full_frames, :2].detach().cpu().numpy()
        metrics.update(forward_metrics(root, trace["states"][count - 1, index, :2],
                                       trace["initial"][index, :2]))
        observed = int(reference.swing[full_frames[:count]].sum())
        metrics["swing_sole_observed_foot_frames"] = observed
        if not observed:
            metrics["swing_sole_z_rmse_m"] = None
    return result


def evaluate_trial(env, policy, head, thresholds, reference, rows, output, label, seconds,
                   progress=lambda **values: None, trained=False):
    import numpy as np
    import trial_support as native
    metrics, summaries, paths = [], [], []
    for repeat in range(3):
        progress(phase="evaluation", label=label, repeat=repeat, frame=0)
        trace = native.rollout(env, policy, head, thresholds, reference,
            mode="trained075" if trained else "static075", upper_alpha=.5,
            progress=lambda frame: progress(phase="evaluation", label=label, repeat=repeat, frame=frame))
        path = output / f"{label}_repeat{repeat}.npz"
        np.savez_compressed(path, **trace)
        paths.append(native.identity(path))
        summary = native.summarize(trace, rows)
        fixed = fixed_metrics(trace, rows, reference, repeat, label)
        summary.update(repeat=repeat, training_seconds=seconds, label=label, goal_fixed=fixed)
        native.save(path.with_suffix(".json"), summary)
        metrics.extend(fixed)
        summaries.append(summary)
    native.save(output / f"{label}_goal_fixed.json", metrics)
    return metrics, summaries, paths


def load_training_module(runtime):
    sys.path[:0] = [str(runtime / "src"), str(runtime)]
    original_path = HERE.parent / "supermimic_q1_gait_goal/train.py"
    spec = importlib.util.spec_from_file_location("frozen_goal_training", original_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from contact_reference import configure
    from diagnose_q1_upper_reference import policy_digest
    from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
    from unisupermimic.rl.algos.supermimic.learner import _empty_stats
    return module


def run(args):
    import trial_support as native
    module = load_training_module(args.runtime)
    original_path = HERE.parent / "supermimic_q1_gait_goal/train.py"
    selected = None
    def save(path, value):
        nonlocal selected
        if path.name == "checkpoint_selection.json":
            for candidate in value["candidates"]:
                candidate["review_rank"] = candidate_rank(candidate)
            selected = min(value["candidates"], key=candidate_rank)
            value["numeric_recommendation"] = selected
            value["rank_protocol"] = "Full completion/10% guards/progress/switching before metric gains; initial fallback"
        elif path.parent.name == "evaluation" and path.name == "receipt.json":
            if selected is None:
                raise AssertionError("Reviewed checkpoint selection must precede delivery receipt")
            value["recommended_traces"] = [row["path"] for row in selected["traces"]]
            value["reviewed_training_entrypoint"] = native.identity(__file__)
            value["frozen_training_kernel"] = native.identity(original_path)
        native.save(path, value)
    # Only the reporting callbacks change. Native setup, collection and updates
    # run from the byte-identical, preflight-frozen training module.
    module.evaluate_trial = evaluate_trial
    module.physical_decision = physical_decision
    module.save = save
    module.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--line", choices=("base", "ppo"), required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args()
    if args.check_imports:
        load_training_module(args.runtime)
        print(json.dumps(dict(status="FROZEN_TRAINING_IMPORTS_PASSED", new_physical_steps=0,
                              policy_inference_calls=0, optimizer_updates=0)))
    else:
        run(args)
