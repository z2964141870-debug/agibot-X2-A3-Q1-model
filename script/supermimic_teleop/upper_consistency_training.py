"""Paired-origin MPS control versus fixed temporal-consistency regularization.

Only the regularizer differs between the two new fits. Both see the same noisy
26-frame histories, two target origins, minibatches, initialization and budget.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from forecast_data import identity, load_plan, window_items
from upper_forecast import SavedUpperCorpus, collect, save, score
from upper_mps_forecast import (MPSBudget, audit_cpu_tensors, cpu_tree, feature_tensor,
                                make_model, prediction, require_mps)
from upper_noise_robust import STRESS_CASES, observe
from upper_rolling_review import (MODEL_SHA, SCENARIOS, dense_windows, fixed_sources,
                                  metrics, observations)


SEED = 2026110606
WEIGHTS = (0., .5)


def specification():
    return dict(schema="q1_upper_consistency_training_plan_v1",
        model="same94672parameter MLP200-256-128-80,SiLU,zero output head",
        initialization_seed=SEED, regularizer_weights=list(WEIGHTS),
        controlled_difference="only squared same-absolute-target revision weight",
        pair_history_frames=26, origins=[0, 1], targets_per_origin=10,
        sampling="existing source-balanced sampler with future_frames11",
        train_pairs=8192, validation_windows=2048, test_windows=2048,
        train_noise="one iid0.01rad26frame stream shared by both overlapping histories",
        normalization="reuse exact previous train-only MLP normalization",
        normalization_model_sha256=MODEL_SHA,
        loss="mean squared angle error over both origins plus weight*mean squared nine shared-target revisions",
        optimizer="Adam", learning_rate=.001, batch_pairs=128,
        maximum_updates_per_model=1000, gradient_norm_limit=1.,
        checkpoint_every_updates=100,
        validation="equal mean0/0.01/0.02rad noise and ten horizon RMSE;each model selected separately",
        validation_noise_seed=2026110604,
        all_model_selections_persisted_before_test=True,
        test_cases=list(STRESS_CASES), test_noise_seed=2026110602,
        dense_scenarios=list(SCENARIOS), fixed_three_test_clips=True,
        no_winner_selected_on_test=True, reused_test_exploratory=True,
        backend="local Apple MPS only", maximum_total_seconds=600,
        maximum_training_seconds=500, maximum_rss_bytes=1024**3,
        maximum_mps_driver_bytes=1024**3, minimum_system_free_percent=25,
        maximum_output_bytes=32*1024**2, independent_local_mps_lock=True,
        robot_policy_physics_or_server_access=False)


def collect_pairs(plan, corpus, limit, check):
    pair_plan = dict(plan, future_frames=11)
    items = window_items(pair_plan, "train", limit)
    histories = np.empty((len(items), 26, 8))
    targets = np.empty((len(items), 11, 8))
    previous = None
    for index, (name, cutoff) in enumerate(items):
        if name != previous:
            check()
            values = corpus.load(name, "train")
            previous = name
        if not 24 <= cutoff < len(values)-11:
            raise ValueError("Paired origins cross a source boundary")
        histories[index] = values[cutoff-24:cutoff+2]
        targets[index] = values[cutoff+1:cutoff+12]
    return histories, targets, items


def paired_loss(torch, forecast, targets, weight):
    if forecast.shape != targets.shape or tuple(forecast.shape[1:]) != (2, 10, 8) or weight not in WEIGHTS:
        raise ValueError("Require two aligned origins and a fixed regularizer weight")
    tracking = torch.mean((forecast-targets)**2)
    revision = torch.mean((forecast[:, 1, :-1]-forecast[:, 0, 1:])**2)
    return tracking+weight*revision, tracking, revision


def fit_pair(torch, histories, targets, validation, val_targets, val_items, normalization,
             weight, budget):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = make_model(torch).to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    val_inputs = [validation if std == 0. else observe(validation, "iid_"+str(std), seed=2026110604)
                  for std in (0., .01, .02)]
    best, trace = None, []

    def validate(step, losses):
        nonlocal best
        model.eval()
        profiles = [score(prediction(torch, model, values, normalization), val_targets,
            val_items, "paired_mlp", "validation", None)["per_horizon_rmse_rad"] for values in val_inputs]
        risk = float(np.mean(profiles))
        if not np.isfinite(risk):
            raise ValueError("Nonfinite validation risk")
        trace.append(dict(update=step, losses=losses, validation_risk_rad=risk, resources=budget.check()))
        if best is None or risk < best["risk"]:
            best = dict(update=step, risk=risk, state=cpu_tree(torch, model.state_dict()))
        model.train()

    validate(0, None)
    for step in range(1, 1001):
        if time.monotonic()-budget.started >= 500:
            raise RuntimeError("Paired training total time budget reached")
        selected = rng.integers(0, len(histories), 128)
        observed = histories[selected]+rng.normal(0., .01, (128, 26, 8))
        pair = np.stack((observed[:, :-1], observed[:, 1:]), axis=1)
        flat = pair.reshape(256, 25, 8)
        features = feature_tensor(torch, flat, normalization, "mps")
        anchor = torch.as_tensor(flat[:, -1, None], dtype=torch.float32, device="mps")
        forecast = (anchor+model(features).reshape(256, 10, 8)).reshape(128, 2, 10, 8)
        labels = np.stack((targets[selected, :-1], targets[selected, 1:]), axis=1)
        loss, tracking, revision = paired_loss(torch, forecast,
            torch.as_tensor(labels, dtype=torch.float32, device="mps"), weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses = [float(value.detach().cpu()) for value in (loss, tracking, revision)]
        if not np.isfinite(losses).all():
            raise ValueError("Nonfinite actual paired-origin training loss")
        if step % 100 == 0:
            validate(step, losses)
        elif step % 50 == 0:
            budget.check()
    counters = [int(state["step"].item()) for state in optimizer.state.values()]
    if counters != [1000]*6:
        raise ValueError("Actual paired Adam steps differ from completed updates")
    last = cpu_tree(torch, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), completed_updates=1000))
    model.load_state_dict(best["state"])
    model.eval()
    return model, best, last, trace, counters


def execute(args):
    if not args.local_mps_study or sys.platform != "darwin":
        raise ValueError("Explicit independent local Apple MPS training only")
    root, output, lock = args.project_root.resolve(), args.output.resolve(), args.gpu_lock.resolve()
    if output.exists() or not output.is_relative_to(root/"data/supermimic_for_q1"):
        raise ValueError("Require a new independent paired experiment root")
    if not lock.is_relative_to(root/"data/locks"):
        raise ValueError("Require the independent local MPS lock directory")
    frozen = {key: identity(getattr(args, key)) for key in ("preregistration", "normalization_model", "plan")}
    if (frozen["normalization_model"]["sha256"] != MODEL_SHA
            or json.loads(args.preregistration.read_text()) != specification()):
        raise ValueError("Fixed normalization or experiment options differ")
    budget = MPSBudget()
    budget.check()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch

        require_mps(torch)
        torch.set_num_threads(1)
        torch.mps.set_per_process_memory_fraction(.125)
        budget.torch = torch
        budget.check()
        saved = torch.load(args.normalization_model, map_location="cpu", weights_only=True)
        audit_cpu_tensors(torch, saved)
        if saved["plan_identity"] != frozen["plan"]:
            raise ValueError("Normalization belongs to a different source split")
        normalization = {key: value.numpy() for key, value in saved["normalization"].items()}
        plan = load_plan(args.admission, args.sources, args.plan)
        corpus = SavedUpperCorpus(plan, json.loads(args.admission.read_text()), args.review_root, args.delivery_manifest)
        output.mkdir(parents=True)
        save(output/"status.json", dict(status="RUNNING", started_utc=datetime.now(timezone.utc).isoformat()))
        try:
            train, labels, items = collect_pairs(plan, corpus, 8192, budget.check)
            val, val_targets, val_items = collect(plan, corpus, "validation", 2048, budget.check)
            fitted, training = {}, []
            for weight in WEIGHTS:
                budget.check()
                name = "pair_revision_"+str(weight)
                model, best, last, trace, counters = fit_pair(torch, train, labels, val,
                    val_targets, val_items, normalization, weight, budget)
                selection = dict(method=name, revision_weight=weight, selected_update=best["update"],
                    validation_risk_rad=best["risk"], actual_updates=1000, actual_adam_steps=counters,
                    selection_split="validation", test_labels_read=False, test_used_for_selection=False)
                checkpoint = dict(schema="q1_upper_mps_forecast_model_v1", model=best["state"],
                    normalization=saved["normalization"], plan_identity=frozen["plan"],
                    preregistration_identity=frozen["preregistration"], selection=selection)
                for suffix, value in (("selected.pt", checkpoint), ("last.pt", last)):
                    audit_cpu_tensors(torch, value)
                    with (output/(name+"_"+suffix)).open("xb") as stream:
                        torch.save(value, stream)
                    audit_cpu_tensors(torch, torch.load(output/(name+"_"+suffix), map_location="cpu", weights_only=True))
                save(output/(name+"_selection.json"), selection)
                save(output/(name+"_trace.json"), trace)
                fitted[name] = model
                training.append(dict(selection=selection, trace=trace,
                    checkpoint=identity(output/(name+"_selected.pt")), last_state=identity(output/(name+"_last.pt"))))
            # Neither test labels nor dense fixed clips are read until both selections exist.
            testing, targets, test_items = collect(plan, corpus, "test", 2048, budget.check)
            tests, dense = [], []
            for name, model in fitted.items():
                for case in STRESS_CASES:
                    budget.check()
                    report = score(prediction(torch, model, observe(testing, case), normalization),
                        targets, test_items, name, "test", None)
                    tests.append(dict(report, stress_case=case))
                for row in fixed_sources(plan):
                    budget.check()
                    reference = corpus.load(row["relative"], "test")
                    for scenario in SCENARIOS:
                        observed, support = observations(reference, scenario)
                        cutoffs, history, truth = dense_windows(reference, observed)
                        forecast = prediction(torch, model, history, normalization)
                        dense.append(dict(relative=row["relative"], scenario=scenario, method=name,
                            windows=len(cutoffs), max_source_age_ms=int((cutoffs-support[cutoffs]).max()*20),
                            **metrics(forecast, truth)))
            for key, expected in frozen.items():
                if identity(getattr(args, key)) != expected:
                    raise ValueError("Paired experiment input changed")
            result = dict(schema="q1_upper_consistency_training_v1", status="COMPLETED",
                code_identity=identity(__file__), fixed_inputs=frozen, training=training,
                train_pair_items_sha256=hashlib.sha256(json.dumps(items,separators=(",", ":")).encode()).hexdigest(),
                train_pairs=len(items), test=tests, dense=dense, source_identities=corpus.verify_unchanged(),
                resources=budget.check(), resource_observations=budget.observations,
                actual_mps_optimizer_updates=2000, per_model_updates=1000,
                reused_test_exploratory=True, test_used_for_selection=False, deployed=False,
                robot_policy_inference=0, physics_steps=0, server_access=False,
                measured_robot_latency_ms=None, naturalness_conclusion=None, user_accepted_checkpoint=None)
            save(output/"result.json", result)
            if sum(path.stat().st_size for path in output.iterdir()) > 32*1024**2:
                raise RuntimeError("Paired training output exceeded budget")
            save(output/"status.json", dict(status="COMPLETED", result=identity(output/"result.json")), replace=True)
            return result
        except BaseException as error:
            save(output/"failure.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}"))
            save(output/"status.json", dict(status="FAILED"), replace=True)
            raise
        finally:
            torch.mps.synchronize()
            torch.mps.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "output", "gpu-lock", "normalization-model", "preregistration",
                 "plan", "admission", "sources", "review-root", "delivery-manifest"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--local-mps-study", action="store_true")
    result = execute(parser.parse_args())
    print(json.dumps(dict(status=result["status"], training=[row["selection"] for row in result["training"]],
        resources=result["resources"]), allow_nan=False))
