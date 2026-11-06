"""Preregistered exploratory noise-risk fit on saved upper reference angles.

The earlier test was already inspected. Selection here still uses validation
only, but reused test results are explicitly exploratory. The raw-history
Jacobian includes the latest-pose output anchor as well as feature differences.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from forecast_data import identity, load_plan
from upper_forecast import (BASELINES, CHANNELS, LAMBDAS, LocalBudget,
                            SavedUpperCorpus, collect, history_features,
                            predict, save, score)
from upper_forecast_sensitivity import (MODEL_SHA, baseline_affine_map,
                                       checked_model, learned_affine_map,
                                       noise_gains)


TRAIN_NOISE = .01
VALIDATION_NOISE = (0., .01, .02)
STRESS_CASES = ("clean", "iid_0.01", "iid_0.02", "ar1_0.8_0.01",
                "window_bias_0.01", "last5_frozen")
PRIOR = "previous_selected_ridge_0.01"


def specification():
    return dict(schema="q1_upper_noise_robust_preregistration_v1",
        history_frames=25, future_frames=10, channels=list(CHANNELS), fps=50,
        max_train_windows=8192, max_evaluation_windows_per_split=2048,
        sampling="source_balanced_uniform_cutoffs_v1",
        normalization="clean train inputs only, std floor1e-6rad",
        training_input_iid_noise_std_rad=TRAIN_NOISE,
        training_noise_objective="exact expected squared prediction error including noisy latest anchor",
        ridge_mean_loss_lambdas=list(LAMBDAS), bias_regularized=False,
        prior_model_sha256=MODEL_SHA, prior_model_refitted=False,
        baseline_methods=list(BASELINES), slope_frames=5, damped_tau_s=.12,
        validation_iid_noise_std_rad=list(VALIDATION_NOISE),
        selection="smallest equal mean over3 validation iid-noise levels and10 horizon RMSEs",
        validation_noise_integration="exact affine covariance; no noisy label sampling",
        tie_break="baseline order then previous model then ascending new ridge lambda",
        test_cases=list(STRESS_CASES), test_noise_seed=2026110602,
        ar1_marginal_std_rad=.01, ar1_rho=.8, bias_std_rad=.01,
        last5_frozen="repeat clean frame19 into20:25; timestamp/availability not hidden",
        test_reuse="exploratory; same128 source clips were inspected in prior studies",
        test_used_for_selection=False, test_only_recordings_used=False,
        clipping_smoothing_or_time_warp=False,
        maximum_seconds=90, maximum_rss_bytes=1024**3,
        local_cpu_only=True, robot_control=False, original_training_modified=False)


def raw_components(scale):
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != (200,) or not np.isfinite(scale).all() or (scale < 1e-6).any():
        raise ValueError("Require finite physical training feature scales")
    transform = np.eye(200)
    transform[192:, :192] -= np.tile(np.eye(8), (1, 24))
    design_jacobian = np.zeros((200, 201))
    design_jacobian[:, :200] = transform / scale[None, :]
    anchor = np.zeros((200, 80))
    for joint in range(8):
        anchor[192+joint, np.arange(10)*8+joint] = 1.
    return design_jacobian, anchor


def fit_noise_ridges(history, future, *, split):
    if split != "train":
        raise ValueError("Only train sources may fit noise-risk models or normalization")
    features = history_features(history)
    future = np.asarray(future, dtype=np.float64)
    if (future.shape != (len(history), 10, 8) or not len(history)
            or not np.isfinite(future).all()):
        raise ValueError("Require complete finite future labels on training windows")
    mean, scale = features.mean(axis=0), np.maximum(features.std(axis=0), 1e-6)
    design = np.column_stack(((features-mean)/scale, np.ones(len(features))))
    targets = (future-history[:, -1, None]).reshape(len(history), 80)
    gram = np.einsum("ni,nj->ij", design, design, optimize=False)/len(design)
    rhs = np.einsum("ni,nj->ij", design, targets, optimize=False)/len(design)
    jacobian, anchor = raw_components(scale)
    gram += TRAIN_NOISE**2*np.einsum("ki,kj->ij", jacobian, jacobian, optimize=False)
    rhs -= TRAIN_NOISE**2*np.einsum("ki,kj->ij", jacobian, anchor, optimize=False)
    penalty = np.eye(201)
    penalty[-1, -1] = 0.
    models = {}
    for strength in LAMBDAS:
        weights = np.linalg.solve(gram+strength*penalty, rhs)
        if not np.isfinite(weights).all():
            raise ValueError("Nonfinite noise-regularized forecast")
        models["noise_ridge_"+str(strength)] = dict(weights=weights, mean=mean, scale=scale,
            lambda_mean_loss=strength, fitted_split="train", training_windows=len(history),
            input_iid_noise_std_rad=TRAIN_NOISE, latest_anchor_in_noise_risk=True)
    return models


def jacobian_for(method, models):
    return learned_affine_map(models[method])[0] if method in models else baseline_affine_map(method)[0]


def expected_noise_profile(clean_profile, jacobian, std):
    clean = np.asarray(clean_profile, dtype=np.float64)
    if (clean.shape != (10,) or not np.isfinite(clean).all() or (clean < 0.).any()
            or not np.isfinite(std) or std < 0.):
        raise ValueError("Require finite nonnegative RMSE and physical noise standard deviation")
    gains = np.asarray(noise_gains(jacobian)["per_horizon_rms_std_gain"])
    return np.sqrt(clean*clean + std*std*gains*gains)


def observe(history, case, *, seed=2026110602):
    history_features(history)
    if case not in STRESS_CASES:
        raise ValueError("Unknown preregistered stress case")
    values = np.array(history, dtype=np.float64, copy=True)
    rng = np.random.default_rng(seed)
    if case.startswith("iid_"):
        values += rng.normal(0., float(case.split("_")[1]), values.shape)
    elif case == "ar1_0.8_0.01":
        noise = rng.normal(0., .01, values.shape)
        for frame in range(1, 25):
            noise[:, frame] = .8*noise[:, frame-1] + np.sqrt(1.-.8**2)*noise[:, frame]
        values += noise
    elif case == "window_bias_0.01":
        values += rng.normal(0., .01, (len(values), 1, 8))
    elif case == "last5_frozen":
        values[:, 20:] = values[:, 19:20]
    return values


def evaluate(plan, corpus, prior_model, *, train_limit=8192, eval_limit=2048,
             budget_check=lambda: None, on_selection=lambda value, models: None):
    train, labels, train_items = collect(plan, corpus, "train", train_limit, budget_check)
    budget_check()
    fitted = fit_noise_ridges(train, labels, split="train")
    models = {PRIOR: prior_model, **fitted}
    methods = (*BASELINES, *models)
    validation, targets, val_items = collect(plan, corpus, "validation", eval_limit, budget_check)
    reports = []
    for method in methods:
        budget_check()
        row = score(predict(validation, method, models), targets, val_items,
                    method, "validation", 0.)
        jacobian = jacobian_for(method, models)
        levels = [dict(iid_noise_std_rad=std, per_horizon_rmse_rad=expected_noise_profile(
                    row["per_horizon_rmse_rad"], jacobian, std).tolist())
                  for std in VALIDATION_NOISE]
        row.update(expected_noise_levels=levels,
            selection_metric_rad=float(np.mean([item["per_horizon_rmse_rad"] for item in levels])))
        reports.append(row)
    selected = min(reports, key=lambda row: row["selection_metric_rad"])
    new_selected = min((row for row in reports if row["method"] in fitted),
                       key=lambda row: row["selection_metric_rad"])
    selection = dict(method=selected["method"], best_new_learned_method=new_selected["method"],
        value_rad=selected["selection_metric_rad"], metric=specification()["selection"],
        fitted_split="train", selection_split="validation", test_labels_read=False,
        test_used_for_selection=False, reused_test_exploratory=True, methods=list(methods))
    on_selection(selection, models)
    testing, targets, test_items = collect(plan, corpus, "test", eval_limit, budget_check)
    test = []
    for case in STRESS_CASES:
        observed = observe(testing, case)
        for method in methods:
            budget_check()
            row = score(predict(observed, method, models), targets, test_items,
                method, "test", float(case.split("_")[1]) if case.startswith("iid_") else None)
            row.update(stress_case=case, reused_test_exploratory=True,
                       selection_allowed=False)
            test.append(row)
    samples = {}
    for split, items in (("train", train_items), ("validation", val_items), ("test", test_items)):
        samples[split] = dict(windows=len(items), sources=len({item[0] for item in items}),
            items_sha256=hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest())
    return dict(schema="q1_upper_noise_robust_exploration_v1", selection=selection,
        sampling=samples, validation=reports, test=test,
        jacobian_noise_gains={method: noise_gains(jacobian_for(method, models)) for method in methods},
        new_model_fits=len(fitted), prior_model_refitted=False, gradient_optimizer_updates=0,
        test_reused_from_previous_experiment=True, independent_confirmation=False,
        actual_robot_angles_used=False, test_only_recordings_used=False,
        robot_policy_inference=0, physics_steps=0, cuda_jobs=0,
        measured_robot_latency_ms=None, naturalness_conclusion=None,
        original_training_modified=False, user_accepted_checkpoint=None)


def execute(args):
    if not args.local_cpu_study or sys.platform != "darwin":
        raise ValueError("Explicit local Mac saved-angle study only; not a server CPU fallback")
    budget = LocalBudget()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(args.project_root.resolve()/"data/supermimic_for_q1"):
        raise ValueError("Use a new independent output root")
    preregistration = identity(args.preregistration)
    if json.loads(args.preregistration.read_text()) != specification():
        raise ValueError("Noise-risk preregistration differs from fixed implementation")
    prior_id = identity(args.prior_model)
    if prior_id["sha256"] != MODEL_SHA or prior_id["bytes"] > 1024**2:
        raise ValueError("Require the exact previously selected unmodified ridge model")
    prior = checked_model(json.loads(args.prior_model.read_text()))
    plan_id = identity(args.plan)
    plan = load_plan(args.admission, args.sources, args.plan)
    corpus = SavedUpperCorpus(plan, json.loads(args.admission.read_text()),
                              args.review_root, args.delivery_manifest)
    output.mkdir(parents=True)
    status = dict(status="RUNNING", started_utc=datetime.now(timezone.utc).isoformat(),
        code_identity=identity(__file__), preregistration_identity=preregistration,
        prior_model_identity=prior_id, plan_identity=plan_id)
    save(output/"status.json", status)

    def selected(value, models):
        save(output/"validation_selection.json", value)
        model = models[value["best_new_learned_method"]]
        saved = {key: item.tolist() if isinstance(item, np.ndarray) else item
                 for key, item in model.items()}
        save(output/"selected_new_model.json", dict(schema="q1_upper_noise_robust_model_v1",
            channels=CHANNELS, history_frames=25, future_frames=10,
            plan_identity=plan_id, model=saved, selection=value,
            deployment_authorized=False, reused_test_exploratory=True))

    try:
        result = evaluate(plan, corpus, prior, budget_check=budget.check, on_selection=selected)
        if (identity(args.plan) != plan_id or identity(args.preregistration) != preregistration
                or identity(args.prior_model) != prior_id):
            raise ValueError("Frozen study inputs changed")
        result.update(status, status="COMPLETED", final_resources=budget.check(),
            input_identities=corpus.verify_unchanged(),
            delivery_manifest=corpus.manifest_identity,
            original_policy_trained_on_all_393=plan["original_policy_trained_on_all_393"])
        save(output/"result.json", result)
        save(output/"status.json", dict(status, status="COMPLETED",
            result=identity(output/"result.json")), replace=True)
        return result
    except BaseException as error:
        save(output/"failure.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}",
                                         elapsed_seconds=time.monotonic()-budget.started))
        save(output/"status.json", dict(status, status="FAILED"), replace=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "admission", "sources", "review-root", "delivery-manifest",
                 "preregistration", "prior-model", "project-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--local-cpu-study", action="store_true")
    result = execute(parser.parse_args())
    print(json.dumps(dict(status=result["status"], selection=result["selection"],
                          resources=result["final_resources"]), allow_nan=False))
