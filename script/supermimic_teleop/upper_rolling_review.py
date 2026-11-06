"""Dense rolling diagnostics of fixed saved upper predictors, without fitting.

This evaluates generated reference signals, not robot commands or hardware
latency. Dropouts keep an explicit source index; held samples are generated.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys
import time

import numpy as np

from forecast_data import identity, load_plan
from upper_forecast import DT, SavedUpperCorpus, history_features, predict, save
from upper_mps_forecast import (MPSBudget, audit_cpu_tensors, make_model,
                                prediction, require_mps)


MODEL_SHA = "7c40d8d6afda111879cb37f2dd43f33c7c494549f75689dea9afb11c67b2fccc"
RIDGE_SHA = "ad57ed43d33b659c34ac5dbf26c5f2bd1c6dcff4597985d927e15d07cad7b971"
FIXED = ("walk_forward_loop_001__a022", "walk_forward_start_002__a023",
         "walk_forward_stop_002__a024")
SCENARIOS = ("clean", "continuous_iid_0.01", "iid_0.01_periodic5_drop")
METHODS = ("hold", "damped_cv", "noise_ridge_1.0", "mps_mlp")


def specification():
    return dict(schema="q1_upper_rolling_review_plan_v1", fitting=False,
        fixed_clip_stems=list(FIXED), scenarios=list(SCENARIOS), methods=list(METHODS),
        history_frames=25, future_frames=10, dt_s=DT,
        dense_cutoffs="24 through frames-11 inclusive, no cross-clip windows",
        noise_seed=2026110605, noise_std_rad=.01,
        noise="one continuous frame stream shared across overlapping histories",
        drops="frame index modulo100 in50..54; repeat last arrived observation",
        provenance="retain last arrived source frame; held samples are generated",
        stale_hold_after_ms=40, stale_variants=["ungated", "hold_after_40ms"],
        revisions="new[t+1,h-1] minus previous[t,h], same absolute target frame",
        command_proxies_horizons_ms=[80, 100], physical_commands=False,
        timing="batch1 feature normalization, MPS transfer/forward, CPU materialization, synchronized",
        benchmark_warmup_calls=20, benchmark_measured_calls=200,
        selection_or_threshold_tuning=False, reused_test_exploratory=True,
        maximum_total_seconds=600, maximum_rss_bytes=1024**3,
        maximum_mps_driver_bytes=1024**3, minimum_memory_free_percent=25,
        maximum_output_bytes=8*1024**2, independent_local_mps_lock=True,
        original_server_access=False, robot_policy_or_physics=False)


def observations(reference, scenario):
    reference = np.asarray(reference, dtype=np.float64)
    if (reference.ndim != 2 or reference.shape[1] != 8 or len(reference) < 35
            or not np.isfinite(reference).all() or scenario not in SCENARIOS):
        raise ValueError("Require a complete finite eight-channel fixed clip")
    values = reference.copy()
    if scenario != "clean":
        values += np.random.default_rng(2026110605).normal(0., .01, values.shape)
    support = np.arange(len(values))
    if scenario == "iid_0.01_periodic5_drop":
        for frame in range(1, len(values)):
            if 50 <= frame % 100 <= 54:
                values[frame] = values[frame-1]
                support[frame] = support[frame-1]
    if (support > np.arange(len(values))).any():
        raise ValueError("Observation depends on an unavailable source")
    return values, support


def dense_windows(reference, observed):
    reference, observed = np.asarray(reference), np.asarray(observed)
    if (reference.shape != observed.shape or reference.ndim != 2
            or reference.shape[1] != 8 or len(reference) < 35
            or not np.isfinite(reference).all() or not np.isfinite(observed).all()):
        raise ValueError("Require matching complete finite reference and observations")
    cutoffs = np.arange(24, len(reference)-10)
    history = np.stack([observed[t-24:t+1] for t in cutoffs])
    target = np.stack([reference[t+1:t+11] for t in cutoffs])
    history_features(history)
    return cutoffs, history, target


def stale_gate(forecast, history, ages_frames):
    forecast = np.asarray(forecast)
    history_features(history)
    ages = np.asarray(ages_frames)
    if (forecast.shape != (len(history), 10, 8) or not np.isfinite(forecast).all()
            or ages.shape != (len(history),) or (ages < 0).any()
            or not np.issubdtype(ages.dtype, np.integer)):
        raise ValueError("Require finite forecasts and explicit nonnegative source ages")
    generated = ages > 2
    result = forecast.copy()
    result[generated] = history[generated, -1, None]
    return result, generated


def distribution(values):
    values = np.asarray(values)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Require nonempty finite diagnostic samples")
    absolute = np.abs(values)
    return dict(rms=float(np.sqrt(np.mean(values*values))),
        absolute_p95=float(np.quantile(absolute, .95)), absolute_max=float(absolute.max()))


def metrics(forecast, target):
    forecast, target = np.asarray(forecast), np.asarray(target)
    if (forecast.shape != target.shape or forecast.ndim != 3
            or forecast.shape[1:] != (10, 8) or len(forecast) < 2
            or not np.isfinite(forecast).all() or not np.isfinite(target).all()):
        raise ValueError("Require aligned dense finite forecasts and labels")
    errors = forecast-target
    # Adjacent origins share nine absolute target timestamps, with shifted slots.
    revisions = forecast[1:, :-1]-forecast[:-1, 1:]
    signals = []
    for horizon in (4, 5):
        signal, wanted = forecast[:, horizon-1], target[:, horizon-1]
        speed, true_speed = np.diff(signal, axis=0)/DT, np.diff(wanted, axis=0)/DT
        acceleration = np.diff(speed, axis=0)/DT
        signals.append(dict(horizon_ms=horizon*20,
            generated_signal_speed_rad_s=distribution(speed),
            generated_signal_acceleration_rad_s2=distribution(acceleration),
            target_speed_rad_s=distribution(true_speed),
            speed_error_rad_s=distribution(speed-true_speed)))
    return dict(per_horizon_rmse_rad=np.sqrt(np.mean(errors*errors, axis=(0, 2))).tolist(),
        error_rad=distribution(errors), same_target_revision_rad=distribution(revisions),
        generated_signal_proxies=signals, physical_robot_command=False)


def timing_review(call, synchronize):
    synchronize()
    started = time.perf_counter_ns()
    call()
    synchronize()
    cold = (time.perf_counter_ns()-started)/1e6
    for _ in range(20):
        call()
    synchronize()
    samples = []
    for _ in range(200):
        started = time.perf_counter_ns()
        call()
        synchronize()
        samples.append((time.perf_counter_ns()-started)/1e6)
    return dict(first_call_ms=cold, warmup_calls=20, measured_calls=200,
        p50_ms=float(np.quantile(samples, .5)), p95_ms=float(np.quantile(samples, .95)),
        maximum_ms=max(samples), over_20ms=sum(value > 20. for value in samples),
        software_scope=specification()["timing"], robot_end_to_end_ms=None)


def fixed_sources(plan):
    found = {Path(row["relative"]).stem.lower(): row for row in plan["clips"]
             if Path(row["relative"]).stem.lower() in FIXED}
    if set(found) != set(FIXED) or any(row["split"] != "test" for row in found.values()):
        raise ValueError("Require all three fixed clips in the frozen test split")
    return [found[name] for name in FIXED]


def execute(args):
    if not args.local_mps_study or sys.platform != "darwin":
        raise ValueError("Explicit independent local Apple MPS review only")
    budget = MPSBudget()
    root, output = args.project_root.resolve(), args.output.resolve()
    lock = args.gpu_lock.resolve()
    if output.exists() or not output.is_relative_to(root/"data/supermimic_for_q1"):
        raise ValueError("Use a new independent review root")
    if not lock.is_relative_to(root/"data/locks"):
        raise ValueError("Require the independent local MPS lock")
    inputs = {key: identity(getattr(args, key)) for key in ("preregistration", "checkpoint", "ridge", "plan")}
    if (inputs["checkpoint"]["sha256"] != MODEL_SHA or inputs["ridge"]["sha256"] != RIDGE_SHA
            or json.loads(args.preregistration.read_text()) != specification()):
        raise ValueError("Fixed models or preregistration differ")
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
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        audit_cpu_tensors(torch, saved)
        if (saved["schema"] != "q1_upper_mps_forecast_model_v1"
                or saved["selection"]["selected_update"] != 800
                or saved["plan_identity"] != inputs["plan"]):
            raise ValueError("Require the exact validation-selected fixed predictor")
        normalization = {key: value.numpy() for key, value in saved["normalization"].items()}
        model = make_model(torch)
        model.load_state_dict(saved["model"], strict=True)
        model.to("mps").eval()
        ridge = json.loads(args.ridge.read_text())
        if (ridge["schema"] != "q1_upper_noise_robust_model_v1"
                or ridge["plan_identity"] != inputs["plan"]):
            raise ValueError("Require the exact noise-risk ridge comparison")
        ridge_model = ridge["model"]
        for key, shape in (("mean", (200,)), ("scale", (200,)), ("weights", (201, 80))):
            ridge_model[key] = np.asarray(ridge_model[key], dtype=np.float64)
            if ridge_model[key].shape != shape or not np.isfinite(ridge_model[key]).all():
                raise ValueError("Invalid saved ridge tensor")
        if (ridge_model["scale"] < 1e-6).any():
            raise ValueError("Invalid physical normalization")
        plan = load_plan(args.admission, args.sources, args.plan)
        corpus = SavedUpperCorpus(plan, json.loads(args.admission.read_text()),
                                  args.review_root, args.delivery_manifest)
        rows, timings = [], None
        output.mkdir(parents=True)
        save(output/"status.json", dict(status="RUNNING", started_utc=datetime.now(timezone.utc).isoformat()))
        try:
            for row in fixed_sources(plan):
                budget.check()
                reference = corpus.load(row["relative"], "test")
                for scenario in SCENARIOS:
                    budget.check()
                    observed, support = observations(reference, scenario)
                    cutoffs, history, targets = dense_windows(reference, observed)
                    if timings is None:
                        timings = timing_review(lambda: prediction(torch, model, history[:1], normalization),
                                                torch.mps.synchronize)
                    ages = cutoffs-support[cutoffs]
                    forecasts = {method: predict(history, method, {"noise_ridge_1.0": ridge_model})
                                 for method in METHODS if method != "mps_mlp"}
                    forecasts["mps_mlp"] = prediction(torch, model, history, normalization)
                    for method, forecast in forecasts.items():
                        gated, replaced = stale_gate(forecast, history, ages)
                        for variant, values in (("ungated", forecast), ("hold_after_40ms", gated)):
                            rows.append(dict(relative=row["relative"], scenario=scenario, method=method,
                                variant=variant, windows=len(cutoffs), max_source_age_ms=int(ages.max()*20),
                                held_source_frames=int(np.count_nonzero(support != np.arange(len(support)))),
                                stale_gate_windows=int(replaced.sum()) if variant != "ungated" else 0,
                                **metrics(values, targets)))
            budget.check()
            for key, expected in inputs.items():
                if identity(getattr(args, key)) != expected:
                    raise ValueError("Fixed review input changed")
            result = dict(schema="q1_upper_rolling_review_v1", status="COMPLETED",
                code_identity=identity(__file__), fixed_inputs=inputs, rows=rows,
                software_timing=timings, resources=budget.check(), resource_observations=budget.observations,
                source_identities=corpus.verify_unchanged(), reused_test_exploratory=True,
                models_refitted=False, model_or_threshold_selection=False, actual_mps_optimizer_updates=0,
                robot_policy_inference=0, physics_steps=0, server_access=False,
                measured_robot_latency_ms=None, naturalness_conclusion=None, user_accepted_checkpoint=None)
            save(output/"result.json", result)
            if sum(path.stat().st_size for path in output.iterdir()) > 8*1024**2:
                raise RuntimeError("Independent rolling review output exceeded budget")
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
    for name in ("project-root", "output", "gpu-lock", "checkpoint", "ridge", "plan",
                 "preregistration", "admission", "sources", "review-root", "delivery-manifest"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--local-mps-study", action="store_true")
    result = execute(parser.parse_args())
    print(json.dumps(dict(status=result["status"], rows=len(result["rows"]),
        timing=result["software_timing"], resources=result["resources"]), allow_nan=False))
