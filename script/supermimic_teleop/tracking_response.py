"""Descriptive phase and amplitude analysis of already saved robot trajectories.

Every lag uses identical reference support. Positive lag compares a later actual
sample to an earlier reference. Alignment never replaces original tracking
metrics and cannot measure capture-to-robot latency or identify a PD-only delay.
"""

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys

import numpy as np

from forecast_data import identity, load_plan
from upper_forecast import CHANNELS, LocalBudget, SavedUpperCorpus, save


FIXED = ("walk_forward_loop_001__A022", "walk_forward_start_002__A023",
         "walk_forward_stop_002__A024")


def configuration():
    return dict(schema="q1_saved_tracking_response_config_v1", fps=50,
        max_lag_frames=15, minimum_common_samples=10, minimum_std_rad=.01,
        near_peak_correlation_difference=.01, narrow_peak_max_span_ms=60,
        high_correlation_threshold=.7, approximate_lag_agreement_ms=40,
        lag_sign="positive means actual trajectory later than reference",
        support="reference indices max_lag .. frames-max_lag-1 for every lag",
        analysis_channels=list(CHANNELS), original_tracking_metrics_replaced=False,
        affine_fit_is_diagnostic_only=True, robot_end_to_end_latency_measured=False,
        physical_parameter_identification=False, maximum_seconds=90,
        maximum_rss_bytes=1024**3, local_cpu_only=True,
        full_search_curve_storage="three fixed clips only; summaries cover all clips")


def _closest_peak(lags, values, *, maximum):
    optimum = max(values) if maximum else min(values)
    indices = [index for index, value in enumerate(values) if abs(value-optimum) <= 1e-12]
    index = min(indices, key=lambda item: (abs(lags[item]), lags[item]))
    return int(lags[index]), index


def analyze_signal(reference, actual, *, max_lag=15):
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if (reference.ndim != 1 or actual.shape != reference.shape or len(reference) < 2
            or not np.isfinite(reference).all() or not np.isfinite(actual).all()
            or type(max_lag) is not int or not 1 <= max_lag <= 15):
        raise ValueError("Require finite paired trajectories and a bounded integer lag")
    config = configuration()
    ref_std, act_std = float(reference.std()), float(actual.std())
    result = dict(frames=len(reference), unshifted_full_rmse_rad=float(np.sqrt(
        np.mean((actual-reference)**2))), reference_std_rad=ref_std, actual_std_rad=act_std,
        std_amplitude_ratio=None if ref_std < 1e-12 else act_std/ref_std,
        mean_offset_rad=float(np.mean(actual-reference)),
        apparent_correlation_lag_ms=None, minimum_rmse_lag_ms=None,
        candidate_is_narrow_high_correlation=False, flags=[],
        measured_end_to_end_latency_ms=None)
    if len(reference)-2*max_lag < config["minimum_common_samples"]:
        result["flags"].append("INSUFFICIENT_COMMON_SUPPORT")
        return result
    indices = np.arange(max_lag, len(reference)-max_lag)
    r = reference[indices]
    r_centered = r-r.mean()
    ref_energy = float(np.sum(r_centered*r_centered))
    lags = list(range(-max_lag, max_lag+1))
    rmse, correlations = [], []
    for lag in lags:
        a = actual[indices+lag]
        centered = a-a.mean()
        energy = float(np.sum(centered*centered))
        rmse.append(float(np.sqrt(np.mean((a-r)**2))))
        denominator = np.sqrt(ref_energy*energy)
        correlations.append(None if denominator < 1e-12 else float(np.clip(
            np.sum(r_centered*centered)/denominator, -1., 1.)))
    best_error_lag, error_index = _closest_peak(lags, rmse, maximum=False)
    result.update(common_samples=len(indices), reference_support_first_frame=int(indices[0]),
        reference_support_last_frame=int(indices[-1]), lag_grid_ms=[lag*20 for lag in lags],
        common_support_rmse_rad=rmse, common_support_correlations=correlations,
        unshifted_common_rmse_rad=rmse[max_lag], minimum_rmse_lag_ms=best_error_lag*20,
        minimum_shifted_diagnostic_rmse_rad=rmse[error_index])
    if ref_std < config["minimum_std_rad"] or act_std < config["minimum_std_rad"]:
        result["flags"].append("LOW_EXCITATION")
        return result
    if any(value is None for value in correlations):
        result["flags"].append("DEGENERATE_COMMON_SUPPORT")
        return result
    best_lag, correlation_index = _closest_peak(lags, correlations, maximum=True)
    peak = correlations[correlation_index]
    near = [lag*20 for lag, value in zip(lags, correlations)
            if peak-value <= config["near_peak_correlation_difference"]]
    span = max(near)-min(near)
    a = actual[indices+best_lag]
    gain = float(np.sum(r_centered*(a-a.mean()))/ref_energy)
    offset = float(a.mean()-gain*r.mean())
    result.update(apparent_correlation_lag_ms=best_lag*20, peak_correlation=peak,
        near_peak_lags_ms=near, near_peak_span_ms=span, descriptive_affine_gain=gain,
        descriptive_affine_offset_rad=offset,
        affine_residual_rmse_rad=float(np.sqrt(np.mean((a-(gain*r+offset))**2))))
    if abs(best_lag) == max_lag:
        result["flags"].append("CORRELATION_PEAK_AT_SEARCH_BOUNDARY")
    if span > config["narrow_peak_max_span_ms"]:
        result["flags"].append("BROAD_OR_MULTIPLE_CORRELATION_PEAKS")
    if peak < config["high_correlation_threshold"]:
        result["flags"].append("WEAK_WAVEFORM_MATCH")
    if abs(best_lag-best_error_lag)*20 > config["approximate_lag_agreement_ms"]:
        result["flags"].append("CORRELATION_AND_RAW_ERROR_DISAGREE")
    result["candidate_is_narrow_high_correlation"] = not result["flags"]
    return result


def _quantiles(values):
    return None if not values else dict(zip(("p05", "p50", "p95"),
        (float(value) for value in np.quantile(values, (.05, .5, .95)))))


def summarize(cases):
    rows = [row for case in cases for row in case["joints"]]
    candidates = [row for row in rows if row["candidate_is_narrow_high_correlation"]]
    return dict(clips=len(cases), joint_traces=len(rows), narrow_candidates=len(candidates),
        heuristic_flags={flag:sum(flag in row["flags"] for row in rows)
                         for flag in sorted({flag for row in rows for flag in row["flags"]})},
        all_observable_correlation_lag_ms=_quantiles([row["apparent_correlation_lag_ms"]
            for row in rows if row["apparent_correlation_lag_ms"] is not None]),
        narrow_candidate_lag_ms=_quantiles([row["apparent_correlation_lag_ms"] for row in candidates]),
        narrow_candidate_std_amplitude_ratio=_quantiles([row["std_amplitude_ratio"] for row in candidates]),
        nonzero_lag_does_not_establish_physical_delay=True,
        heuristic_flags_are_not_statistical_confidence_intervals=True)


def analyze_corpus(plan, corpus, *, budget_check=lambda: None):
    cases = []
    for row in plan["clips"]:
        budget_check()
        reference = corpus.load(row["relative"], row["split"])
        stem = Path(row["relative"]).stem
        raw = corpus._artifact("final_library/" + stem + "_s0/upper_timing.csv")
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        actual = np.asarray([[float(item["actual_" + name]) for name in CHANNELS]
                             for item in reader], dtype=np.float64)
        if actual.shape != reference.shape or not np.isfinite(actual).all():
            raise ValueError("Incomplete or nonfinite actual joint trace")
        joints = [dict(joint=name, **analyze_signal(reference[:, index], actual[:, index]))
                  for index, name in enumerate(CHANNELS)]
        if stem not in FIXED:
            for joint in joints:
                for field in ("lag_grid_ms", "common_support_rmse_rad", "common_support_correlations"):
                    joint.pop(field, None)
        cases.append(dict(relative=row["relative"], frames=len(actual), joints=joints,
            unshifted_all_joint_rmse_rad=float(np.sqrt(np.mean((actual[1:]-reference[1:])**2)))))
    return dict(schema="q1_saved_tracking_response_v1", status="COMPLETED",
        configuration=configuration(), cases=cases, summary=summarize(cases),
        simulation_sample_interval_ms=20, robot_policy_inference=0, physics_steps=0,
        model_fits=0, original_tracking_metrics_replaced=False,
        measured_robot_end_to_end_latency_ms=None, isolated_pd_delay_ms=None,
        naturalness_conclusion=None)


def execute(args):
    if not args.local_cpu_study or sys.platform != "darwin":
        raise ValueError("This entry is an explicit local Mac saved-data CPU analysis")
    budget = LocalBudget()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(args.project_root.resolve() / "data/supermimic_for_q1"):
        raise ValueError("Use a new independent saved-data analysis output root")
    plan_id = identity(args.plan)
    plan = load_plan(args.admission, args.sources, args.plan)
    corpus = SavedUpperCorpus(plan, json.loads(args.admission.read_text()), args.review_root,
                             args.delivery_manifest)
    output.mkdir(parents=True)
    save(output / "configuration.json", configuration())
    status = dict(status="RUNNING", started_utc=datetime.now(timezone.utc).isoformat(),
        code_identity=identity(__file__), corpus_reader_code_identity=identity(
            Path(__file__).with_name("upper_forecast.py")), plan_identity=plan_id,
        start_resources=budget.check())
    save(output / "status.json", status)
    try:
        result = analyze_corpus(plan, corpus, budget_check=budget.check)
        if identity(args.plan) != plan_id:
            raise ValueError("Frozen source plan changed during analysis")
        result.update(run=status, input_identities=corpus.verify_unchanged(),
            final_resources=budget.check(), user_accepted_checkpoint=None)
        save(output / "result.json", result)
        save(output / "status.json", dict(status, status="COMPLETED",
            result=identity(output / "result.json"), final_resources=budget.check()), replace=True)
        return result
    except BaseException as error:
        save(output / "status.json", dict(status, status="FAILED",
            error=f"{type(error).__name__}: {error}"), replace=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "admission", "sources", "review-root", "delivery-manifest",
                "project-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--local-cpu-study", action="store_true")
    result = execute(parser.parse_args())
    print(json.dumps(dict(status=result["status"], summary=result["summary"],
                          resources=result["final_resources"]), allow_nan=False))
