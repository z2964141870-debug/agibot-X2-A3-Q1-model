"""Small, explicit CPU forecast study on already delivered reference angles.

These eight retargeted shoulder/elbow channels are not a live SMPL stream.
Actual robot angles never enter fitting or evaluation. Test labels are loaded
only after the validation choice has been persisted.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import resource
import sys
import time

import numpy as np

from forecast_data import ADMISSION_SHA, identity, load_plan, window_items


DELIVERY_SHA = "a1821f1f792046ab535047fa11fd963388441187ec5bdeded9365a8f14a35d16"
CHANNELS = tuple(side + "_" + joint + "_joint" for side in ("left", "right")
                 for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"))
BASELINES = ("hold", "cv", "damped_cv")
LAMBDAS = (0.0001, 0.01, 1.0)
DT = .02


def specification():
    return dict(schema="q1_saved_upper_forecast_preregistration_v1",
        history_frames=25, future_frames=10, channels=list(CHANNELS), fps=50,
        max_train_windows=8192, max_evaluation_windows_per_split=2048,
        sampling="source_balanced_uniform_cutoffs_v1",
        features="24 past poses relative to latest, followed by latest pose; 200 scalars",
        targets="next ten poses minus latest visible pose; 80 scalars",
        normalization="training inputs only, feature std floor 1e-6 rad",
        ridge_mean_loss_lambdas=list(LAMBDAS), bias_regularized=False,
        baseline_methods=list(BASELINES), slope_frames=5, damped_tau_s=.12,
        clipping_or_smoothing=False,
        selection="smallest validation mean of ten per-horizon angle RMSE values",
        tie_break="declared baseline order, then ascending ridge lambda",
        test_stress_noise_std_rad=.01, noise_seed=20261106,
        stress_method_selection=False, training_noise=False,
        maximum_seconds=90, maximum_rss_bytes=1024**3,
        local_cpu_only=True, source_labels="saved reference angles only",
        hardware_capture_causal_verified=False, robot_control=False,
        test_only_recordings_used=False, original_training_modified=False)


def save(path, value, *, replace=False):
    with Path(path).open("w" if replace else "x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


class LocalBudget:
    def __init__(self, seconds=90, rss_bytes=1024**3):
        self.started = time.monotonic()
        self.seconds, self.rss_bytes = seconds, rss_bytes

    def check(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = int(rss if sys.platform == "darwin" else rss*1024)
        elapsed = time.monotonic()-self.started
        if elapsed >= self.seconds or rss_bytes >= self.rss_bytes:
            raise RuntimeError("Independent local CPU study exceeded time or RSS budget")
        return dict(elapsed_seconds=elapsed, peak_rss_bytes=rss_bytes)


class SavedUpperCorpus:
    def __init__(self, plan, admission, review_root, delivery_manifest, *,
                 expected_manifest_sha=DELIVERY_SHA):
        self.root = Path(review_root).resolve()
        self.rows = {row["relative"]: row for row in plan["clips"]}
        self.admission = {row["relative"]: row for row in admission["rows"]}
        if (len(self.rows) != len(plan["clips"]) or set(self.rows) != set(self.admission)
                or len({Path(name).stem for name in self.rows}) != len(self.rows)):
            raise ValueError("Require unique admitted sources and unambiguous delivered stems")
        self.manifest_path = Path(delivery_manifest)
        self.manifest_identity = identity(delivery_manifest)
        if self.manifest_identity["sha256"] != expected_manifest_sha:
            raise ValueError("Delivery manifest identity differs from the pinned publication")
        manifest = json.loads(self.manifest_path.read_text())
        if (manifest.get("frozen_data", {}).get("sha256") != ADMISSION_SHA
                or identity(delivery_manifest) != self.manifest_identity):
            raise ValueError("Delivery belongs to a different frozen admission or changed")
        self.artifacts = {}
        for item in manifest["artifacts"]:
            name = item["relative"]
            path = PurePosixPath(name)
            if (path.is_absolute() or ".." in path.parts or str(path) != name
                    or "\\" in name or name in self.artifacts):
                raise ValueError("Unsafe or duplicate delivered artifact path")
            self.artifacts[name] = {key: item[key] for key in ("bytes", "sha256")}
        owners = {}
        for name, row in self.rows.items():
            digest = self.admission[name]["q1_primebot"]["sha256"]
            if owners.setdefault(digest, row["split"]) != row["split"]:
                raise ValueError("Identical complete Q1 references cross the frozen split")
        self.used = {}
        self.cached_name, self.cached_reference = None, None

    def _artifact(self, name):
        expected = self.artifacts.get(name)
        if expected is None or not 0 < expected["bytes"] <= 4*1024**2:
            raise ValueError("Missing or oversized delivered upper-body artifact")
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Delivered path escapes the review root")
        raw = path.read_bytes()
        actual = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if actual != expected or identity(path) != actual:
            raise ValueError("Delivered artifact content changed")
        self.used[name] = actual
        return raw

    def load(self, relative, split):
        row = self.rows.get(relative)
        if row is None or row["split"] != split:
            raise ValueError("Requested source does not belong to this split")
        if self.cached_name == relative:
            return self.cached_reference
        stem = Path(relative).stem
        folder = "final_library/" + stem + "_s0/"
        result = json.loads(self._artifact(folder + "result.json"))
        if (result.get("clip") != stem or result.get("seed") != 0
                or result.get("complete") is not True or result.get("failure_frame") is not None
                or result.get("reference_steps") != row["frames"]-1
                or result.get("valid_steps") != row["frames"]-1
                or result.get("reference_sha256") != self.admission[relative]["q1_primebot"]["sha256"]):
            raise ValueError("Require the complete saved case for the exact admitted reference")
        reader = csv.DictReader(io.StringIO(
            self._artifact(folder + "upper_timing.csv").decode("utf-8-sig")))
        expected = ["frame", "time_s"] + [prefix + name for prefix in ("actual_", "reference_")
                                         for name in CHANNELS]
        if reader.fieldnames != expected:
            raise ValueError("Delivered upper-body channels differ from the frozen eight channels")
        values = []
        for index, item in enumerate(reader):
            if (int(item["frame"]) != index or not math.isclose(float(item["time_s"]),
                    index*DT, abs_tol=1e-7, rel_tol=0.)):
                raise ValueError("Require contiguous unshifted 50 Hz frames")
            values.append([float(item["reference_" + name]) for name in CHANNELS])
        reference = np.asarray(values, dtype=np.float64)
        if reference.shape != (row["frames"], 8) or not np.isfinite(reference).all():
            raise ValueError("Incomplete or nonfinite saved reference angles")
        reference.flags.writeable = False
        self.cached_name, self.cached_reference = relative, reference
        return reference

    def verify_unchanged(self):
        if identity(self.manifest_path) != self.manifest_identity:
            raise ValueError("Delivery manifest changed during this study")
        for name, expected in self.used.items():
            if identity(self.root / name) != expected:
                raise ValueError("Delivered source changed during this study")
        return [dict(relative=name, **value) for name, value in sorted(self.used.items())]


def history_features(history):
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 3 or history.shape[1:] != (25, 8) or not np.isfinite(history).all():
        raise ValueError("Require a batch of exactly 25 visible eight-channel poses")
    latest = history[:, -1]
    return np.concatenate(((history[:, :-1]-latest[:, None]).reshape(len(history), -1),
                           latest), axis=1)


def fit_ridges(history, future, *, split):
    if split != "train":
        raise ValueError("Only training sources may fit a model or normalization")
    features = history_features(history)
    future = np.asarray(future, dtype=np.float64)
    if future.shape != (len(history), 10, 8) or not np.isfinite(future).all() or not len(history):
        raise ValueError("Require ten physical target poses per training history")
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    x = np.column_stack(((features-mean)/scale, np.ones(len(features))))
    y = (future-history[:, -1, None]).reshape(len(history), -1)
    # Explicit contractions avoid spurious Accelerate matmul floating flags on Mac.
    gram = np.einsum("ni,nj->ij", x, x, optimize=False) / len(x)
    rhs = np.einsum("ni,nj->ij", x, y, optimize=False) / len(x)
    penalty = np.eye(x.shape[1])
    penalty[-1, -1] = 0.
    models = {}
    for strength in LAMBDAS:
        weights = np.linalg.solve(gram + strength*penalty, rhs)
        if not np.isfinite(weights).all():
            raise ValueError("Nonfinite fitted linear forecast")
        models["ridge_" + str(strength)] = dict(lambda_mean_loss=strength,
            fitted_split="train", mean=mean, scale=scale, weights=weights,
            training_windows=len(history))
    return models


def predict(history, method, models):
    features = history_features(history)
    latest = history[:, -1, None]
    if method in models:
        model = models[method]
        x = np.column_stack(((features-model["mean"])/model["scale"], np.ones(len(history))))
        delta = np.einsum("ni,ij->nj", x, model["weights"], optimize=False).reshape(len(history), 10, 8)
        return latest + delta
    if method not in BASELINES:
        raise ValueError("Unknown upper-body forecast method")
    offsets = np.arange(1, 11)*DT
    if method == "hold":
        return np.repeat(latest, 10, axis=1)
    times = np.arange(-4, 1)*DT
    slope = np.sum(times[None, :, None]*(history[:, -5:]-latest), axis=1) / np.sum(times*times)
    travel = .12 * (-np.expm1(-offsets/.12)) if method == "damped_cv" else offsets
    return latest + travel[None, :, None]*slope[:, None]


def collect(plan, corpus, split, limit, budget_check):
    items = window_items(plan, split, limit)
    if not items:
        raise ValueError("No complete forecast windows")
    history, future = np.empty((len(items), 25, 8)), np.empty((len(items), 10, 8))
    previous = None
    for index, (name, cutoff) in enumerate(items):
        if name != previous:
            budget_check()
            values = corpus.load(name, split)
            previous = name
        if not 24 <= cutoff < len(values)-10:
            raise ValueError("Forecast window crosses a source or terminal boundary")
        history[index], future[index] = values[cutoff-24:cutoff+1], values[cutoff+1:cutoff+11]
    return history, future, items


def score(prediction, target, items, method, split, noise):
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite forecast or changed target time slots")
    errors = prediction-target
    per_clip = []
    names = np.asarray([item[0] for item in items])
    for name in sorted(set(names)):
        subset = errors[names == name]
        per_clip.append(dict(relative=name, windows=len(subset),
            per_horizon_rmse_rad=np.sqrt(np.mean(subset*subset, axis=(0, 2))).tolist()))
    return dict(method=method, split=split, noise_std_rad=noise, windows=len(items),
        sources=len(per_clip), per_horizon_rmse_rad=np.sqrt(np.mean(errors*errors, axis=(0, 2))).tolist(),
        per_joint_per_horizon_rmse_rad=np.sqrt(np.mean(errors*errors, axis=0)).tolist(),
        absolute_error_p95_rad=float(np.quantile(np.abs(errors), .95)),
        per_clip=per_clip, target_offsets_ms=list(range(20, 201, 20)),
        temporal_realignment=False, measured_robot_latency_ms=None)


def evaluate(plan, corpus, *, train_limit=8192, eval_limit=2048,
             budget_check=lambda: None, on_selection=lambda value, models: None):
    training, labels, train_items = collect(plan, corpus, "train", train_limit, budget_check)
    budget_check()
    models = fit_ridges(training, labels, split="train")
    methods = (*BASELINES, *models)
    validation, targets, val_items = collect(plan, corpus, "validation", eval_limit, budget_check)
    reports = []
    for method in methods:
        budget_check()
        reports.append(score(predict(validation, method, models), targets, val_items,
                             method, "validation", 0.))
    chosen = min(reports, key=lambda row: np.mean(row["per_horizon_rmse_rad"]))
    learned = min((row for row in reports if row["method"] in models),
                  key=lambda row: np.mean(row["per_horizon_rmse_rad"]))
    selection = dict(method=chosen["method"], best_learned_method=learned["method"],
        metric="mean per-horizon angle RMSE", value=float(np.mean(chosen["per_horizon_rmse_rad"])),
        fitted_split="train", selection_split="validation", test_labels_read=False,
        test_used_for_selection=False, methods=list(methods))
    on_selection(selection, models)
    testing, targets, test_items = collect(plan, corpus, "test", eval_limit, budget_check)
    test = []
    for noise in (0., .01):
        observed = testing if not noise else testing + np.random.default_rng(20261106).normal(
            0., noise, testing.shape)
        for method in methods:
            budget_check()
            test.append(score(predict(observed, method, models), targets, test_items,
                              method, "test", noise))
    samples = {}
    for split, items in (("train", train_items), ("validation", val_items), ("test", test_items)):
        samples[split] = dict(windows=len(items), sources=len({item[0] for item in items}),
            items_sha256=hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest())
    return dict(schema="q1_saved_upper_forecast_v1", selection=selection, sampling=samples,
        validation=reports, test=test, trained_model_fits=len(models), gradient_optimizer_updates=0,
        robot_policy_inference=0, physics_steps=0, cuda_jobs=0, actual_robot_angles_used=False,
        measured_robot_latency_ms=None, naturalness_conclusion=None,
        source_kind="offline retargeted eight-joint reference, not live human SMPL")


def execute(args):
    if not args.local_cpu_study or sys.platform != "darwin":
        raise ValueError("This explicit small local Mac CPU study is not a server training fallback")
    budget = LocalBudget()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(args.project_root.resolve() / "data/supermimic_for_q1"):
        raise ValueError("Use a new independent Q1 data output directory")
    config_identity = identity(args.preregistration)
    if json.loads(args.preregistration.read_text()) != specification():
        raise ValueError("Preregistered options differ from the fixed implementation")
    plan_identity = identity(args.plan)
    plan = load_plan(args.admission, args.sources, args.plan)
    admission = json.loads(args.admission.read_text())
    corpus = SavedUpperCorpus(plan, admission, args.review_root, args.delivery_manifest)
    output.mkdir(parents=True)
    status = dict(status="RUNNING", started_utc=datetime.now(timezone.utc).isoformat(),
        code_identity=identity(__file__), preregistration_identity=config_identity,
        plan_identity=plan_identity, resources_at_start=budget.check())
    save(output / "status.json", status)

    def persist_selection(value, models):
        save(output / "validation_selection.json", value)
        model = models[value["best_learned_method"]]
        saved = {key: item.tolist() if isinstance(item, np.ndarray) else item
                 for key, item in model.items()}
        save(output / "selected_ridge_model.json", dict(schema="q1_upper_linear_model_v1",
            channels=CHANNELS, history_frames=25, future_frames=10,
            plan_identity=plan_identity, model=saved, selection=value))

    try:
        result = evaluate(plan, corpus, budget_check=budget.check, on_selection=persist_selection)
        if identity(args.plan) != plan_identity or identity(args.preregistration) != config_identity:
            raise ValueError("Preregistration or frozen split changed during the study")
        result.update(status)
        result.update(status="COMPLETED", final_resources=budget.check(),
            input_identities=corpus.verify_unchanged(), delivery_manifest=corpus.manifest_identity,
            test_only_recordings_used=False, test_used_for_model_selection=False,
            original_policy_trained_on_all_393=plan["original_policy_trained_on_all_393"],
            original_training_modified=False, user_accepted_checkpoint=None)
        result["status"] = "COMPLETED"
        save(output / "result.json", result)
        save(output / "completion.json", dict(status="COMPLETED", resources=budget.check(),
            result=identity(output / "result.json")))
        save(output / "status.json", dict(status, status="COMPLETED",
            result=identity(output / "result.json")), replace=True)
        return result
    except BaseException as error:
        save(output / "failure.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.monotonic()-budget.started))
        save(output / "status.json", dict(status, status="FAILED",
            error=f"{type(error).__name__}: {error}"), replace=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "admission", "sources", "review-root", "delivery-manifest",
                "preregistration", "project-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--local-cpu-study", action="store_true")
    value = execute(parser.parse_args())
    print(json.dumps(dict(status=value["status"], selection=value["selection"],
                          resources=value["final_resources"]), allow_nan=False))
