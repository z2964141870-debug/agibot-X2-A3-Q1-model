"""Independent bounded Apple MPS forecast study on saved reference angles.

This is an eight-channel history-only predictor, not a robot controller. The
previous test was inspected already; all reused test comparisons are exploratory.
No server, CUDA, original training input or original GPU lock is accessed.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np

from forecast_data import identity, load_plan
from upper_forecast import (CHANNELS, LocalBudget, SavedUpperCorpus, collect,
                            history_features, save, score)
from upper_noise_robust import STRESS_CASES, observe


PRIOR_RESULT_SHA = "48a728eb84e35d9d3aceeb9c3bf542f02daa0ceed6b76253e9ad442fce9cfd5f"
SEED = 2026110603
STEPS = 1000


def specification():
    return dict(schema="q1_upper_mps_forecast_preregistration_v1",
        channels=list(CHANNELS), history_frames=25, future_frames=10,
        features="24 past poses relative to latest plus latest8,200 scalars",
        model="Linear200-256,SiLU,Linear256-128,SiLU,Linear128-80;zero output head",
        normalized_feature_scale_floor_rad=.01, normalization_fitted_split="train",
        train_windows=8192, evaluation_windows_per_split=2048,
        training_iid_history_noise_std_rad=.01, labels_noise_free=True,
        output_anchor="latest observed pose;training noise includes anchor",
        optimizer="Adam", learning_rate=.001, batch_size=128, maximum_updates=STEPS,
        gradient_norm_limit=1., seed=SEED, validation_every_updates=100,
        validation_iid_noise_std_rad=[0., .01, .02], validation_noise_seed=2026110604,
        checkpoint_selection="minimum equal mean over3 validation noise levels and10 horizon RMSEs;earliest tie",
        selection_persisted_before_test=True, test_cases=list(STRESS_CASES),
        test_noise_seed=2026110602, test_reuse="exploratory;previously inspected128 source clips",
        independent_confirmation=False, prior_result_sha256=PRIOR_RESULT_SHA,
        prior_models_retrained_or_rescored=False, deployment=False,
        backend="Apple MPS only;no CPU or CUDA training fallback",
        torch_version="2.7.0", numpy_version="1.26.4",
        maximum_total_seconds=600, maximum_training_seconds=500,
        maximum_rss_bytes=1024**3, maximum_mps_driver_bytes=1024**3,
        minimum_system_memory_free_percent=25, mps_per_process_memory_fraction=.125,
        independent_local_gpu_lock_required=True, maximum_output_bytes=16*1024**2,
        test_only_recordings_used=False, robot_physics=False,
        original_training_or_server_access=False)


def require_mps(torch):
    if (sys.platform != "darwin" or platform.machine() != "arm64"
            or not torch.backends.mps.is_built() or not torch.backends.mps.is_available()
            or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0"):
        raise ValueError("Require actual local Apple MPS with CPU fallback disabled")
    if torch.__version__.split("+")[0] != "2.7.0" or np.__version__ != "1.26.4":
        raise ValueError("Require the independently pinned local study runtime")


class MPSBudget(LocalBudget):
    def __init__(self):
        super().__init__(seconds=600, rss_bytes=1024**3)
        self.torch = None
        self.observations = []

    def check(self):
        if self.torch is not None:
            self.torch.mps.synchronize()
        values = super().check()
        pressure = subprocess.run(["memory_pressure", "-Q"], check=True,
            capture_output=True, text=True, timeout=3)
        match = re.search(r"^System-wide memory free percentage: (\d+)%$",
                          pressure.stdout, flags=re.MULTILINE)
        if match is None or int(match.group(1)) < 25:
            raise RuntimeError("Missing or insufficient actual Mac system memory headroom")
        values["system_memory_free_percent"] = int(match.group(1))
        if self.torch is not None:
            values["mps_current_allocated_bytes"] = self.torch.mps.current_allocated_memory()
            values["mps_driver_allocated_bytes"] = self.torch.mps.driver_allocated_memory()
            if values["mps_driver_allocated_bytes"] >= 1024**3:
                raise RuntimeError("Independent MPS study driver allocation exceeded1GiB")
        self.observations.append(values)
        return values


def fit_normalization(history, *, split):
    if split != "train":
        raise ValueError("Only train sources may fit the MPS predictor normalization")
    features = history_features(history)
    if not len(features):
        raise ValueError("Require nonempty complete training histories")
    return dict(mean=features.mean(axis=0), scale=np.maximum(features.std(axis=0), .01))


def make_model(torch):
    model = torch.nn.Sequential(torch.nn.Linear(200, 256), torch.nn.SiLU(),
        torch.nn.Linear(256, 128), torch.nn.SiLU(), torch.nn.Linear(128, 80))
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    return model


def feature_tensor(torch, history, normalization, device):
    features = history_features(history)
    return torch.as_tensor((features-normalization["mean"])/normalization["scale"],
                           dtype=torch.float32, device=device)


def prediction(torch, model, history, normalization, *, device="mps"):
    result = []
    with torch.no_grad():
        for begin in range(0, len(history), 256):
            part = history[begin:begin+256]
            delta = model(feature_tensor(torch, part, normalization, device))
            result.append(part[:, -1, None]+delta.detach().cpu().numpy().reshape(-1, 10, 8))
    return np.concatenate(result)


def cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().clone()
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite training checkpoint tensor")
        return value
    if isinstance(value, dict):
        return {key: cpu_tree(torch, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(torch, item) for item in value]
    return value


def audit_cpu_tensors(torch, value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError("Require finite CPU-only saved training tensors")
        return 1
    if isinstance(value, dict):
        return sum(audit_cpu_tensors(torch, item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(audit_cpu_tensors(torch, item) for item in value)
    return 0


def fit(torch, model, train, labels, validation, val_labels, val_items, normalization, budget):
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = np.random.default_rng(SEED)
    validation_inputs = [validation if not std else observe(validation, "iid_"+str(std), seed=2026110604)
                         for std in (0., .01, .02)]
    trace, best = [], None

    def validate(step, loss):
        nonlocal best
        model.eval()
        profiles = []
        for std, observed in zip((0., .01, .02), validation_inputs):
            report = score(prediction(torch, model, observed, normalization), val_labels,
                           val_items, "mps_mlp", "validation", std)
            profiles.append(report["per_horizon_rmse_rad"])
        metric = float(np.mean(profiles))
        if not np.isfinite(metric):
            raise ValueError("Nonfinite validation objective")
        row = dict(completed_updates=step, last_training_loss=loss,
            validation_risk_rad=metric, noise_horizon_rmse_rad=profiles, resources=budget.check())
        trace.append(row)
        if best is None or metric < best["metric"]:
            best = dict(metric=metric, step=step, state=cpu_tree(torch, model.state_dict()))
        model.train()

    validate(0, None)
    steps, last_loss, stop_reason = 0, None, "UPDATE_BOUND"
    for step in range(1, STEPS+1):
        if time.monotonic()-budget.started >= 500:
            stop_reason = "TRAINING_WALL_TIME_BOUND"
            break
        indices = rng.integers(0, len(train), 128)
        observed = train[indices]+rng.normal(0., .01, (128, 25, 8))
        inputs = feature_tensor(torch, observed, normalization, "mps")
        anchor = torch.as_tensor(observed[:, -1], dtype=torch.float32, device="mps")
        target = torch.as_tensor(labels[indices].reshape(128, 80), dtype=torch.float32, device="mps")
        forecast = anchor.repeat(1, 10)+model(inputs)
        loss = torch.nn.functional.mse_loss(forecast, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        steps, last_loss = step, float(loss.detach().cpu())
        if not np.isfinite(last_loss):
            raise ValueError("Nonfinite actual MPS training loss")
        if step % 100 == 0:
            validate(step, last_loss)
        elif step % 50 == 0:
            budget.check()
    if steps != trace[-1]["completed_updates"]:
        validate(steps, last_loss)
    final_state = cpu_tree(torch, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        completed_updates=steps, stop_reason=stop_reason))
    optimizer_steps = [int(state["step"].item()) for state in optimizer.state.values()]
    if optimizer_steps != [steps]*len(list(model.parameters())):
        raise ValueError("Actual Adam parameter counters differ from completed updates")
    return best, final_state, trace, optimizer_steps


def execute(args):
    if not args.local_mps_study or sys.platform != "darwin":
        raise ValueError("Explicit independent local Apple MPS study only")
    budget = MPSBudget()
    project = args.project_root.resolve()
    output, lock = args.output.resolve(), args.gpu_lock.resolve()
    if output.exists() or not output.is_relative_to(project/"data/supermimic_for_q1"):
        raise ValueError("Use a new independent output root")
    if not lock.is_relative_to(project/"data/locks"):
        raise ValueError("Use only the independent local MPS lock directory")
    config_id, plan_id = identity(args.preregistration), identity(args.plan)
    if json.loads(args.preregistration.read_text()) != specification():
        raise ValueError("Preregistered model or execution bounds differ")
    prior_id = identity(args.prior_result)
    if prior_id["sha256"] != PRIOR_RESULT_SHA or prior_id["bytes"] > 4*1024**2:
        raise ValueError("Require the pinned prior exploratory result for paired comparisons")
    budget.check()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch

        require_mps(torch)
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        torch.mps.set_per_process_memory_fraction(.125)
        budget.torch = torch
        budget.check()
        plan = load_plan(args.admission, args.sources, args.plan)
        corpus = SavedUpperCorpus(plan, json.loads(args.admission.read_text()),
                                   args.review_root, args.delivery_manifest)
        output.mkdir(parents=True)
        status = dict(status="RUNNING", started_utc=datetime.now(timezone.utc).isoformat(),
            code_identity=identity(__file__), preregistration_identity=config_id,
            plan_identity=plan_id, prior_result_identity=prior_id,
            actual_backend="mps", torch_version=str(torch.__version__), numpy_version=np.__version__)
        save(output/"status.json", status)
        try:
            train, labels, train_items = collect(plan, corpus, "train", 8192, budget.check)
            val, val_labels, val_items = collect(plan, corpus, "validation", 2048, budget.check)
            normalization = fit_normalization(train, split="train")
            model = make_model(torch).to("mps")
            best, last, trace, optimizer_steps = fit(torch, model, train, labels, val,
                                                    val_labels, val_items, normalization, budget)
            selected = dict(schema="q1_upper_mps_selection_v1", method="mps_mlp",
                selected_update=best["step"], completed_updates=last["completed_updates"],
                validation_risk_rad=best["metric"], selection_split="validation",
                test_labels_read=False, test_used_for_selection=False, reused_test_exploratory=True)
            saved_model = dict(schema="q1_upper_mps_forecast_model_v1", model=best["state"],
                normalization={key: torch.from_numpy(value.copy()) for key, value in normalization.items()},
                selection=selected, plan_identity=plan_id, preregistration_identity=config_id)
            for name, payload in (("selected_model.pt", saved_model), ("last_training_state.pt", last)):
                audit_cpu_tensors(torch, payload)
                with (output/name).open("xb") as stream:
                    torch.save(payload, stream)
                audit_cpu_tensors(torch, torch.load(output/name, map_location="cpu", weights_only=True))
            save(output/"validation_selection.json", selected)
            save(output/"training_trace.json", trace)
            model.load_state_dict(best["state"])
            model.eval()
            testing, targets, test_items = collect(plan, corpus, "test", 2048, budget.check)
            previous = json.loads(args.prior_result.read_text())
            test_hash = hashlib.sha256(json.dumps(test_items, separators=(",", ":")).encode()).hexdigest()
            if (previous["schema"] != "q1_upper_noise_robust_exploration_v1"
                    or previous["sampling"]["test"]["items_sha256"] != test_hash):
                raise ValueError("Previous baseline used different test windows")
            reports = []
            for case in STRESS_CASES:
                budget.check()
                observed = observe(testing, case)
                value = score(prediction(torch, model, observed, normalization), targets,
                              test_items, "mps_mlp", "test", None)
                value.update(stress_case=case, reused_test_exploratory=True)
                reports.append(value)
            if (identity(args.preregistration) != config_id or identity(args.plan) != plan_id
                    or identity(args.prior_result) != prior_id):
                raise ValueError("Study inputs changed")
            inputs = corpus.verify_unchanged()
            resources = budget.check()
            if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 8*1024**2:
                raise ValueError("Study output reserve exceeded")
            result = dict(status, status="COMPLETED", selection=selected, test=reports,
                prior_baseline_test_reused=previous["test"], prior_models_refit_or_rescored=False,
                training_trace=trace, final_resources=resources, resource_observations=budget.observations,
                actual_adam_steps=optimizer_steps, parameter_count=sum(p.numel() for p in model.parameters()),
                checkpoints={name: identity(output/name) for name in ("selected_model.pt", "last_training_state.pt")},
                input_identities=inputs, cpu_checkpoint_tensors_verified=True,
                original_training_modified=False, server_access=False, robot_policy_inference=0,
                physics_steps=0, cuda_jobs=0, actual_mps_optimizer_updates=last["completed_updates"],
                test_used_for_selection=False, test_only_recordings_used=False,
                independent_confirmation=False, measured_robot_latency_ms=None,
                naturalness_conclusion=None, user_accepted_checkpoint=None)
            save(output/"result.json", result)
            if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 16*1024**2:
                raise ValueError("Independent study output exceeded16MiB")
            save(output/"status.json", dict(status, status="COMPLETED",
                result=identity(output/"result.json")), replace=True)
            return result
        except BaseException as error:
            save(output/"failure.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}",
                                             elapsed_seconds=time.monotonic()-budget.started))
            save(output/"status.json", dict(status, status="FAILED"), replace=True)
            raise
        finally:
            torch.mps.synchronize()
            torch.mps.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "admission", "sources", "review-root", "delivery-manifest",
                 "preregistration", "prior-result", "project-root", "output", "gpu-lock"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--local-mps-study", action="store_true")
    value = execute(parser.parse_args())
    print(json.dumps(dict(status=value["status"], selection=value["selection"],
        actual_mps_optimizer_updates=value["actual_mps_optimizer_updates"],
        resources=value["final_resources"]), allow_nan=False))
