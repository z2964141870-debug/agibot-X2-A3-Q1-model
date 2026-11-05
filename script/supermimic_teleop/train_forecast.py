"""Bounded causal forecaster training on the frozen Q1 source split.

This is an independent experiment. It never touches the V04 training output,
its checkpoints, input manifest or save cadence. The default ``auto`` device
requires a safe CUDA device; CPU training must be explicitly enabled.
"""

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np

from forecast_data import ForecastCorpus, identity, load_plan, window_items
from forecast_geometry import encode_history, encode_targets
from forecast_model import ForecastNormalization, ForecastSpec, NormalizationFit, make_model


GIB = 1024**3
FORMAL_TRAINING = Path("/opt/data/private/ZHY/data/supermimic_for_q1/MORPHMIMIC_Q1_CONDITIONAL_SHARED01_V04_20261104")
DEFAULT_GPU = "GPU-c602a45a-0dd8-9ad2-9adc-4f863facd1d4"
DEFAULT_BOOT = "90395f47-4035-4785-858e-b3d05c44a738"


def cgroup_snapshot():
    v1 = Path("/sys/fs/cgroup/memory")
    if (v1 / "memory.usage_in_bytes").is_file():
        total = int((v1 / "memory.usage_in_bytes").read_text())
        limit = int((v1 / "memory.limit_in_bytes").read_text())
        control = dict(line.split() for line in (v1 / "memory.oom_control").read_text().splitlines())
        return dict(total_bytes=total, limit_bytes=limit, available_bytes=limit-total,
                    oom_kill=int(control["oom_kill"]), under_oom=int(control["under_oom"]), path=str(v1))
    current = Path("/sys/fs/cgroup/memory.current")
    maximum = Path("/sys/fs/cgroup/memory.max")
    if current.is_file() and maximum.is_file():
        raw = maximum.read_text().strip()
        events = dict(line.split() for line in (current.parent / "memory.events").read_text().splitlines())
        total, limit = int(current.read_text()), None if raw == "max" else int(raw)
        return dict(total_bytes=total, limit_bytes=limit,
                    available_bytes=None if limit is None else limit-total,
                    oom_kill=int(events["oom_kill"]), under_oom=int(events["oom"]), path=str(current.parent))
    raise RuntimeError("No supported cgroup memory accounting files")


def gpu_snapshot(expected=None, allowed_pids=None, require_idle=True):
    import subprocess
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,temperature.gpu,gpu_recovery_action",
        "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    rows = [row.split(",") for row in raw.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("Require exactly one visible GPU for the bounded experiment")
    uuid, used, temperature, recovery = (value.strip() for value in rows[0])
    if expected and uuid != expected:
        raise RuntimeError("GPU identity differs from the admitted device")
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    allowed_pids = {os.getpid()} if allowed_pids is None else set(allowed_pids)
    pids = [int(value.strip()) for value in compute.splitlines()
            if value.strip() and int(value.strip()) not in allowed_pids]
    if pids and require_idle:
        raise RuntimeError("A compute process already uses the admitted GPU: " + ",".join(map(str, pids)))
    return dict(uuid=uuid, used_mib=int(used), temperature_c=int(temperature),
                recovery=recovery, other_compute_pids=pids)


def resources(output, *, expected_gpu=None, start=False, require_idle=True):
    memory = cgroup_snapshot()
    if memory["limit_bytes"] is None or memory["limit_bytes"] <= 30*GIB:
        raise RuntimeError("Container memory limit is not above the 30 GiB safety line")
    if memory["oom_kill"] or memory["under_oom"]:
        raise RuntimeError("Cgroup OOM counter is nonzero")
    if memory["total_bytes"] >= (22 if start else 24)*GIB:
        raise RuntimeError("Independent forecast resource reserve reached")
    gpu = gpu_snapshot(expected_gpu, allowed_pids={os.getpid()}, require_idle=require_idle)
    if gpu["used_mib"] >= 18*1024 or gpu["temperature_c"] >= 80 or gpu["recovery"] != "None":
        raise RuntimeError("Independent forecast GPU reserve reached")
    usage = shutil.disk_usage(output)
    if usage.free < 100*GIB:
        raise RuntimeError("Independent forecast disk reserve reached")
    return dict(memory=memory, gpu=gpu, free_bytes=usage.free)


class ForecastBudget:
    """Check the whole experiment, including data preparation and final saving."""

    def __init__(self, output, *, max_seconds, expected_gpu=DEFAULT_GPU,
                 expected_boot=DEFAULT_BOOT, require_idle=True):
        if not math.isfinite(max_seconds) or not 0 < max_seconds <= 1800:
            raise ValueError("Forecast wall bound must be finite and within 1800 seconds")
        self.output, self.max_seconds = Path(output), max_seconds
        self.expected_gpu, self.expected_boot = expected_gpu, expected_boot
        self.require_idle, self.started = require_idle, time.monotonic()
        self.last, self.last_checked = None, float("-inf")

    def remaining(self):
        return self.max_seconds - (time.monotonic()-self.started)

    def check(self, *, start=False, force=False):
        if self.remaining() <= 0:
            raise RuntimeError("Forecast whole-experiment wall bound reached")
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != self.expected_boot:
            raise RuntimeError("Original server boot identity differs")
        if force or start or time.monotonic()-self.last_checked >= 2:
            self.last = resources(self.output, expected_gpu=self.expected_gpu,
                                  start=start, require_idle=self.require_idle)
            self.last_checked = time.monotonic()
            write_json(self.output / "resources.json", self.last)
        return self.last


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def windows(plan, split):
    for row in plan["clips"]:
        if row["split"] != split:
            continue
        for cutoff in range(plan["history_frames"]-1, row["frames"]-plan["future_frames"]):
            yield row["relative"], cutoff


def feature_label(corpus, relative, cutoff, split):
    window = corpus.window(relative, cutoff, split=split)
    encoded = encode_history(window.input_positions, window.input_quaternions)
    labels = encode_targets(window.target_positions, window.target_quaternions, encoded.anchor)
    return encoded.features, labels


def fit_normalization(corpus, plan, output, budget_check):
    fit = NormalizationFit()
    count = 0
    for relative, cutoff in windows(plan, "train"):
        if count % 16 == 0:
            budget_check()
        fit.update(corpus.window(relative, cutoff, split="train"))
        count += 1
    normalization = fit.freeze()
    write_json(output / "normalization.json", normalization.to_dict())
    return normalization, count


def batches(corpus, plan, normalization, items, batch_size, split):
    features, labels = [], []
    for relative, cutoff in items:
        history, target = feature_label(corpus, relative, cutoff, split)
        features.append(normalization.inputs(history))
        labels.append(normalization.targets(target))
        if len(features) == batch_size:
            yield np.stack(features), np.stack(labels)
            features, labels = [], []
    if features:
        yield np.stack(features), np.stack(labels)


def validation_loss(model, corpus, plan, normalization, items, device, torch, budget_check):
    model.eval()
    total, count = 0., 0
    with torch.inference_mode():
        for batch in batches(corpus, plan, normalization, items, 32, "validation"):
            budget_check()
            inputs, labels = (torch.from_numpy(value).to(device) for value in batch)
            prediction = model(inputs)
            squared = (prediction-labels)**2
            total += float(squared.sum().cpu())
            count += squared.numel()
    return total/count if count else None


@contextmanager
def gpu_lock(project_root):
    path = Path(project_root) / "data/supermimic_for_q1/.gait_goal01.gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Original training or another GPU job owns the Q1 lock") from error
        yield path
        fcntl.flock(stream, fcntl.LOCK_UN)


def execute(args):
    args.output = args.output.resolve()
    project_root = args.project_root.resolve()
    if args.output.exists() or not args.output.is_relative_to(project_root / "data/supermimic_for_q1"):
        raise ValueError("Use one new output root under the sole Q1 data directory")
    if args.output.is_relative_to(FORMAL_TRAINING):
        raise ValueError("Independent forecast output cannot be inside formal V04 training")
    if (not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 1800
            or not 1 <= args.max_steps <= 2000 or not 1 <= args.batch_size <= 256):
        raise ValueError("Invalid bounded training budget")
    spec = ForecastSpec(hidden_size=args.hidden_size)
    plan = load_plan(args.admission, args.sources, args.plan)
    args.output.mkdir(parents=True)
    args._owns_output = True
    budget = ForecastBudget(args.output, max_seconds=args.max_seconds,
                            expected_gpu=args.expected_gpu, expected_boot=args.expected_boot)
    record = dict(status="PREFLIGHT", experiment="Q1_CAUSAL_FORECAST_GRU_20261106",
        plan_identity=identity(args.plan), manifest_identities=plan["manifest_identities"], spec=spec.to_dict(),
        max_seconds=args.max_seconds, max_steps=args.max_steps, batch_size=args.batch_size,
        device_requested=args.device, allow_cpu=args.allow_cpu, training_signals_sent=0,
        optimizer_updates=0, test_used_for_model_selection=False, user_accepted_checkpoint=None)
    write_json(args.output / "status.json", record)
    with gpu_lock(project_root):
        initial = budget.check(start=True)
        record["initial_resources"] = initial
        import torch
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            device = torch.device("cuda")
        elif args.allow_cpu:
            device = torch.device("cpu")
        else:
            raise RuntimeError("No safe CUDA device; pass --allow-cpu to override the GPU preference")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.set_float32_matmul_precision("high")
        corpus = ForecastCorpus(plan, args.human_root)
        budget_check = budget.check
        normalization, normalization_windows = fit_normalization(corpus, plan, args.output, budget_check)
        all_train = list(windows(plan, "train"))
        validation = window_items(plan, "validation", args.validation_windows)
        rng = random.Random(args.seed)
        model = make_model(spec).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        model.train()
        reserve_seconds = min(60., args.max_seconds*.2)
        history = []
        for step in range(1, args.max_steps+1):
            if budget.remaining() <= reserve_seconds:
                break
            budget_check()
            rng.shuffle(all_train)
            selected = all_train[:args.batch_size]
            for batch in batches(corpus, plan, normalization, selected, args.batch_size, "train"):
                inputs, labels = (torch.from_numpy(value).to(device) for value in batch)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(inputs)
                loss = torch.mean((prediction-labels)**2)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite forecast loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                record["optimizer_updates"] += 1
                record["training_signals_sent"] += 1
                history.append(dict(step=step, loss=float(loss.detach().cpu())))
            if step == 1 or step % args.check_every == 0:
                current = budget_check()
                record.update(status="RUNNING", last_resources=current,
                    last_loss=history[-1]["loss"], elapsed_seconds=args.max_seconds-budget.remaining())
                write_json(args.output / "status.json", record)
        if not history:
            raise RuntimeError("No optimizer update completed within the whole-experiment budget")
        validation_mse = validation_loss(model, corpus, plan, normalization, validation, device, torch,
                                         budget_check)
        budget_check()
        if identity(args.plan) != record["plan_identity"]:
            raise RuntimeError("Frozen forecast split changed during training")
        cpu_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        if any(not torch.isfinite(tensor).all().item() for tensor in cpu_state.values()):
            raise RuntimeError("Nonfinite forecast checkpoint tensor")
        torch.save(dict(model=cpu_state, spec=spec.to_dict(), plan_identity=record["plan_identity"],
                        status="COMPLETED", optimizer_updates=record["optimizer_updates"],
                        normalization=normalization.to_dict(), seed=args.seed,
                        train_windows=normalization_windows, validation_windows=len(validation)),
                   args.output / "forecast_gru.pt")
        final_resources = budget.check(force=True)
        result = dict(record, status="COMPLETED", device=str(device), normalization_windows=normalization_windows,
            training_windows=len(all_train), validation_windows=len(validation), validation_mse=validation_mse,
            validation_sampling="source_balanced_uniform_cutoffs_v1",
            checkpoint_identity=identity(args.output / "forecast_gru.pt"),
            history=history, final_resources=final_resources, ended_utc=time.time(),
            test_evaluated=False, test_used_for_model_selection=False)
        write_json(args.output / "result.json", result)
        write_json(args.output / "status.json", {key: value for key, value in result.items() if key != "history"})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("plan", "admission", "sources", "human-root", "output", "project-root"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--expected-gpu", default=DEFAULT_GPU)
    parser.add_argument("--expected-boot", default=DEFAULT_BOOT)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--validation-windows", type=int, default=256)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    if (not 0 < args.validation_windows <= 8192 or not 0 < args.check_every <= 100
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("Validation, check and learning-rate values must be positive")
    try:
        result = execute(args)
    except BaseException as error:
        status_path = args.output / "status.json"
        if getattr(args, "_owns_output", False) and status_path.is_file():
            record = json.loads(status_path.read_text())
            record.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=time.time())
            write_json(status_path, record)
        raise
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
