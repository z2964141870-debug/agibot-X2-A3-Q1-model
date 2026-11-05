"""Bounded causal forecaster training on the frozen Q1 source split.

This is an independent experiment. It never touches the V04 training output,
its checkpoints, input manifest or save cadence. The default ``auto`` device
requires a safe CUDA device; CPU training must be explicitly enabled.
"""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np

from forecast_data import ForecastCorpus, load_plan
from forecast_geometry import encode_history, encode_targets
from forecast_model import ForecastNormalization, ForecastSpec, NormalizationFit, make_model


GIB = 1024**3
FORMAL_TRAINING = Path("/opt/data/private/ZHY/data/supermimic_for_q1/MORPHMIMIC_Q1_CONDITIONAL_SHARED01_V04_20261104")
DEFAULT_GPU = "GPU-c602a45a-0dd8-9ad2-9adc-4f863facd1d4"


def cgroup_snapshot():
    v1 = Path("/sys/fs/cgroup/memory")
    if (v1 / "memory.usage_in_bytes").is_file():
        total = int((v1 / "memory.usage_in_bytes").read_text())
        limit = int((v1 / "memory.limit_in_bytes").read_text())
        oom = int((v1 / "memory.oom_control").read_text().split("oom_kill ")[1].split()[0]) \
            if "oom_kill " in (v1 / "memory.oom_control").read_text() else 0
        return dict(total_bytes=total, limit_bytes=limit, oom_kill=oom, path=str(v1))
    current = Path("/sys/fs/cgroup/memory.current")
    maximum = Path("/sys/fs/cgroup/memory.max")
    if current.is_file() and maximum.is_file():
        raw = maximum.read_text().strip()
        return dict(total_bytes=int(current.read_text()),
                    limit_bytes=None if raw == "max" else int(raw), oom_kill=0,
                    path=str(current.parent))
    raise RuntimeError("No supported cgroup memory accounting files")


def gpu_snapshot(expected=None):
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
    pids = [int(value.strip()) for value in compute.splitlines() if value.strip()]
    if pids:
        raise RuntimeError("A compute process already uses the admitted GPU: " + ",".join(map(str, pids)))
    return dict(uuid=uuid, used_mib=int(used), temperature_c=int(temperature),
                recovery=recovery, compute_pids=[])


def resources(output, *, expected_gpu=None, start=False):
    memory = cgroup_snapshot()
    if memory["limit_bytes"] is None or memory["limit_bytes"] <= 30*GIB:
        raise RuntimeError("Container memory limit is not above the 30 GiB safety line")
    if memory["oom_kill"]:
        raise RuntimeError("Cgroup OOM counter is nonzero")
    if memory["total_bytes"] >= (22 if start else 24)*GIB:
        raise RuntimeError("Independent forecast resource reserve reached")
    gpu = gpu_snapshot(expected_gpu)
    if gpu["used_mib"] >= 18*1024 or gpu["temperature_c"] >= 80 or gpu["recovery"] != "None":
        raise RuntimeError("Independent forecast GPU reserve reached")
    usage = shutil.disk_usage(output)
    if usage.free < 100*GIB:
        raise RuntimeError("Independent forecast disk reserve reached")
    return dict(memory=memory, gpu=gpu, free_bytes=usage.free)


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
        fit.update(corpus.window(relative, cutoff, split="train"))
        count += 1
        if count % 256 == 0:
            budget_check()
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


def validation_loss(model, corpus, plan, normalization, items, device, torch):
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in batches(corpus, plan, normalization, items, 32, "validation"):
            inputs, labels = (torch.from_numpy(value).to(device) for value in batch)
            prediction = model(inputs)
            losses.append(float(torch.mean((prediction-labels)**2).cpu()))
    return float(np.mean(losses)) if losses else None


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
    if args.max_seconds <= 0 or args.max_steps <= 0 or not 1 <= args.batch_size <= 256:
        raise ValueError("Invalid bounded training budget")
    plan = load_plan(args.admission, args.sources, args.plan)
    args.output.mkdir(parents=True)
    record = dict(status="PREFLIGHT", experiment="Q1_CAUSAL_FORECAST_GRU_20261106",
        plan_identity=plan["manifest_identities"], spec=ForecastSpec(hidden_size=args.hidden_size).to_dict(),
        max_seconds=args.max_seconds, max_steps=args.max_steps, batch_size=args.batch_size,
        device_requested=args.device, allow_cpu=args.allow_cpu, training_signals_sent=0,
        optimizer_updates=0, test_used_for_model_selection=False, user_accepted_checkpoint=None)
    write_json(args.output / "status.json", record)
    with gpu_lock(project_root):
        initial = resources(args.output, expected_gpu=args.expected_gpu, start=True)
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
        budget_check = lambda: resources(args.output, expected_gpu=args.expected_gpu)
        normalization, normalization_windows = fit_normalization(corpus, plan, args.output, budget_check)
        all_train = list(windows(plan, "train"))
        validation = list(windows(plan, "validation"))[:args.validation_windows]
        rng = random.Random(args.seed)
        model = make_model(ForecastSpec(hidden_size=args.hidden_size)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        model.train()
        started = time.monotonic()
        history = []
        for step in range(1, args.max_steps+1):
            if time.monotonic()-started >= args.max_seconds:
                break
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                record["optimizer_updates"] += 1
                record["training_signals_sent"] += 1
                history.append(dict(step=step, loss=float(loss.detach().cpu())))
            if step == 1 or step % args.check_every == 0:
                current = budget_check()
                record.update(status="RUNNING", last_resources=current,
                    last_loss=history[-1]["loss"], elapsed_seconds=time.monotonic()-started)
                write_json(args.output / "status.json", record)
        validation_mse = validation_loss(model, corpus, plan, normalization, validation, device, torch)
        torch.save(dict(model=model.state_dict(), spec=ForecastSpec(hidden_size=args.hidden_size).to_dict(),
                        normalization=normalization.to_dict(), seed=args.seed,
                        train_windows=normalization_windows, validation_windows=len(validation)),
                   args.output / "forecast_gru.pt")
        final_resources = budget_check()
        result = dict(record, status="COMPLETED", device=str(device), normalization_windows=normalization_windows,
            training_windows=len(all_train), validation_windows=len(validation), validation_mse=validation_mse,
            history=history, final_resources=final_resources, ended_utc=time.time(),
            test_evaluated=False, test_used_for_model_selection=False)
        write_json(args.output / "result.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("plan", "admission", "sources", "human-root", "output", "project-root"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--expected-gpu", default=DEFAULT_GPU)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--validation-windows", type=int, default=256)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    if args.validation_windows <= 0 or args.check_every <= 0 or args.learning_rate <= 0:
        parser.error("Validation, check and learning-rate values must be positive")
    print(json.dumps(execute(args), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
