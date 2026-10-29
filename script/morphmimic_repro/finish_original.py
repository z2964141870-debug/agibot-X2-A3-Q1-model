#!/usr/bin/env python3
"""Audit a completed run, then call the unchanged upstream deployment tools."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": value.hexdigest()}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    plan = json.loads(args.plan.read_text())
    progress = json.loads((experiment / "training_observation/progress.json").read_text())
    guard = json.loads((experiment / "training_guard.json").read_text())
    if progress["status"] != "TRAINING_COMPLETED" or guard["status"] != "WORKER_COMPLETED":
        raise RuntimeError("Original training and its guard must have completed")
    for name in ("wm", "policy", "critic"):
        if progress["optimizer_steps"][name] <= 0:
            raise RuntimeError(f"No actual {name} updates")
    output = experiment / "deployment_review"
    output.mkdir(exist_ok=False)
    record = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
              "plan": plan, "commands": [], "checkpoint_audit": {}, "replays": []}
    save(output / "progress.json", record)

    def command(label, argv, limit=300):
        item = {"label": label, "argv": [str(x) for x in argv], "timeout_seconds": limit}
        record["commands"].append(item)
        save(output / "progress.json", record)
        started = time.monotonic()
        with (output / f"{label}.log").open("w") as stream:
            child = subprocess.Popen(item["argv"], cwd=args.source, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                item["returncode"] = child.wait(timeout=limit)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                item["wall_seconds"] = time.monotonic() - started
                save(output / "progress.json", record)
        if item.get("returncode") != 0:
            raise RuntimeError(f"{label} failed; see its retained log")
        print(f"completed: {label}", flush=True)

    try:
        import torch
        run = Path(progress["run_dir"])
        initial = torch.load(run / "ckpt_0.pt", map_location="cpu", weights_only=False)
        final_iteration = plan["checkpoints"][-1]
        final = torch.load(run / f"ckpt_{final_iteration}.pt", map_location="cpu", weights_only=False)
        for iteration, checkpoint in ((0, initial), (final_iteration, final)):
            steps = {}
            for name, optimizer in checkpoint["optim"].items():
                values = sorted({int(v["step"].item()) for v in optimizer["state"].values()})
                steps[name] = values
                expected = 0 if iteration == 0 else progress["optimizer_steps"][name]
                if values != ([] if expected == 0 else [expected]):
                    raise RuntimeError(f"{name} saved optimizer step differs from observer")
            tensors = [v for name in ("wm", "policy", "critic")
                       for v in checkpoint[name].values() if isinstance(v, torch.Tensor)]
            finite = all(torch.isfinite(v).all().item() for v in tensors)
            if not finite or checkpoint["iteration"] != iteration:
                raise RuntimeError("Nonfinite checkpoint or incorrect iteration")
            record["checkpoint_audit"][str(iteration)] = {
                **digest(run / f"ckpt_{iteration}.pt"), "optimizer_step_values": steps,
                "all_network_tensors_finite": finite}
        changes = {}
        for name in ("wm", "policy", "critic"):
            changes[name] = max(float((value.float() - initial[name][key].float()).abs().max())
                                for key, value in final[name].items() if value.numel())
        if not all(value > 0 for value in changes.values()):
            raise RuntimeError("A network has no saved tensor change")
        record["network_max_abs_changes"] = changes
        del initial, final
        save(output / "progress.json", record)
        command("assets", [sys.executable, "-m", "deploy.sim2sim.export_assets",
                "--robot", "g1", "--subset-yaml", progress["config"]["subset_yaml"],
                "--clips", *[clip["name"] for clip in plan["clips"]],
                "--out", experiment / "assets"])
        for iteration in plan["checkpoints"]:
            exported = experiment / f"onnx_{iteration}"
            command(f"export_{iteration}", [sys.executable, "-m", "deploy.export_onnx",
                    run / f"ckpt_{iteration}.pt", exported])
            for clip in plan["clips"]:
                label = f"it{iteration}_{clip['name']}"
                video = output / f"{label}.mp4"
                summary = output / f"{label}.json"
                command(label, [sys.executable, "-m", "deploy.run", "--deploy", exported,
                        "--assets", experiment / "assets", "--robot", "g1", "--clip",
                        clip["name"], "--interface", "motion", "--video", video,
                        "--summary", summary], limit=180)
                result = json.loads(summary.read_text())
                record["replays"].append({"iteration": iteration, "upstream_summary": result,
                                          "video": digest(video), "summary": digest(summary)})
                save(output / "progress.json", record)
        command("curves", [sys.executable, Path(__file__).with_name("summarize_original.py"),
                           "--run", run, "--output", output / "curves"])
        record["status"] = "DEPLOYMENT_REVIEW_COMPLETED"
    except BaseException as exc:
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save(output / "progress.json", record)
        print(json.dumps(record, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
