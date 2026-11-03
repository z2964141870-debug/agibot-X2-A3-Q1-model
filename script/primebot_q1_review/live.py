"""Blocking checkpoint-only CPU observation at complete optimizer boundaries."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "primebot_q1"))
from common import identity, now, require, save
from metrics import scheduled, sustained_stop


def observe(checkpoint, cfg, iteration, modules, capture_rng):
    if not os.environ.get("Q1_LIVE_MONITOR") or not scheduled(iteration, cfg.iterations):
        return False
    from primebot_hooks import signature
    root = cfg.output_dir / "live_review"
    directory = root / f"iteration_{iteration:06d}"
    history_path = root / "history.json"
    root.mkdir(parents=True, exist_ok=True)
    history = json.loads(history_path.read_text())["observations"] if history_path.exists() else []
    require(not directory.exists(), "Do not silently repeat a prior physical observation")
    before = signature(capture_rng(cfg.device))
    parameters = signature([module.state_dict() for module in modules])
    path = cfg.output_dir / "checkpoint_0.pt" if iteration == 0 else checkpoint()
    command = [sys.executable, "-u", str(Path(__file__).with_name("evaluate.py")),
        "--checkpoint", str(path), "--output", str(directory), "--render"]
    baseline = root / "iteration_000000/result.json"
    if iteration:
        command += ["--baseline", str(baseline)]
    started = time.monotonic()
    row = dict(iteration=iteration, started_utc=now(), checkpoint=identity(path), status="FAILED")
    try:
        with (root / f"iteration_{iteration:06d}.log").open("x") as log:
            result = subprocess.run(command, cwd=cfg.output_dir, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        if result.returncode == 0:
            evaluated = json.loads((directory / "result.json").read_text())
            row.update(status=evaluated["status"], result=identity(directory / "result.json"), gate=evaluated.get("gate"))
        else:
            row["error"] = "CPU observer exit " + str(result.returncode)
    except (subprocess.TimeoutExpired, OSError, ValueError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    require(before == signature(capture_rng(cfg.device)), "Observer changed training RNG")
    require(parameters == signature([module.state_dict() for module in modules]), "Observer changed training parameters")
    row.update(seconds=time.monotonic() - started, rng_unchanged=True, parameters_unchanged=True,
               new_training_transitions=0, new_optimizer_updates=0)
    history.append(row)
    early = sustained_stop(history)
    save(history_path, dict(observations=history, early_stop=early,
        rule="three consecutive fixed9/9 plus paired-prefix tracking improvements; provisional only"))
    if early:
        save(cfg.output_dir / "early_stop.json", dict(iteration=iteration, reason="sustained_fixed_tracking_gate",
             observations=history[-3:], truncated_recipe=iteration < cfg.iterations,
             authorizes_morph_training=False, user_accepted_checkpoint=None))
    print(json.dumps(dict(live_review_iteration=iteration, status=row["status"],
        gate=(row.get("gate") or {}).get("passed"), early_stop=early, seconds=row["seconds"])), flush=True)
    return early
