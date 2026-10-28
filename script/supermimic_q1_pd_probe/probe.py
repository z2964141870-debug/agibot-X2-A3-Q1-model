"""Bounded frozen-policy leg PD sensitivity probe, zero optimization."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = Path("/opt/data/private/ZHY")
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_target_probe"),
               str(HERE.parent / "supermimic_q1_gait_comp"), str(HERE.parent / "supermimic_q1_hp"),
               str(HERE.parent / "supermimic_q1_ppo"), str(HERE.parent / "project_tools")]
from worker import process_identity
from resources import identity, save, stamp
from run_trial import Q1Budget
from runtime_guard import guarded_run
from overnight import OPTIMIZATION_DEADLINE
from pd_control import CANDIDATES, scene_variant, verify_native_gains


def prepare_scene(a, library, case, scene, seed, count):
    from omegaconf import OmegaConf
    sys.path[:0] = [str(a.runtime / "src"), str(a.runtime)]
    from unilab.base.config_adapter import create_env
    from unilab.base.process_device import configure_backend_process_device
    from unilab.training.seed import apply_configured_training_seed
    from unisupermimic.tasks.motion_tracking.q1.supermimic import registry
    from unisupermimic.scripts.train_supermimic import build_env_override

    cfg = OmegaConf.load(a.runtime / "q1_config.yaml")
    rows = json.loads(library.read_text())["splits"]["train"]
    for row in rows:
        if identity(row["path"])["sha256"] != row["sha256"]:
            raise ValueError("Frozen Q1 reference changed")
    cfg.env.commands.motion.motion_files = [row["path"] for row in rows]
    cfg.env.scene.model_file = str(scene)
    cfg.env.commands.motion.mjcf_path = str(scene)
    cfg.algo.seed = seed
    cfg.env.seed = seed
    cfg.training.log_dir = str(case)
    apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    configure_backend_process_device("mjwarp", "cuda:0")
    override = build_env_override(cfg, eval_mode=True, learner_device="cuda:0")
    env = create_env(cfg, num_envs=count, env_cfg_override=override)
    save(case / "seed_contract.json", dict(configured_environment_seed=int(cfg.env.seed),
         configured_algorithm_seed=int(cfg.algo.seed), rollout_global_seed=seed, environment_count=count))
    return env


def evaluate(a):
    import numpy as np
    import torch
    import importlib.util
    spec = importlib.util.spec_from_file_location("frozen_target_probe", HERE.parent / "supermimic_q1_target_probe/probe.py")
    target = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target)
    from physical import FootReference, FIXED_DEMOS, summarize, gait_summary, comparison
    from diagnose_q1_upper_reference import policy_digest
    from overnight import continuing_evidence

    torch.set_num_threads(8)
    source = json.loads(a.library.read_text())
    rows = [dict(row, split=split) for split in ("train", "validation") for row in source["splits"][split]
            if a.scope == "full330" or split == "validation" or row["name"] in FIXED_DEMOS]
    if len(rows) != (330 if a.scope == "full330" else 33):
        raise ValueError("Original fixed library required")
    library = a.output / "evaluation_library.json"
    save(library, {"splits": {"train": rows}, "source": identity(a.library)})
    run = a.output / "run"
    run.mkdir(exist_ok=False)
    status = dict(status="INITIALIZING", process=process_identity(), started_utc=stamp(), optimizer_steps=0)
    save(run / "status.json", status)
    baseline, results = [], []
    thresholds = json.loads(a.calibration.read_text())["thresholds"]
    with torch.no_grad():
        for candidate in a.candidates:
            summaries, comparisons = [], []
            contract = json.loads((a.output / "assets" / (candidate+".json")).read_text())
            for repeat in range(3):
                seed = 32000 + repeat
                case = run / f"{candidate}_repeat{repeat}"
                case.mkdir()
                env = None
                try:
                    env = prepare_scene(a, library, case, a.output / "assets" / (candidate+".xml"), seed, len(rows))
                    save(case / "native_pd_contract.json", verify_native_gains(env, contract))
                    from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
                    teacher = load_policy(env, torch.load(a.checkpoint, map_location=env.device, weights_only=True)).requires_grad_(False).eval()
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    digest = policy_digest(teacher)
                    foot = FootReference(env.supermimic_mdp)
                    def progress(frame):
                        status.update(status="RUNNING", candidate=candidate, repeat=repeat, frame=frame, updated_utc=stamp())
                        save(run / "status.json", status)
                    progress(0)
                    trace = target.physical_rollout(env, teacher, foot, thresholds, "baseline", progress)
                    verify_native_gains(env, contract)
                    path = a.output / f"{candidate}_repeat{repeat}.npz"
                    np.savez_compressed(path, **trace)
                    summary = summarize(trace, rows)
                    summary.update(gait=gait_summary(trace, rows), repeat=repeat, seed=seed, candidate=candidate,
                                   upper_reference_alpha=.5, pd_contract=contract, trace=identity(path))
                    if candidate == "baseline":
                        baseline.append(path)
                    else:
                        with np.load(baseline[repeat], allow_pickle=False) as saved:
                            paired = comparison(trace, {key: saved[key] for key in saved.files}, rows)
                        summary["initial_static075"] = paired
                        comparisons.append(paired)
                    if policy_digest(teacher) != digest:
                        raise AssertionError("Frozen base/normalization changed")
                    save(path.with_suffix(".json"), summary)
                    summaries.append(summary)
                    del teacher, foot, trace
                finally:
                    if env is not None:
                        env.close()
                        env.backend.close()
                    del env
                    gc.collect()
                    torch.cuda.empty_cache()
            results.append(dict(candidate=candidate, summaries=summaries,
                physical_continuation_diagnostic=continuing_evidence(comparisons) if comparisons else None))
            save(a.output / "partial_results.json", dict(candidates=results, optimizer_steps=0))
    save(a.output / "results.json", dict(status="COMPLETED", optimizer_steps=0, rows=rows, candidates=results,
        checkpoint=identity(a.checkpoint), upper_reference_alpha=.5, scope=a.scope, user_accepted_checkpoint=None,
        limitations="Search on existing validation, not held-out test. Correctly configured three seeds, fresh environments; no bitwise native solver claim. Gains changed only in independent XML, original action scale unchanged."))
    status.update(status="COMPLETED", ended_utc=stamp(), frozen_teacher_unchanged=True)
    save(run / "status.json", status)


def main(a):
    if a.worker:
        try:
            evaluate(a)
        except BaseException as error:
            path = a.output / "run/status.json"
            status = json.loads(path.read_text()) if path.exists() else {}
            status.update(status="FAILED", ended_utc=stamp(), error=f"{type(error).__name__}: {error}")
            save(path, status)
            raise
        return
    if (a.output / "status.json").exists():
        raise FileExistsError(a.output / "status.json")
    if not a.controller:
        a.output.mkdir(parents=True, exist_ok=False)
    previous = json.loads(a.frozen_plan.read_text())["source"]
    for item in previous:
        if identity(item["path"]) != item:
            raise RuntimeError("Frozen dependency changed: " + item["path"])
    for candidate in a.candidates:
        contract = scene_variant(a.runtime / "q1_assets/scene.xml", a.output / "assets" / (candidate+".xml"), candidate)
        save(a.output / "assets" / (candidate+".json"), contract)
    sources = {item["path"]: item for item in previous}
    sources.update({str(p): identity(p) for p in [*HERE.glob("*.py"), *(a.output / "assets").glob("*"), a.checkpoint, a.library, a.calibration]})
    source = list(sources.values())
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != a.expected_boot:
        raise RuntimeError("Unexpected container boot")
    save(a.output / "plan.json", dict(source=source, started_utc=stamp(), process=process_identity(),
        optimizer_steps=0, maximum_wall_seconds=2400, scope=a.scope, candidates=a.candidates,
        code_commit=a.code_commit, optimization_deadline_unix=OPTIMIZATION_DEADLINE,
        expected_boot=a.expected_boot, expected_gpu=a.expected_gpu))
    class Budget(Q1Budget):
        def check(self):
            value = super().check()
            if value["gpu"]["uuid"] != a.expected_gpu:
                raise RuntimeError("Unexpected GPU")
            if time.monotonic()-self.started >= 2400 or time.time() >= OPTIMIZATION_DEADLINE-55:
                raise RuntimeError("Bounded PD probe deadline reached")
            return value
    budget = Budget(SimpleNamespace(output=a.output, runtime=a.runtime, mode="diagnosis", expected_oom_kill=0))
    command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"),
               "-u", str(Path(__file__)), *sys.argv[1:], "--worker"]
    status = dict(status="RUNNING", process=process_identity(), started_utc=stamp(), optimizer_steps=0, no_automatic_retry=True)
    save(a.output / "status.json", status)
    try:
        guard = guarded_run(command, ROOT, budget, a.output / "guard.json", ROOT / "logs" / (a.output.name+".log"),
            progress_path=a.output / "run/status.json", maximum_idle_s=900, grace_s=45)
        if guard["status"] != "WORKER_COMPLETED" or guard["cleanup"]["worker_return_code"] != 0:
            raise RuntimeError("PD probe failed; no automatic restart")
        for item in source:
            if identity(item["path"]) != item:
                raise RuntimeError("Frozen dependency changed: " + item["path"])
        status.update(status="COMPLETED", ended_utc=stamp(), frozen_source_count=len(source))
    except BaseException as error:
        status.update(status="FAILED", ended_utc=stamp(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(a.output / "status.json", status)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "calibration", "frozen-plan", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--scope", choices=("quick33", "full330"), default="quick33")
    p.add_argument("--candidates", choices=tuple(CANDIDATES), nargs="+", default=list(CANDIDATES))
    p.add_argument("--expected-boot", required=True)
    p.add_argument("--expected-gpu", required=True)
    p.add_argument("--code-commit", required=True)
    p.add_argument("--detach", action="store_true")
    p.add_argument("--controller", action="store_true")
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT / "data/supermimic_for_q1"):
        p.error("Use sole ZHY experiment data root")
    if len(a.code_commit) != 40 or len(set(a.candidates)) != len(a.candidates) or a.candidates[0] != "baseline":
        p.error("Require published40-digit source and unique candidates with baseline first")
    if time.time() >= OPTIMIZATION_DEADLINE-2400:
        p.error("Insufficient window for bounded probe")
    if a.detach:
        a.output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", str(Path(__file__)), *[v for v in sys.argv[1:] if v != "--detach"], "--controller"]
        with (ROOT / "logs" / (a.output.name+"_controller.log")).open("x") as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(a.output / "launch.json", dict(controller_pid=process.pid, started_utc=stamp(), optimizer_steps=0))
        print(json.dumps(dict(controller_pid=process.pid, output=str(a.output), optimizer_steps=0)))
    else:
        main(a)
