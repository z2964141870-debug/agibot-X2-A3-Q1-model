"""Independent zero-optimizer reference-activity upper composition probe."""

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
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_pd_probe"),
               str(HERE.parent / "supermimic_q1_gait_comp"), str(HERE.parent / "supermimic_q1_hp"),
               str(HERE.parent / "supermimic_q1_ppo"), str(HERE.parent / "project_tools")]
from worker import process_identity
from resources import identity, save, stamp
from run_trial import Q1Budget
from runtime_guard import guarded_run
from overnight import OPTIMIZATION_DEADLINE
from activity_control import CANDIDATES, UpperActivityBlend


def rollout(env, teacher, thresholds, candidate, progress):
    import numpy as np
    import torch
    from physical import NativeContacts, Fallback, inputs, compose, risk_features, enrich
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state, mat_to_quat
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale

    mdp, motion = env.supermimic_mdp, env.supermimic_mdp.motion
    contact = NativeContacts(env).install()
    try:
        _, info = env.reset_to_frames(motion.clip_start.clone())
        current = info[SUPERMIMIC_STATE_KEY]
        initial = current["s"].cpu().numpy()
        history = None
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        failed, steps = done.clone(), torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        returns = torch.zeros(env.num_envs, device=env.device)
        records = {key: [] for key in ("states", "frames", "alive", "targets", "force", "slip", "alpha", "residual", "reference_speed")}
        scale = torch.tensor([action_scale(name) for name in Q1_JOINT_NAMES], device=env.device)
        gate = Fallback(env.num_envs, thresholds, env.device)
        activity = UpperActivityBlend(env.num_envs, candidate, env.device)
        cap = torch.full((env.num_envs,), .5, device=env.device)
        gate.reset(~done, cap)
        for iteration in range(int(motion.clip_len.max())):
            measured = mdp.obs(unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"])
            history = measured[:, None].repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], measured[:, None]), 1)
            base = env._supermimic_command.default_q + current["xi"][:, 1:23]
            frames = current["f"]
            next_frames = torch.minimum(frames+1, motion.clip_end[motion.clip_id[frames]])
            speed = torch.linalg.vector_norm(motion.root_p[next_frames, :2]-motion.root_p[frames, :2], dim=-1)/.02
            x0, x1 = motion.root_R[frames, :2, 0], motion.root_R[next_frames, :2, 0]
            yaw_rate = torch.atan2(x0[:, 0]*x1[:, 1]-x0[:, 1]*x1[:, 0], (x0*x1).sum(-1))/.02
            cap = activity.update(speed, yaw_rate)
            gate.alpha = cap.clone()
            _, nominal, reference = inputs(teacher, history, mdp.goal(frames, 10), motion.q[next_frames],
                base, scale, cap, gate.filtered, gate.cooldown, cap)
            action, residual = compose(nominal, reference, torch.zeros(env.num_envs, 14, device=env.device), cap, scale)
            targets = base + scale * action
            state = env.step(action)
            tr = state.info[SUPERMIMIC_TRANSITION_KEY]
            alive = ~done
            values = dict(states=tr["s1"], frames=tr["f"]+1, alive=alive, targets=targets,
                          force=contact.last[0], slip=contact.last[1], alpha=cap, residual=residual, reference_speed=speed)
            for key, value in values.items():
                records[key].append(value.cpu().numpy())
            risk = risk_features(tr["s1"], motion.root_p[tr["f"]+1], mat_to_quat(motion.root_R[tr["f"]+1]), *contact.last)
            gate.advance(risk, cap)
            gate.alpha = cap.clone()
            gate.reset(tr["done"].bool(), cap)
            activity.reset(tr["done"].bool())
            returns += torch.where(alive, tr["r"], 0.)
            steps += alive
            failed |= alive & tr["term"]
            done |= alive & tr["done"]
            current = state.info[SUPERMIMIC_STATE_KEY]
            if iteration % 50 == 0:
                progress(iteration)
            if bool(done.all()):
                break
        if not bool(done.all()):
            raise AssertionError("All original clips must terminate")
        trace = {key: np.stack(value) for key, value in records.items()}
        trace.update(initial=initial, steps=steps.cpu().numpy(), failed=failed.cpu().numpy(), returns=returns.cpu().numpy())
        wrists = [mdp.kin.tracked_idx.index(mdp.kin.body_names.index(side+"_wrist_yaw_link")) for side in ("left", "right")]
        shape = trace["states"].shape[:2]
        flat = torch.from_numpy(trace["states"].reshape(-1, 57)).to(env.device)
        points = [mdp.kin.forward(*unflatten_state(block))[0][:, wrists].cpu().numpy() for block in flat.split(8192)]
        trace["wrists"] = np.concatenate(points).reshape(*shape, 2, 3)
        frames = torch.from_numpy(trace["frames"]).to(env.device)
        trace.update(reference_q=motion.q[frames].cpu().numpy(), reference_p=motion.root_p[frames].cpu().numpy(),
                     reference_quat=mat_to_quat(motion.root_R[frames]).cpu().numpy(),
                     reference_wrists=motion.body_p[frames][..., wrists, :].cpu().numpy(),
                     reference_steps=(motion.clip_len-1).cpu().numpy())
        if not bool(np.isfinite(trace["states"]).all()):
            raise FloatingPointError("Nonfinite physical state")
        return enrich(trace, mdp)
    finally:
        contact.close()


def worker(a):
    import importlib.util
    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location("frozen_pd_probe", HERE.parent / "supermimic_q1_pd_probe/probe.py")
    pd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pd)
    from physical import FIXED_DEMOS, summarize, gait_summary, comparison
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
            for repeat in range(3):
                seed = 32000+repeat
                case = run / f"{candidate}_repeat{repeat}"
                case.mkdir()
                env = None
                try:
                    env = pd.prepare_scene(a, library, case, a.runtime / "q1_assets/scene.xml", seed, len(rows))
                    from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
                    teacher = load_policy(env, torch.load(a.checkpoint, map_location=env.device, weights_only=True)).requires_grad_(False).eval()
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    digest = policy_digest(teacher)
                    def progress(frame):
                        status.update(status="RUNNING", candidate=candidate, repeat=repeat, frame=frame, updated_utc=stamp())
                        save(run / "status.json", status)
                    progress(0)
                    trace = rollout(env, teacher, thresholds, candidate, progress)
                    path = a.output / f"{candidate}_repeat{repeat}.npz"
                    np.savez_compressed(path, **trace)
                    summary = summarize(trace, rows)
                    summary.update(gait=gait_summary(trace, rows), repeat=repeat, seed=seed, candidate=candidate,
                                   upper_reference_alpha=dict(low=CANDIDATES[candidate][0], high=CANDIDATES[candidate][1], tau_seconds=.2),
                                   trace=identity(path))
                    if candidate == "baseline":
                        baseline.append(path)
                    else:
                        with np.load(baseline[repeat], allow_pickle=False) as saved:
                            paired = comparison(trace, {key: saved[key] for key in saved.files}, rows)
                        summary["initial_static075"] = paired
                        comparisons.append(paired)
                    if policy_digest(teacher) != digest:
                        raise AssertionError("Frozen teacher changed")
                    save(path.with_suffix(".json"), summary)
                    summaries.append(summary)
                    del teacher, trace
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
        checkpoint=identity(a.checkpoint), scope=a.scope, user_accepted_checkpoint=None,
        limitations="No clip identity in scheduler. Existing validation is searched, not held-out. Three configured seeds are evaluations, not training seeds or bitwise native solver equality."))
    status.update(status="COMPLETED", ended_utc=stamp(), frozen_teacher_unchanged=True)
    save(run / "status.json", status)


def main(a):
    if a.worker:
        try:
            worker(a)
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
    sources = {item["path"]: item for item in previous}
    sources.update({str(p): identity(p) for p in [*HERE.glob("*.py"), a.checkpoint, a.library, a.calibration]})
    source = list(sources.values())
    for item in source:
        if identity(item["path"]) != item:
            raise RuntimeError("Frozen dependency changed: " + item["path"])
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != a.expected_boot:
        raise RuntimeError("Unexpected boot")
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
                raise RuntimeError("Bounded activity probe deadline reached")
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
            raise RuntimeError("Activity probe failed; no automatic restart")
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
        p.error("Use sole experiment root")
    if len(a.code_commit) != 40 or len(set(a.candidates)) != len(a.candidates) or a.candidates[0] != "baseline":
        p.error("Require published40-digit source and unique baseline-first candidates")
    if time.time() >= OPTIMIZATION_DEADLINE-2400:
        p.error("Insufficient bounded exploration window")
    if a.detach:
        a.output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", str(Path(__file__)), *[v for v in sys.argv[1:] if v != "--detach"], "--controller"]
        with (ROOT / "logs" / (a.output.name+"_controller.log")).open("x") as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(a.output / "launch.json", dict(controller_pid=process.pid, started_utc=stamp(), optimizer_steps=0))
        print(json.dumps(dict(controller_pid=process.pid, output=str(a.output), optimizer_steps=0)))
    else:
        main(a)
