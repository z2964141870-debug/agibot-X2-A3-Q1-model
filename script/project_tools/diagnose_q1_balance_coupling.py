"""Attribute UR01 balance changes and check a frozen-teacher residual candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "supermimic_q1_hp"),
               str(HERE.parent / "supermimic_q1_ppo")]
from worker import identity, now, prepare, process_identity, save
from run_trial import Q1Budget
from runtime_guard import guarded_run
from diagnose_q1_upper_reference import policy_digest


def load_trace(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def input_history(trace, mdp, default, scale, device, policy=None, reference_arms=False):
    import torch
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state

    context = torch.zeros(len(trace["steps"]), mdp.context_dim, device=device)
    context[:, 0] = 1.
    previous = torch.zeros(len(context), 22, device=device)
    history = None
    for step in range(int(trace["steps"].max())):
        state = trace["initial"] if step == 0 else trace["states"][step - 1]
        # Finished rows have reset states and no saved reset context/history.
        # Their synthetic inputs are excluded from every reported measurement.
        frames = (mdp.motion.clip_start + step).minimum(mdp.motion.clip_end - 1)
        obs = mdp.obs(unflatten_state(torch.as_tensor(state, device=device)),
                      previous, context, frames)
        history = (obs[:, None].repeat(1, 10, 1) if history is None else
                   torch.cat((history[:, 1:], obs[:, None]), 1))
        yield step, history, mdp.goal(frames, 10)
        if policy is None:
            previous = (torch.as_tensor(trace["targets"][step], device=device) - default) / scale
        else:
            # Inverting rounded PD targets loses action bits at BF16 thresholds.
            # Reproduce the original deterministic action recursion instead.
            previous = policy(history, mdp.goal(frames, 10)).float().clamp(-20, 20)
            if reference_arms:
                from q1_upper_reference import reference_centered_actions

                previous = reference_centered_actions(previous, mdp.motion.q[frames + 1],
                                                     default, scale, 0.).clamp(-20, 20)


def attribution(env, policy, args, progress):
    import numpy as np
    import torch
    from q1_reference_balance import UPPER_OBSERVATIONS, swap_upper_history

    mdp = env.supermimic_mdp
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale
    scale = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], device=env.device)
    default = env._supermimic_command.default_q
    result, maximum_error = [], 0.
    for repeat in range(3):
        traces = {name: load_trace(args.source / f"{name}_repeat{repeat}.npz")
                  for name in ("RF11", "reference")}
        inputs = {name: input_history(t, mdp, default, scale, env.device,
                                    policy, name == "reference")
                  for name, t in traces.items()}
        sums = {direction: {key: np.zeros(len(traces["RF11"]["steps"]))
                for key in (*UPPER_OBSERVATIONS, "all_upper", "total_closed_loop")}
                for direction in ("inject_into_RF11", "remove_from_reference")}
        counts = np.zeros(len(traces["RF11"]["steps"]), dtype=int)
        for left, right in zip(inputs["RF11"], inputs["reference"]):
            step, base_h, goal = left
            if step != right[0] or not torch.equal(goal, right[2]):
                raise AssertionError("Counterfactual goals/frames differ")
            arm_h = right[1]
            active = step < np.minimum(traces["RF11"]["steps"], traces["reference"]["steps"])
            means = {name: policy(h, goal).float().clamp(-20, 20)
                     for name, h in (("RF11", base_h), ("reference", arm_h))}
            for name, mean in means.items():
                observed = torch.as_tensor(traces[name]["targets"][step], device=env.device)
                error = ((default + scale * mean)[:, :14] - observed[:, :14]).abs()
                valid = step < traces[name]["steps"]
                maximum_error = max(maximum_error, float(error[valid].max()))
            for direction, own, donor, original in (
                ("inject_into_RF11", base_h, arm_h, means["RF11"]),
                ("remove_from_reference", arm_h, base_h, means["reference"]),
            ):
                for field in (*UPPER_OBSERVATIONS, "all_upper", "total_closed_loop"):
                    if field == "total_closed_loop":
                        changed = means["reference" if direction == "inject_into_RF11" else "RF11"]
                    else:
                        fields = tuple(UPPER_OBSERVATIONS) if field == "all_upper" else (field,)
                        changed = policy(swap_upper_history(own, donor, fields), goal).float().clamp(-20, 20)
                    delta = ((changed[:, :14] - original[:, :14]) * scale[:14]).square().mean(-1)
                    sums[direction][field] += delta.cpu().numpy() * active
            counts += active
            if step % 50 == 0:
                progress(phase="saved_attribution", repeat=repeat, frame=step)
        if not bool((counts > 0).all()):
            raise AssertionError("Empty paired prefix")
        rows = json.loads((args.source / "results.json").read_text())["rows"]
        clips = [dict(name=row["name"], split=row["split"], common_steps=int(counts[i]),
                 **{direction: {k: float(np.sqrt(v[i] / counts[i])) for k, v in group.items()}
                    for direction, group in sums.items()}) for i, row in enumerate(rows)]
        result.append(dict(repeat=repeat, per_clip=clips,
                      validation={direction: {k: float(np.mean([c[direction][k] for c in clips
                                   if c["split"] == "validation"])) for k in group}
                                  for direction, group in sums.items()}))
    if maximum_error > 2e-6:
        raise AssertionError(f"BF16 saved lower-target reconstruction mismatch: {maximum_error} rad")
    payload = dict(repeats=result, maximum_saved_lower_target_reconstruction_error_rad=maximum_error,
                   units="Per-clip RMS PD-target change across 14 lower coordinates, in radians; validation is mean across original30 clips.",
                   limits="Synthetic observation intervention at identical goals, histories otherwise unchanged. Quantifies input dependence, not a reachable trajectory or causal percentage of closed-loop root regression. Individual effects do not add linearly.",
                   physics_integration_steps=0, optimizer_updates=0)
    save(args.output / "attribution.json", payload)
    return payload


class TargetReplay:
    """Diagnostic open-loop lower commands; deliberately removes lower feedback."""

    def __init__(self, env, trace, reference_arms):
        import torch
        from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale

        self.env, self.reference_arms = env, reference_arms
        self.targets = torch.as_tensor(trace["targets"], device=env.device)
        self.scale = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], device=env.device)
        self.step = 0

    def __call__(self, history, goal):
        import torch
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY

        del history, goal
        current = self.env.state.info[SUPERMIMIC_STATE_KEY]
        motion = self.env.supermimic_mdp.motion
        frames = current["f"]
        base = self.env._supermimic_command.default_q + current["xi"][:, 1:23]
        target = self.targets[min(self.step, len(self.targets) - 1)].clone()
        if self.reference_arms:
            next_frames = (frames + 1).minimum(motion.clip_end[motion.clip_id[frames]])
            target[:, 14:] = motion.q[next_frames, 14:]
        self.step += 1
        action = (target - base) / self.scale
        if not bool(torch.isfinite(action).all()) or bool((action.abs() > 20).any()):
            raise AssertionError("Replay target cannot be transmitted without clipping")
        return action


def candidate_contract(env, teacher, args):
    import torch
    from q1_reference_balance import ReferenceBalance, lower_gaussian_kl
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale

    trace = load_trace(args.source / "RF11_repeat0.npz")
    scale = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], device=env.device)
    base = env._supermimic_command.default_q
    _, history, goal = next(input_history(trace, env.supermimic_mdp, base, scale, env.device))
    reference = env.supermimic_mdp.motion.q[env.supermimic_mdp.motion.clip_start + 1]
    model = ReferenceBalance(teacher, scale).to(env.device).eval()
    checks = []
    for count in (16, 4096):
        ids = torch.arange(count, device=env.device) % len(history)
        h, g, ref = history[ids], goal[ids], reference[ids]
        physical_base = base[ids] if base.ndim == 2 else base.expand(count, -1)
        with torch.no_grad():
            nominal = model.teacher_actions(h, g)
            mean = model(h, g, ref, physical_base)
        torch.testing.assert_close(mean[:, :14], nominal[:, :14], atol=0, rtol=0)
        upper_error = float(((physical_base + scale * mean)[:, 14:] - ref[:, 14:]).abs().max())
        if upper_error > 1e-6:
            raise AssertionError("Candidate arm target mapping changed")
        checks.append(dict(batch=count, lower_mean_max_error=0., upper_target_max_error_rad=upper_error,
                           lower_gaussian_kl_max=float(lower_gaussian_kl(mean, nominal).max())))
    save(args.output / "candidate_contract.json", dict(checks=checks,
         trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
         frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
         lower_residual_radius_rad=model.residual_radius, optimizer_updates=0,
         limits="16/4096 are forward batch sizes on tiled saved inputs, not environment counts or trained success. Zero head equals UR01 reference-arm candidate; its known instability is retained."))
    del model


def worker(args):
    import numpy as np
    import torch
    from evaluation import FIXED_DEMOS, clip_metrics, compare, rollout, summarize

    torch.set_num_threads(8)
    (args.output / "run").mkdir(exist_ok=False)
    status = dict(status="INITIALIZING", started_utc=now(), optimizer_updates=0,
                  process=process_identity())
    def progress(**values):
        status.update(status="RUNNING", updated_utc=now(), **values)
        save(args.output / "run/status.json", status)
    progress(phase="initialization")
    env = None
    try:
        original = json.loads((args.source / "results.json").read_text())
        rows = original["rows"]
        if len(rows) != 33 or sum(r["split"] == "validation" for r in rows) != 30:
            raise ValueError("Retain UR01 original30 validation and three fixed demos")
        library = args.output / "evaluation_library.json"
        save(library, {"splits": {"train": rows}})
        _, env, _ = prepare(SimpleNamespace(runtime=args.runtime, library=library,
                                           output=args.output), num_envs=33, eval_mode=True)
        from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
        teacher = load_policy(env, torch.load(args.checkpoint, map_location=env.device, weights_only=True))
        before = policy_digest(teacher)
        if args.saved_attribution:
            saved = json.loads(args.saved_attribution.read_text())
            if saved["maximum_saved_lower_target_reconstruction_error_rad"] != 0.:
                raise AssertionError("Only an exact deterministic reconstruction may be reused")
            prior = json.loads((args.saved_attribution.parent / "plan.json").read_text())
            for item in prior["source"]:
                if Path(item["path"]) == args.checkpoint or Path(item["path"]).parent == args.source:
                    if identity(item["path"]) != item:
                        raise AssertionError("Saved attribution input changed")
            save(args.output / "attribution.json", dict(saved,
                 reused_from=identity(args.saved_attribution)))
        else:
            with torch.no_grad():
                attribution(env, teacher, args, progress)
        progress(phase="candidate_contract")
        candidate_contract(env, teacher, args)
        results = []
        for repeat in range(3):
            source = load_trace(args.source / f"RF11_repeat{repeat}.npz")
            pair = {}
            for condition, changed in (("replay_RF11", False), ("replay_reference_arms", True)):
                trace = rollout(env, TargetReplay(env, source, changed),
                                progress=lambda frame: progress(phase="command_replay", repeat=repeat,
                                                               condition=condition, frame=frame))
                error = 0.
                for i, count in enumerate(trace["steps"]):
                    count = min(int(count), int(source["steps"][i]))
                    error = max(error, float(np.max(np.abs(trace["targets"][:count, i, :14]
                                                - source["targets"][:count, i, :14]))))
                if error > 1e-6:
                    raise AssertionError("Lower command playback changed")
                np.savez_compressed(args.output / f"{condition}_repeat{repeat}.npz", **trace)
                summary = summarize(trace, rows)
                summary["fixed_demos"] = [dict(name=r["name"], **clip_metrics(trace, i))
                                           for i, r in enumerate(rows) if r["name"] in FIXED_DEMOS]
                summary["lower_target_max_playback_error_rad"] = error
                save(args.output / f"{condition}_repeat{repeat}.json", summary)
                pair[condition] = trace
            paired = compare(pair["replay_reference_arms"], pair["replay_RF11"], rows)
            reproduction = compare(pair["replay_RF11"], source, rows)
            results.append(dict(repeat=repeat, arm_intervention=paired,
                                playback_vs_saved_closed_loop=reproduction))
            save(args.output / "physical_replay.json", dict(repeats=results,
                 limits="Both conditions replay the identical recorded RF11 lower PD targets, with lower feedback disabled. Only arm targets differ. This measures physical coupling under open-loop lower commands, not a deployable balance strategy or an additive explanation of UR01 regression."))
        if policy_digest(teacher) != before:
            raise AssertionError("Teacher or normalizer changed")
        status.update(status="COMPLETED", completed_utc=now(), frozen_teacher_unchanged=True,
                      result=identity(args.output / "physical_replay.json"))
    except BaseException as error:
        status.update(status="FAILED", ended_utc=now(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
            env.backend.close()
        save(args.output / "run/status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "source", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--saved-attribution", type=Path,
                        help="Reuse completed exact saved-history attribution from a prior nontraining stage")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), HERE / "q1_reference_balance.py", HERE / "q1_upper_reference.py",
               args.checkpoint, args.source / "results.json"]
    sources.extend(args.source / f"{name}_repeat{i}.npz"
                   for name in ("RF11", "reference") for i in range(3))
    if args.saved_attribution:
        sources.extend((args.saved_attribution, args.saved_attribution.parent / "plan.json"))
    plan = dict(status="RUNNING", started_utc=now(), controller=process_identity(),
                source=[identity(p) for p in sources], optimizer_updates=0,
                protocol="UR02: three paired saved-history attributions, zero-head forward16/4096 contracts, three paired33-env lower-command replay interventions. No training, reward/control/reference changes or validation tuning.")
    save(args.output / "plan.json", plan)
    save(args.output / "status.json", plan)
    class DiagnosticBudget(Q1Budget):
        def check(self):
            value = super().check()
            if time.monotonic() - self.started > 1500:
                raise RuntimeError("Nontraining diagnostic exceeded 25 minutes")
            return value
    budget = DiagnosticBudget(SimpleNamespace(output=args.output, runtime=args.runtime,
                                               mode="diagnosis", expected_oom_kill=0))
    command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"),
               "-u", str(Path(__file__)), *sys.argv[1:], "--worker"]
    try:
        guard = guarded_run(command, Path("/opt/data/private/ZHY"), budget,
                            args.output / "guard.json", Path("/opt/data/private/ZHY/logs") / (args.output.name + ".log"),
                            grace_s=45, maximum_idle_s=900, progress_path=args.output / "run/status.json")
        if guard["status"] != "WORKER_COMPLETED" or guard["cleanup"]["worker_return_code"] != 0:
            raise RuntimeError("Diagnostic worker guard: " + guard["status"])
        observed = json.loads((args.output / "run/status.json").read_text())
        if observed["status"] != "COMPLETED":
            raise RuntimeError("Diagnostic worker not completed")
        for item in plan["source"]:
            if identity(item["path"]) != item:
                raise AssertionError("Diagnostic input changed")
        plan.update(status="COMPLETED", result=observed["result"])
    except BaseException as error:
        plan.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        plan["ended_utc"] = now()
        save(args.output / "status.json", plan)


if __name__ == "__main__":
    main()
