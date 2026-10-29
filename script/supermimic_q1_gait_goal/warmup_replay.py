"""Locate the first native warm-up divergence with identical replayed controls."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="INITIALIZING", captured_utc=datetime.now(timezone.utc).isoformat(),
                   new_control_transitions=0, new_integration_substeps=0,
                   policy_inference_calls=0, optimizer_updates=0, h32_direction_evaluated=False,
                   user_accepted_checkpoint=None)
    env = None
    try:
        import torch
        import warp as wp
        sys.path[:0] = [str(HERE.parent / "supermimic_q1_sole"),
                       str(HERE.parent / "supermimic_q1_hp"), str(args.runtime / "src"), str(args.runtime)]
        from check_contract import identity
        from h32_direction import digest
        from initialization_audit import InitializationAudit
        from physical_gate import FIXED
        from worker import prepare
        from unisupermimic.rl.algos.supermimic import networks
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
        from unisupermimic.rl.algos.supermimic.networks import Policy
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY

        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("Existing GPU compute process; no replay launched")
        torch.set_num_threads(8)
        library = json.loads(args.library.read_text())
        rows = [row for split in library["splits"].values() for row in split if row["name"] in FIXED]
        if len(rows) != 3 or len({row["name"] for row in rows}) != 3:
            raise ValueError("Expected original fixed3 clips")
        native = args.runtime / "src/unisupermimic"
        paths = [Path(__file__), args.checkpoint, args.library, args.runtime / "q1_assets/scene.xml",
                 HERE.parent / "supermimic_q1_hp/worker.py", HERE.parent / "supermimic_q1_sole/initialization_audit.py",
                 native / "rl/algos/supermimic/networks.py",
                 native / "tasks/motion_tracking/q1/supermimic/env.py",
                 native / "tasks/motion_tracking/q1/supermimic/manager_terms.py"]
        for row in rows:
            if identity(row["path"])["sha256"] != row["sha256"]:
                raise ValueError("Original reference changed")
            paths.append(Path(row["path"]))
        before = [identity(path) for path in paths]
        receipt["sources_before"] = before
        subset = args.output / "evaluation_library.json"
        subset.write_text(json.dumps({"splits": {"train": rows}}, indent=2) + "\n")
        preparation = SimpleNamespace(runtime=args.runtime, library=subset, output=args.output)
        saved = torch.load(args.checkpoint, map_location="cuda:0", weights_only=True)
        cfg = saved["args"]
        if cfg["hist"] != 10 or cfg["horizon"] != 32:
            raise ValueError("Expected hist10/H32 checkpoint")
        networks.AMP_DTYPE = torch.bfloat16 if cfg["amp"] == "bf16" else None
        policy = None
        actions = []
        audit = InitializationAudit(receipt.setdefault("initialization_audit", {}))

        def snapshot(values):
            wp.synchronize()
            torch.cuda.synchronize()
            return {key: value.detach().float().cpu().numpy() if value.dtype == torch.bfloat16
                    else value.detach().cpu().numpy() for key, value in values.items()}

        with torch.no_grad():
            for branch in range(2):
                _, env, _ = prepare(preparation, num_envs=16, eval_mode=True)
                mdp = env.supermimic_mdp
                if policy is None:
                    policy = Policy(mdp.obs_dim, mdp.goal_frame_dim * cfg["goal_frames"], hist=10,
                                    z_dim=cfg["z_dim"], num_joints=22, hidden=cfg["policy_hidden"]).to(env.device)
                    policy.load_state_dict(saved["pi"])
                    policy_hash = digest(policy)
                ids = torch.arange(16, device=env.device)
                clips = ids % 3
                starts = torch.minimum(mdp.motion.clip_start[clips] + 30 * (ids // 3),
                                       mdp.motion.clip_end[clips] - 10 - 32 - 1)
                env.reset_to_frames(starts)
                wp.to_torch(env.backend._device_data.qacc_warmstart).zero_()
                audit.begin("closed_loop_reference" if branch == 0 else "identical_action_replay",
                            dict(policy_sha256=digest(policy), precision=cfg["amp"],
                                 mdp_is_environment_owned=mdp is env.supermimic_mdp,
                                 exact_action_replay=bool(branch), maximum_control_steps=9))
                model = env.backend._device_model
                model_arrays = {}
                for name in ("actuator_gainprm", "actuator_biasprm", "dof_damping", "dof_armature",
                             "geom_friction", "body_mass", "body_inertia", "body_ipos"):
                    if hasattr(model, name):
                        model_arrays["model." + name] = wp.to_torch(getattr(model, name))
                if not all("model." + name in model_arrays for name in
                           ("actuator_gainprm", "actuator_biasprm", "geom_friction", "body_mass")):
                    raise ValueError("Cannot inspect required native model coefficients")
                history = None
                for step in range(10):
                    info = env.state.info[SUPERMIMIC_STATE_KEY]
                    obs = mdp.obs(unflatten_state(info["s"]), info["a_prev"], info["xi"], info["f"])
                    history = obs[:, None].repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
                    values = {"state." + key: info[key] for key in ("s", "f", "xi", "a_prev")}
                    values.update({"policy.history": history, **model_arrays})
                    for name in ("qpos", "qvel", "qacc_warmstart", "qacc", "ctrl", "time", "qfrc_applied", "xfrc_applied", "act"):
                        if hasattr(env.backend._device_data, name):
                            values["native." + name] = wp.to_torch(getattr(env.backend._device_data, name))
                    audit.record(f"state_{step:02d}", snapshot(values))
                    if branch and audit.trace["first_state_action_difference"] is not None:
                        break
                    if step == 9:
                        continue
                    goal = mdp.goal(info["f"], cfg["goal_frames"])
                    action = policy(history, goal)
                    receipt["policy_inference_calls"] += 1
                    applied = action if branch == 0 else actions[step]
                    audit.record(f"action_{step:02d}", snapshot({"policy.computed_action": action,
                                                               "control.applied_action": applied, "policy.goal": goal}))
                    if branch and audit.trace["first_state_action_difference"] is not None:
                        break
                    if branch == 0:
                        actions.append(action.clone())
                    output = env.step(applied)
                    receipt["new_control_transitions"] += 16
                    receipt["new_integration_substeps"] += 64
                    if bool(output.info[SUPERMIMIC_TRANSITION_KEY]["done"].any()):
                        raise ValueError("A fixed warm-up sample terminated; no H32 comparison performed")
                audit.finish()
                env.close()
                env.backend.close()
                env = None
        trace = audit.receipt["traces"][1]
        unchanged = digest(policy) == policy_hash and before == [identity(path) for path in paths]
        if not unchanged:
            raise AssertionError("Warm-up isolation changed policy or frozen sources")
        receipt.update(status="WARMUP_FIXED_ACTION_REPLAY_MEASURED",
                       warmup_matches_at_original_tolerance=trace["complete"] and trace["first_state_action_difference"] is None,
                       first_divergence=trace["first_state_action_difference"], source_parameters_unchanged=True,
                       sources_after=before, early_stop_on_first_divergence=True,
                       limits=["One native fixed-action replay, no gradient, optimizer or H32 directional intervention.",
                           "The second branch applies saved first-branch controls; computed policy actions are diagnostic only.",
                           "A replay divergence localizes this preparation but does not establish a single cause of failed gait training.",
                           "Selected arrays are observed; this is not a complete backend-state serialization."])
    except BaseException as error:
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
            env.backend.close()
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "warmup_matches_at_original_tolerance",
        "first_divergence", "new_control_transitions", "policy_inference_calls")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
