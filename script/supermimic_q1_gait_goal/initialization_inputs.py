"""Isolate reset inputs and identical-input policy inference, with no env.step."""

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="INITIALIZING", captured_utc=datetime.now(timezone.utc).isoformat(),
                   new_control_transitions=0, new_integration_substeps=0, optimizer_updates=0,
                   policy_inference_calls=0, user_accepted_checkpoint=None)
    env = None
    original_rng = None
    try:
        import torch
        import warp as wp
        sys.path[:0] = [str(HERE.parent / "supermimic_q1_sole"),
                       str(HERE.parent / "supermimic_q1_hp"),
                       str(args.runtime / "src"), str(args.runtime)]
        from check_contract import identity
        from h32_direction import digest
        from initialization_audit import InitializationAudit
        from physical_gate import FIXED
        from worker import prepare
        from types import SimpleNamespace
        from unisupermimic.rl.algos.supermimic import networks
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
        from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig
        from unisupermimic.rl.algos.supermimic.networks import Policy, WorldModel, Critic
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY

        busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if busy:
            raise RuntimeError("GPU has an existing compute process; no inference isolation started")
        torch.set_num_threads(8)
        original_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        torch.manual_seed(32100)
        device = torch.device("cuda:0")
        library = json.loads(args.library.read_text())
        rows = [row for split in library["splits"].values() for row in split if row["name"] in FIXED]
        if len(rows) != 3 or len({row["name"] for row in rows}) != 3:
            raise ValueError("Expected original fixed3 motions")
        for row in rows:
            if identity(row["path"])["sha256"] != row["sha256"]:
                raise ValueError("Original motion identity changed")
        native = args.runtime / "src/unisupermimic"
        source_paths = [Path(__file__), args.checkpoint, args.library, args.runtime / "q1_assets/scene.xml",
                        HERE.parent / "supermimic_q1_hp/worker.py", HERE.parent / "supermimic_q1_sole/initialization_audit.py",
                        native / "rl/algos/supermimic/networks.py",
                        native / "tasks/motion_tracking/q1/supermimic/mdp_base.py",
                        native / "tasks/motion_tracking/q1/supermimic/env.py",
                        native / "tasks/motion_tracking/q1/supermimic/manager_terms.py"]
        before = [identity(path) for path in source_paths]
        receipt["sources_before"] = before
        subset = args.output / "evaluation_library.json"
        subset.write_text(json.dumps({"splits": {"train": rows}}, indent=2) + "\n")
        preparation = SimpleNamespace(runtime=args.runtime, library=subset, output=args.output)
        saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
        cfg = SuperMimicConfig(**{field.name: saved["args"][field.name]
                                 for field in fields(SuperMimicConfig) if field.name in saved["args"]})
        if cfg.hist != 10 or cfg.horizon != 32:
            raise ValueError("Expected original hist10/H32 checkpoint")
        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
        audit = InitializationAudit(receipt.setdefault("initialization_audit", {}))
        reference_mdp = policy = constant_history = constant_goal = None
        parameter_sha = None
        branch_results = []

        def forbid_step(*values, **options):
            raise AssertionError("This isolation forbids every environment control step")

        def arrays(values):
            wp.synchronize()
            torch.cuda.synchronize()
            result = {key: value.detach().float().cpu().numpy() if value.dtype == torch.bfloat16
                      else value.detach().cpu().numpy() for key, value in values.items()}
            result["rng.torch_cpu"] = torch.get_rng_state().cpu().numpy()
            result["rng.torch_cuda"] = torch.cuda.get_rng_state().cpu().numpy()
            return result

        with torch.no_grad():
            for branch in range(2):
                _, env, _ = prepare(preparation, num_envs=16, eval_mode=True)
                env.step = forbid_step
                mdp = env.supermimic_mdp
                if branch == 0:
                    reference_mdp = mdp
                    policy = Policy(mdp.obs_dim, mdp.goal_frame_dim * cfg.goal_frames, hist=cfg.hist,
                                    z_dim=cfg.z_dim, num_joints=22, hidden=cfg.policy_hidden).to(device)
                    # Match the original diagnostic's constructor/RNG path exactly.
                    world_model = WorldModel(hidden=cfg.wm_hidden, num_joints=22, context_dim=26).to(device)
                    context = torch.zeros(1, 26, device=device)
                    context[:, 0] = 1
                    frame = mdp.motion.clip_start[:1]
                    feature_dim = mdp.critic_features(mdp.motion.ref_state(frame), frame,
                        mdp.normalize_context(context)).shape[-1] + mdp.goal_frame_dim * cfg.goal_frames
                    critic = Critic(feature_dim, hidden=cfg.critic_hidden).to(device)
                    for name, module in (("pi", policy), ("wm", world_model), ("critic", critic)):
                        module.load_state_dict(saved[name])
                    parameter_sha = digest(policy)
                    del world_model, critic
                ids = torch.arange(16, device=device)
                clips = ids % 3
                initial_frames = torch.minimum(mdp.motion.clip_start[clips] + 30 * (ids // 3),
                    mdp.motion.clip_end[clips] - cfg.hist - cfg.horizon - 1)
                env.reset_to_frames(initial_frames)
                wp.to_torch(env.backend._device_data.qacc_warmstart).zero_()
                info = env.state.info[SUPERMIMIC_STATE_KEY]
                state = unflatten_state(info["s"])
                own_obs = mdp.obs(state, info["a_prev"], info["xi"], info["f"])
                own_goal = mdp.goal(info["f"], cfg.goal_frames)
                old_obs = reference_mdp.obs(state, info["a_prev"], info["xi"], info["f"])
                old_goal = reference_mdp.goal(info["f"], cfg.goal_frames)
                own_history = own_obs[:, None].repeat(1, cfg.hist, 1)
                if branch == 0:
                    constant_history, constant_goal = own_history.clone(), own_goal.clone()
                audit.begin("initial_after_network_constructors" if branch == 0 else "fresh_existing_networks",
                            dict(policy_sha256=digest(policy), precision=cfg.amp,
                                 reference_mdp_is_owned=reference_mdp is mdp,
                                 policy_training_mode=policy.training,
                                 history="Repeated reset observation; no physical warm-up"))
                values = {"state." + key: info[key] for key in ("s", "f", "xi", "a_prev")}
                values.update({"policy.history": own_history, "policy.goal": own_goal,
                               "ownership.reference_mdp_obs": old_obs,
                               "ownership.owned_mdp_obs": own_obs,
                               "ownership.reference_mdp_goal": old_goal})
                for name in ("qpos", "qvel", "qacc_warmstart", "ctrl", "time", "qacc", "qfrc_applied", "xfrc_applied", "act"):
                    if hasattr(env.backend._device_data, name):
                        values["native." + name] = wp.to_torch(getattr(env.backend._device_data, name))
                audit.record("reset_inputs", arrays(values))
                action_own = policy(own_history, own_goal)
                action_fixed = policy(constant_history, constant_goal)
                torch.manual_seed(32100)
                action_new_seed = policy(constant_history, constant_goal)
                action_repeat = policy(constant_history, constant_goal)
                receipt["policy_inference_calls"] += 4
                audit.record("policy_outputs", arrays({"own_input": action_own,
                    "fixed_input": action_fixed, "fixed_input_changed_rng": action_new_seed,
                    "fixed_input_repeat": action_repeat}))
                audit.finish()
                def matches(first, second):
                    return bool(torch.allclose(first, second, atol=2e-6, rtol=2e-5))
                branch_results.append(dict(branch=branch,
                    owned_and_reference_mdp_obs_match=matches(own_obs, old_obs),
                    owned_and_reference_mdp_goal_match=matches(own_goal, old_goal),
                    identical_inputs_rng_change_match=matches(action_fixed, action_new_seed),
                    identical_inputs_repeat_match=matches(action_fixed, action_repeat),
                    policy_parameters_unchanged=digest(policy) == parameter_sha))
                env.close()
                env.backend.close()
                env = None
        comparison = audit.receipt["traces"][1]["snapshots"]
        input_match = all(value["matches"] for name, value in comparison[0]["comparison"].items()
                          if not name.startswith("rng."))
        output_match = all(value["matches"] for name, value in comparison[1]["comparison"].items()
                           if not name.startswith("rng."))
        if before != [identity(path) for path in source_paths]:
            raise ValueError("Initialization isolation sources changed")
        receipt.update(status="RESET_INPUT_EQUIVALENCE_MEASURED", reset_inputs_match=input_match,
                       cross_environment_policy_outputs_match=output_match, branches=branch_results,
                       sources_after=before, observation_noise_added=False, physical_warmup_steps=0,
                       limits=["No integration: does not test nine-step warm-up, solver determinism or H32 direction.",
                           "Repeated reset observation is synthetic history for input isolation, not production policy evaluation.",
                           "MDP ownership checks apply to exact reset inputs; they do not establish later dynamics equivalence.",
                           "Passing this isolation alone never authorizes a claim that the original H32 mismatch is fixed."])
    except BaseException as error:
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
            env.backend.close()
        if original_rng is not None:
            random.setstate(original_rng[0])
            np.random.set_state(original_rng[1])
            torch.set_rng_state(original_rng[2])
            torch.cuda.set_rng_state_all(original_rng[3])
            receipt["rng_restored"] = True
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "reset_inputs_match",
        "cross_environment_policy_outputs_match", "new_control_transitions", "policy_inference_calls")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
