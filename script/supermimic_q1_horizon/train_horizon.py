"""R03 continuation loop with a single registered imagination-horizon change."""

from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import signal
import time

import horizon_contract as contract


def run(plan, arm):
    import numpy as np
    import torch
    from contact_protocol import ContactCorrectedAssets
    from supermimic.data.replay_buffer import ReplayBuffer
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
    from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
    from supermimic.q1_assets import TRACKED
    from supermimic.q1_sim import Q1Sim, SimCfg
    from supermimic.robot_spec import ACT_DIM, STATE_DIM, XI_DIM
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.wm_trainer import WorldModelTrainer, WMTrainerCfg
    from train import health, atomic_json

    root, output = Path(plan["root"]), Path(arm["run"])
    if output.exists():
        raise FileExistsError(output)
    steps, seed = plan["choice"]["steps_per_arm"], plan["choice"]["seed"]
    evidence = contract.validated_parent(plan["parent_experiment"], Path(plan["parent_experiment"]) / "run")
    if evidence != plan["parent"]:
        raise ValueError("Registered parent lineage changed")
    parent = json.loads(Path(evidence["admission"]["path"]).read_text())
    baseline = health(root, plan["baseline_health"]["oom_kill"])
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assets = ContactCorrectedAssets(root)
    backend = json.loads(Path(parent["backend_receipt"]["path"]).read_text())
    if backend["status"] != "BACKEND_AND_GRADIENT_CONTRACT_PASSED" or backend["assets"] != assets.manifest():
        raise ValueError("Corrected backend contract differs")
    for item in backend["source_files"] + [parent["backend_receipt"], parent["reference_receipt"], parent["motion_manifest"], *parent["train_clips"]]:
        contract.check_identity(item)
    device = "cuda:0"
    heads = assets.heads(device)
    lib = MotionLibrary([Path(item["path"]) for item in parent["train_clips"]], TRACKED,
                        assets.q0, device=device, active_indices=assets.active_indices)
    scfg = SimCfg(**parent["sim"])
    sim = Q1Sim(assets, lib, scfg, heads, seed)
    try:
        buffer = ReplayBuffer(scfg.num_envs, 1000, STATE_DIM, ACT_DIM, XI_DIM, device)
        wcfg = WorldModelCfg(**{**parent["world_model"], "state_spec": StateSpecCfg(**parent["world_model"]["state_spec"])})
        pcfg, ccfg = PolicyCfg(**parent["policy"]), CriticCfg(**parent["critic"])
        wtcfg = WMTrainerCfg(**parent["wm_trainer"])
        ptcfg = PolicyTrainerCfg(**{**parent["policy_trainer"], "horizon": arm["horizon"]})
        wm = WorldModel(wcfg, heads, device)
        policy, critic = Policy(pcfg).to(device), Critic(ccfg, pcfg=pcfg).to(device)
        wt, pt = WorldModelTrainer(wm, lib, wtcfg), PolicyTrainer(wm, policy, critic, lib, ptcfg)
        binding = {"plan": contract.identity(Path(plan["output"]) / "plan.json"),
                   "arm": arm["name"], "horizon": arm["horizon"],
                   "sources": plan["horizon_sources"], "entry": contract.identity(__file__)}
        manifest = {key: parent[key] for key in contract.CONTRACT_KEYS}
        manifest.update({"status": "ADMITTED_NOT_TRAINED", "assets": assets.manifest(), "sim": asdict(scfg),
            "world_model": asdict(wcfg), "policy": asdict(pcfg), "critic": asdict(ccfg),
            "wm_trainer": asdict(wtcfg), "policy_trainer": asdict(ptcfg),
            "args": {"root": str(root), "pair": plan["parent_experiment"],
                "parent_run": str(Path(plan["parent_experiment"]) / "run"), "output": str(output),
                "steps": steps, "seed": seed, "code_commit": plan["code_commit"], "device": device},
            "code_commit": plan["code_commit"], "baseline_health": baseline,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
            "training_entry": contract.identity(__file__),
            "source_files": [contract.identity(p) for p in sorted([
                *(root / "script/supermimic_for_q1").rglob("*.py"),
                *(root / "script/supermimic_q1_extension").rglob("*.py")])],
            "warmup_steps": parent["warmup_steps"], "model_warmup_updates": parent["model_warmup_updates"],
            "paper_deviations": parent["paper_deviations"][:]})
        manifest["initialization"] = contract.load_parent(evidence, manifest, (wm, policy, critic),
            (wt.opt, pt.opt, pt.critic_opt), sim, binding)
        manifest["learning_rate_step_offset"] = manifest["initialization"]["learning_rate_step_offset"]
        gc.collect()
        health(root, baseline["oom_kill"])
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "admission.json", manifest)
        step = updates = policy_updates = 0
        started = time.monotonic()
        stop = False

        def request_stop(signum, frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

        def save(name):
            destination = output / name
            temporary = destination.with_suffix(".tmp")
            torch.save({"step": step, "wm_updates": updates, "policy_updates": policy_updates,
                "total_collection_rounds": manifest["learning_rate_step_offset"] + step,
                "wm": wm.state_dict(), "policy": policy.state_dict(), "critic": critic.state_dict(),
                "wm_opt": wt.opt.state_dict(), "policy_opt": pt.opt.state_dict(), "critic_opt": pt.critic_opt.state_dict(),
                "bin_return": sim.bin_return, "bin_memory": sim.bin_memory,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "manifest": manifest}, temporary)
            os.replace(temporary, destination)
            with destination.open("rb") as stream:
                os.fsync(stream.fileno())
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

        status, error = "FAILED", None
        try:
            save("step_000000.pt")
            sim.reset(sim.ids)
            accumulated = 0.
            last_metrics = {}
            for step in range(1, steps + 1):
                with torch.no_grad():
                    obs, goal = sim.policy_inputs()
                    result = sim.step(policy(obs, goal), buffer)
                if step >= manifest["warmup_steps"]:
                    accumulated += .4
                    if accumulated >= 1.:
                        accumulated -= 1.
                        scheduled = manifest["learning_rate_step_offset"] + step
                        lr = 2e-4 * (.03 + .97 * .5 * (1 + math.cos(math.pi * min(scheduled / 250000., 1.))))
                        wt.set_lr(lr)
                        pt.set_lr(lr)
                        wm_metrics = wt.update(buffer)
                        updates += 1
                        if updates > manifest["model_warmup_updates"]:
                            last_metrics = pt.update(buffer)
                            policy_updates += 1
                        if not all(np.isfinite(v) for v in [*wm_metrics.values(), *last_metrics.values()]):
                            raise RuntimeError("Nonfinite horizon-continuation metrics")
                        last_metrics = {**last_metrics, **{"wm/" + k: v for k, v in wm_metrics.items()}}
                if step % 25 == 0:
                    row = {"status": "RUNNING", "step": step, "env_steps": step * scfg.num_envs,
                        "total_collection_rounds": manifest["learning_rate_step_offset"] + step,
                        "wm_updates": updates, "policy_updates": policy_updates,
                        "seconds": time.monotonic() - started, "pid": os.getpid(),
                        "reward_mean": float(result["reward"].mean()),
                        "termination_fraction": float(result["terminated"].float().mean()),
                        "joint_rmse_rad": float(result["joint_rmse"].mean()),
                        "health": health(root, baseline["oom_kill"]), **last_metrics}
                    atomic_json(output / "status.json", row)
                    with (output / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    print(json.dumps(row, allow_nan=False), flush=True)
                if step % 500 == 0:
                    save(f"step_{step:06d}.pt")
                if stop:
                    status = "STOPPED_SIGNAL"
                    break
            else:
                status = "TRAINING_FINISHED_PENDING_EVALUATION"
            save("final.pt")
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            receipt = {"status": status, "step": step, "env_steps": step * scfg.num_envs,
                "wm_updates": updates, "policy_updates": policy_updates,
                "total_collection_rounds": manifest["learning_rate_step_offset"] + step,
                "seconds": time.monotonic() - started, "error": error,
                "artifacts": [contract.identity(p) for p in sorted(output.glob("*.pt"))],
                "storage_status": "LOCAL_ONLY", "not_hardware_validated": True}
            atomic_json(output / "receipt.json", receipt)
            atomic_json(output / "status.json", receipt)
            print(json.dumps(receipt, allow_nan=False), flush=True)
    finally:
        sim.close()
