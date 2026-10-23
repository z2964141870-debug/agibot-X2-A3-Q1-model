#!/usr/bin/env python3
"""Compare a frozen model's local action response with deterministic native physics."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from supermimic.a3_assets import A3Assets, TRACKED, identity
from supermimic.a3_sim import A3Sim, SimCfg
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


HORIZON = 32
READOUTS = (1, 8, 32)
SEED = 20261023
FRACTIONS = (.25, .65)
MAGNITUDES = (.05, .10)
CONTROL_EPS = (1e-4, 2e-4)
REPLAY_TOL = np.array([2e-6] * 42 + [2e-5] * 41)
NORM_FLOORS = {"root": 1e-6, "active_joints": 1e-5}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def vector_comparison(reference, candidate, floor):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite sensitivity vector")
    nr, nc = float(np.linalg.norm(reference)), float(np.linalg.norm(candidate))
    usable = nr > floor and nc > floor
    return {"reference_norm": nr, "candidate_norm": nc,
            "degenerate_reference": nr <= floor, "degenerate_candidate": nc <= floor,
            "cosine": float(np.clip(np.dot(reference, candidate) / (nr * nc), -1., 1.)) if usable else None,
            "norm_ratio": nc / nr if nr > floor else None,
            "relative_l2_error": float(np.linalg.norm(candidate - reference)) / max(nr, floor),
            "absolute_rmse": float(np.sqrt(np.mean((candidate - reference) ** 2)))}


def variants():
    return [{"direction": None, "magnitude": 0., "sign": 0}] + [
        {"direction": direction, "magnitude": magnitude, "sign": sign}
        for direction in range(4) for magnitude in MAGNITUDES for sign in (-1, 1)]


def analyze(raw, active_indices):
    groups = {"root": np.arange(3), "active_joints": 7 + np.asarray(active_indices)}
    rows, linearity, controls = [], [], []
    v = variants()
    for direction in range(4):
        for hindex, horizon in enumerate(READOUTS):
            native_derivatives, model_derivatives, validity = {}, {}, {}
            for magnitude in MAGNITUDES:
                indices = [i for i, x in enumerate(v) if x["direction"] == direction and x["magnitude"] == magnitude]
                minus, plus = indices
                valid = all(int(raw["lengths"][i]) >= horizon and
                            not raw["terminated"][i, :horizon].any() and
                            not raw["timeouts"][i, :horizon].any() for i in indices)
                validity[magnitude] = valid
                native_derivatives[magnitude] = ((raw["native"][plus, horizon - 1].astype(np.float64) -
                                                  raw["native"][minus, horizon - 1].astype(np.float64)) / (2 * magnitude)) if valid else None
                model_derivatives[magnitude] = (raw["model"][plus, horizon - 1].astype(np.float64) -
                                                 raw["model"][minus, horizon - 1].astype(np.float64)) / (2 * magnitude)
                for group, selected in groups.items():
                    item = {"direction": direction, "horizon": horizon, "magnitude": magnitude,
                            "group": group, "valid_physics_response": valid,
                            "clipped_first_action_coordinates": int(raw["clipped"][indices].sum()),
                            "native_vs_model_fd": None, "native_vs_model_jvp": None}
                    if valid:
                        truth = native_derivatives[magnitude][selected]
                        item["native_vs_model_fd"] = vector_comparison(truth, model_derivatives[magnitude][selected], NORM_FLOORS[group])
                        item["native_vs_model_jvp"] = vector_comparison(truth, raw["jvp32"][direction, hindex, selected], NORM_FLOORS[group])
                    rows.append(item)
            for group, selected in groups.items():
                item = {"direction": direction, "horizon": horizon, "group": group,
                        "native_small_vs_large_fd": None,
                        "model_small_vs_large_fd": vector_comparison(model_derivatives[.05][selected], model_derivatives[.10][selected], NORM_FLOORS[group])}
                if all(validity.values()):
                    item["native_small_vs_large_fd"] = vector_comparison(native_derivatives[.05][selected], native_derivatives[.10][selected], NORM_FLOORS[group])
                linearity.append(item)
                ad = raw["jvp64"][direction, hindex, selected]
                ad32 = raw["jvp32"][direction, hindex, selected].astype(np.float64)
                fd_controls = []
                for eindex, epsilon in enumerate(CONTROL_EPS):
                    fd = raw["control_fd64"][eindex, direction, hindex, selected]
                    fd_controls.append({"epsilon": epsilon,
                                        "passed": bool(np.all(np.abs(fd - ad) <= 1e-7 + 1e-3 * np.abs(ad))),
                                        "max_abs_error": float(np.max(np.abs(fd - ad)))})
                controls.append({"direction": direction, "horizon": horizon, "group": group,
                                 "float64_jvp_vs_fd": fd_controls,
                                 "float32_vs_float64_jvp_passed": bool(np.all(np.abs(ad32 - ad) <= 1e-5 + .02 * np.abs(ad))),
                                 "float32_vs_float64_max_abs_error": float(np.max(np.abs(ad32 - ad)))})
    return {"responses": rows, "magnitude_consistency": linearity, "numerical_controls": controls}


def summarize(samples):
    summary = {}
    for horizon in READOUTS:
        summary[str(horizon)] = {}
        for group in NORM_FLOORS:
            selected = [x for sample in samples for x in sample["analysis"]["responses"]
                        if x["horizon"] == horizon and x["group"] == group]
            item = {"total": len(selected), "valid": sum(x["valid_physics_response"] for x in selected)}
            for name in ("native_vs_model_fd", "native_vs_model_jvp"):
                valid = [x[name] for x in selected if x[name] is not None]
                cosines = [x["cosine"] for x in valid if x["cosine"] is not None]
                ratios = [x["norm_ratio"] for x in valid if x["norm_ratio"] is not None]
                item[name] = {"nondegenerate": len(cosines),
                              "median_cosine": float(np.median(cosines)) if cosines else None,
                              "positive_cosine_fraction": float(np.mean(np.asarray(cosines) > 0)) if cosines else None,
                              "median_norm_ratio": float(np.median(ratios)) if ratios else None,
                              "median_relative_l2_error": float(np.median([x["relative_l2_error"] for x in valid])) if valid else None}
            summary[str(horizon)][group] = item
    return summary


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    evaluation = json.loads(args.evaluation.read_text())
    if evaluation["status"] != "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED":
        raise ValueError("Expected completed deterministic evaluation")
    if identity(evaluation["checkpoint"]["path"]) != evaluation["checkpoint"]:
        raise ValueError("Checkpoint changed")
    checkpoint = torch.load(evaluation["checkpoint"]["path"], map_location="cpu", mmap=True, weights_only=False)
    assets = A3Assets(args.root)
    if assets.manifest() != checkpoint["manifest"]["assets"]:
        raise ValueError("Asset contract changed")
    backend_identity = checkpoint["manifest"]["backend_receipt"]
    if identity(backend_identity["path"]) != backend_identity:
        raise ValueError("Backend receipt changed")
    sources = json.loads(Path(backend_identity["path"]).read_text())["source_files"]
    for source in sources:
        if identity(source["path"]) != source:
            raise ValueError("Verified backend changed")
    rows = [row for row in evaluation["rows"] if row["split"] == "training"]
    if len(rows) != 4 or len({row["clip"] for row in rows}) != 4:
        raise ValueError("Expected exactly four distinct training clips")
    rng = np.random.default_rng(SEED)
    samples = []
    for row in rows:
        for key in ("reference", "trace"):
            if identity(row[key]["path"]) != row[key]:
                raise ValueError("Evaluation input changed")
        with np.load(row["trace"]["path"], allow_pickle=False) as raw:
            n = len(raw["state"])
            if raw["state"].shape != (n, 83) or raw["actions"].shape != (n, 29) or n != row["executed_steps"] or n < HORIZON:
                raise ValueError("Trace state/action contract")
            if not np.isfinite(raw["state"]).all() or not np.isfinite(raw["actions"]).all():
                raise ValueError("Nonfinite saved input")
        for fraction in FRACTIONS:
            directions = rng.normal(size=(4, 29))
            directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
            samples.append({"sample": len(samples), "clip": row["clip"], "fraction": fraction,
                            "start": int(np.floor(fraction * (n - HORIZON))),
                            "trace": row["trace"], "reference": row["reference"],
                            "expected_steps": row["expected_steps"], "executed_steps": n,
                            "directions": directions.tolist()})
    plan = {"status": "PREPARED_NOT_EXECUTED", "seed": SEED, "fractions": FRACTIONS,
            "horizon": HORIZON, "readouts": READOUTS, "magnitudes": MAGNITUDES,
            "control_eps": CONTROL_EPS, "replay_abs_tolerance": REPLAY_TOL.tolist(),
            "norm_floors": NORM_FLOORS, "root": str(Path(args.root).resolve()),
            "output": str(args.output.resolve()), "evaluation": identity(args.evaluation),
            "checkpoint": evaluation["checkpoint"], "backend": backend_identity,
            "backend_sources": sources, "entry": identity(__file__),
            "active_indices": assets.active_indices, "samples": samples,
            "max_control_steps": 8 * len(variants()) * HORIZON, "max_physics_substeps": 87040,
            "new_policy_inference": 0, "optimizer_updates": 0,
            "sim": asdict(SimCfg(num_envs=1, threads=1, domain_randomization=False,
                                 observation_noise=False, action_noise=0.)),
            "protocol": "All eight baselines must reproduce saved states before any perturbation. Only first raw action perturbed; actions clamped as native execution; remaining saved actions fixed. Stop a variant at termination and retain it. No sample replacement or automatic retry.",
            "numerical_protocol": "Native float64 integration with original float32 control/state boundaries; learned trajectories/JVP float32; float64 copy of identical weights checks all directions against central differences at 1e-4/2e-4. Float64 control: abs <=1e-7+1e-3*abs(JVP); float32 JVP: abs <=1e-5+.02*abs(float64 JVP). All metrics descriptive, no policy-success gate.",
            "interpretation": "Local open-loop sensitivity at saved visited states; not full closed-loop actor-gradient correctness or a unique cause of failure.",
            "storage_status": "LOCAL_ONLY"}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    print(json.dumps({"status": plan["status"], "plan": identity(args.output / "plan.json"),
                      "samples": [{k: s[k] for k in ("clip", "start", "fraction")} for s in samples],
                      "max_control_steps": plan["max_control_steps"]}, indent=2), flush=True)


def model_rollout(wm, state, actions, delta):
    batch = len(delta)
    s = state.expand(batch, -1)
    xi = torch.zeros(batch, 33, device=s.device, dtype=s.dtype)
    xi[:, 0] = 1.
    states = []
    for t in range(HORIZON):
        a = actions[t].expand(batch, -1)
        if t == 0:
            a = a + delta
        s = wm.step(s, a.clamp(-20., 20.), xi)
        states.append(s)
    return torch.stack(states, dim=1)


def native_rollout(sim, sample, state, actions, previous_action, counter):
    device = sim.device
    sim.reset(sim.ids, torch.tensor([sample["start"]], device=device))
    sim.state.copy_(state)
    sim.previous_action.copy_(previous_action)
    sim.steps.fill_(sample["start"])
    sim.hist[:] = sim.observation()[None]
    states = np.full((HORIZON, 83), np.nan, dtype=np.float32)
    terminated, timeouts = np.zeros(HORIZON, bool), np.zeros(HORIZON, bool)
    length = 0
    for action in actions:
        if counter["control_steps_attempted"] >= 4352:
            raise RuntimeError("J01 control-step budget exhausted")
        counter["control_steps_attempted"] += 1
        result = sim.step(action[None], auto_reset=False)
        counter["control_steps_completed"] += 1
        states[length] = result["state"][0].cpu().numpy()
        terminated[length] = bool(result["terminated"][0])
        timeouts[length] = bool(result["timeout"][0])
        length += 1
        if terminated[length - 1] or timeouts[length - 1]:
            break
    return states, terminated, timeouts, length


def execute(args):
    output = args.output.resolve()
    plan_path = output / "plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["entry"] != identity(__file__) or plan["output"] != str(output):
        raise ValueError("Plan source or output changed")
    if (output / "admission.json").exists() or (output / "receipt.json").exists():
        raise FileExistsError("This diagnostic has already been admitted; no automatic retry")
    for item in [plan["checkpoint"], plan["evaluation"], plan["backend"], *plan["backend_sources"]]:
        if identity(item["path"]) != item:
            raise ValueError("Frozen input/source changed")
    for sample in plan["samples"]:
        for key in ("trace", "reference"):
            if identity(sample[key]["path"]) != sample[key]:
                raise ValueError("Frozen trace/reference changed")
    write_json(output / "admission.json", {"plan": identity(plan_path), "status": "ADMITTED",
                                           "device": args.device, "seed": plan["seed"]})
    started = time.monotonic()
    counter = {"control_steps_attempted": 0, "control_steps_completed": 0}
    receipt = {"status": "FAILED", "plan": identity(plan_path), "samples": [], "native_runs": [],
               "baselines": [], "counter": counter, "optimizer_updates": 0,
               "new_policy_inference": 0, "storage_status": "LOCAL_ONLY"}
    error = None
    sims = {}
    try:
        torch.manual_seed(plan["seed"])
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        assets = A3Assets(plan["root"])
        ck = torch.load(plan["checkpoint"]["path"], map_location="cpu", mmap=True, weights_only=False)
        if assets.manifest() != ck["manifest"]["assets"]:
            raise ValueError("Frozen asset contract changed")
        heads = assets.heads(args.device)
        prepared = []
        for sample in plan["samples"]:
            clip = sample["clip"]
            if clip not in sims:
                lib = MotionLibrary([Path(sample["reference"]["path"])], TRACKED, assets.q0,
                                    device=args.device, active_indices=assets.active_indices)
                cfg = SimCfg(**{**plan["sim"], "episode_steps": lib.num_frames + 1})
                sims[clip] = A3Sim(assets, lib, cfg, heads, seed=plan["seed"])
            sim = sims[clip]
            with np.load(sample["trace"]["path"], allow_pickle=False) as trace:
                saved = torch.tensor(trace["state"], device=args.device, dtype=torch.float32)
                actions = torch.tensor(trace["actions"], device=args.device, dtype=torch.float32)
            t = sample["start"]
            if t:
                state, previous = saved[t - 1:t], actions[t - 1:t].clamp(-20., 20.)
            else:
                qp, qv = sim.lib.ref_qpos_qvel(torch.tensor([0], device=args.device))
                state, previous = torch.cat([qp, qv], -1), torch.zeros(1, 29, device=args.device)
            window = actions[t:t + HORIZON].clone()
            native, term, timeout, length = native_rollout(sim, sample, state, window, previous, counter)
            truth = saved[t:t + HORIZON].cpu().numpy()
            valid = length == HORIZON and not term.any() and not timeout.any()
            difference = np.abs(native.astype(np.float64) - truth.astype(np.float64))
            passed = valid and bool(np.all(difference <= REPLAY_TOL))
            raw_path = output / (f"sample_{sample['sample']:02d}_baseline.npz")
            np.savez_compressed(raw_path, state=native, truth=truth, terminated=term,
                                timeouts=timeout, length=length)
            baseline = {"sample": sample["sample"], "clip": clip, "start": t, "passed": passed,
                        "length": length, "raw": identity(raw_path),
                        "max_position_abs": float(np.nanmax(difference[:, :42])),
                        "max_velocity_abs": float(np.nanmax(difference[:, 42:]))}
            receipt["baselines"].append(baseline)
            print(json.dumps({"phase": "baseline", **baseline}), flush=True)
            if not passed:
                raise ValueError("Saved-state replay failed; no perturbed physics admitted")
            prepared.append((sample, sim, state, window, previous, native, term, timeout))

        wcfg = dict(ck["manifest"]["world_model"])
        wcfg["state_spec"] = StateSpecCfg(**wcfg["state_spec"])
        wm = WorldModel(WorldModelCfg(**wcfg), heads, args.device).eval()
        wm.load_state_dict(ck["wm"], strict=True)
        wm.requires_grad_(False)
        wm64 = WorldModel(WorldModelCfg(**wcfg), heads, args.device).double().eval()
        wm64.load_state_dict(ck["wm"], strict=True)
        wm64.requires_grad_(False)
        for sample, sim, state, window, previous, baseline, base_term, base_timeout in prepared:
            directions = torch.tensor(sample["directions"], device=args.device, dtype=torch.float32)
            deltas = torch.zeros(len(variants()), 29, device=args.device)
            native = np.full((len(variants()), HORIZON, 83), np.nan, dtype=np.float32)
            term = np.zeros((len(variants()), HORIZON), bool)
            timeout = np.zeros_like(term)
            lengths = np.zeros(len(variants()), dtype=np.int64)
            native[0], term[0], timeout[0], lengths[0] = baseline, base_term, base_timeout, HORIZON
            for i, variant in enumerate(variants()[1:], start=1):
                deltas[i] = directions[variant["direction"]] * (variant["magnitude"] * variant["sign"])
                actions = window.clone()
                actions[0] += deltas[i]
                native[i], term[i], timeout[i], lengths[i] = native_rollout(sim, sample, state, actions, previous, counter)
            first = window[0][None] + deltas
            clipped = (first != first.clamp(-20., 20.)).cpu().numpy()
            native_path = output / (f"sample_{sample['sample']:02d}_native.npz")
            np.savez_compressed(native_path, native=native, terminated=term, timeouts=timeout,
                                lengths=lengths, clipped=clipped, deltas=deltas.cpu().numpy(),
                                initial_state=state.cpu().numpy(), saved_actions=window.cpu().numpy(),
                                previous_action=previous.cpu().numpy())
            receipt["native_runs"].append({"sample": sample["sample"], "raw": identity(native_path)})
            with torch.no_grad():
                model = model_rollout(wm, state, window, deltas).cpu().numpy()

            def readout(model, s, a, delta):
                return model_rollout(model, s, a, delta)[:, np.asarray(READOUTS) - 1]

            zeros = torch.zeros(4, 29, device=args.device)
            _, jvp32 = torch.autograd.functional.jvp(lambda d: readout(wm, state, window, d), zeros, directions, strict=True)
            s64, a64 = state.double(), window.double()
            directions64 = torch.tensor(sample["directions"], device=args.device, dtype=torch.float64)
            _, jvp64 = torch.autograd.functional.jvp(lambda d: readout(wm64, s64, a64, d), zeros.double(), directions64, strict=True)
            fd64 = []
            with torch.no_grad():
                for epsilon in CONTROL_EPS:
                    plus = readout(wm64, s64, a64, epsilon * directions64)
                    minus = readout(wm64, s64, a64, -epsilon * directions64)
                    fd64.append(((plus - minus) / (2 * epsilon)).cpu().numpy())
            raw = {"native": native, "model": model, "terminated": term, "timeouts": timeout,
                   "lengths": lengths, "clipped": clipped, "deltas": deltas.cpu().numpy(),
                   "directions64": np.asarray(sample["directions"]), "initial_state": state.cpu().numpy(),
                   "saved_actions": window.cpu().numpy(), "previous_action": previous.cpu().numpy(),
                   "jvp32": jvp32.cpu().numpy(), "jvp64": jvp64.cpu().numpy(),
                   "control_fd64": np.asarray(fd64)}
            if not all(np.isfinite(raw[key]).all() for key in ("model", "jvp32", "jvp64", "control_fd64")):
                raise ValueError("Nonfinite learned response; completed native variants retained separately")
            raw_path = output / (f"sample_{sample['sample']:02d}_responses.npz")
            np.savez_compressed(raw_path, **raw)
            item = {"sample": sample["sample"], "clip": sample["clip"], "start": sample["start"],
                    "raw": identity(raw_path), "terminated_variants": int(term.any(-1).sum()),
                    "timeout_variants": int(timeout.any(-1).sum()), "analysis": analyze(raw, assets.active_indices)}
            receipt["samples"].append(item)
            print(json.dumps({"phase": "response", "sample": sample["sample"], "clip": sample["clip"],
                              "control_steps_completed": counter["control_steps_completed"],
                              "terminated_variants": item["terminated_variants"], "raw": item["raw"]}), flush=True)
        receipt["wm_float32_unchanged"] = all(torch.equal(value.cpu(), ck["wm"][key]) for key, value in wm.state_dict().items())
        if not receipt["wm_float32_unchanged"]:
            raise ValueError("Frozen world model changed")
        receipt["summary"] = summarize(receipt["samples"])
        controls = [c for s in receipt["samples"] for c in s["analysis"]["numerical_controls"]]
        receipt["all_numerical_controls_passed"] = all(c["float32_vs_float64_jvp_passed"] and
                                                      all(x["passed"] for x in c["float64_jvp_vs_fd"]) for c in controls)
        receipt["status"] = "ACTION_RESPONSE_DIAGNOSTIC_COMPLETED"
    except Exception as exc:
        error = repr(exc)
        receipt["error"] = error
    finally:
        for sim in sims.values():
            sim.close()
        counter["physics_substeps_completed"] = 20 * counter["control_steps_completed"]
        receipt["seconds"] = time.monotonic() - started
        write_json(output / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "counter", "seconds")}, indent=2), flush=True)
    if error:
        raise RuntimeError(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prepare:
        if args.evaluation is None or args.root is None:
            parser.error("--prepare requires --evaluation and --root")
        prepare(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
