#!/usr/bin/env python3
"""Compare R01 model action sensitivity with preserved-protocol Q1 physics."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.q1_assets import Q1Assets, TRACKED, identity
from supermimic.q1_sim import Q1Sim, SimCfg
from supermimic.robot_spec import ACT_DIM, STATE_DIM, XI_DIM
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


HORIZON = 32
READOUTS = (1, 8, 32)
MAGNITUDES = (.05, .10)
VARIANTS = [(None, 0., 0)] + [(d, m, sign) for d in range(4)
                             for m in MAGNITUDES for sign in (-1, 1)]


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def compare(truth, predicted, floor):
    truth, predicted = np.asarray(truth, float), np.asarray(predicted, float)
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("Nonfinite action response")
    nt, npred = float(np.linalg.norm(truth)), float(np.linalg.norm(predicted))
    return {"physics_norm": nt, "model_norm": npred,
            "cosine": float(np.clip(np.dot(truth, predicted) / (nt * npred), -1., 1.))
            if nt > floor and npred > floor else None,
            "norm_ratio": npred / nt if nt > floor else None,
            "relative_l2_error": float(np.linalg.norm(predicted - truth)) / max(nt, floor)}


def prepare(args):
    evaluation = json.loads(args.evaluation.read_text())
    if identity(evaluation["checkpoint"]["path"]) != evaluation["checkpoint"]:
        raise ValueError("Checkpoint changed")
    ck = torch.load(evaluation["checkpoint"]["path"], map_location="cpu", mmap=True, weights_only=False)
    assets = Q1Assets(args.root)
    if assets.manifest() != ck["manifest"]["assets"]:
        raise ValueError("Original R01 asset protocol changed")
    backend = ck["manifest"]["backend_receipt"]
    sources = json.loads(Path(backend["path"]).read_text())["source_files"]
    for item in [backend, *sources]:
        if identity(item["path"]) != item:
            raise ValueError("Original R01 backend changed")
    rows = [r for r in evaluation["rows"] if r["split"] == "training"]
    if len(rows) != 2:
        raise ValueError("Expected the two fixed R01 training clips")
    rng = np.random.default_rng(20261023)
    samples = []
    for row in rows:
        for key in ("reference", "trace"):
            if identity(row[key]["path"]) != row[key]:
                raise ValueError("Preserved input changed")
        for fraction in (.25, .65):
            directions = rng.normal(size=(4, ACT_DIM))
            directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
            samples.append({"clip": row["clip"], "fraction": fraction,
                            "start": int(np.floor(fraction * (row["executed_steps"] - HORIZON))),
                            "reference": row["reference"], "trace": row["trace"],
                            "directions": directions.tolist()})
    plan = {"status": "PREPARED_NOT_EXECUTED", "source": identity(__file__),
            "root": str(args.root.resolve()), "checkpoint": evaluation["checkpoint"],
            "evaluation": identity(args.evaluation), "backend": backend, "sources": sources,
            "samples": samples, "horizon": HORIZON, "readouts": READOUTS,
            "max_control_steps": 4 * len(VARIANTS) * HORIZON,
            "max_physics_substeps": 4 * len(VARIANTS) * HORIZON * 20,
            "protocol": "Original R01 effective friction0.6; no policy calls or optimizer. Four saved-state baselines must replay before perturbing the first action only. All remaining actions fixed. Early failures kept; no replacement or retry.",
            "limits": "Local visited-state open-loop sensitivity, not full actor-gradient correctness or a policy-success gate."}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    print(json.dumps(plan, indent=2), flush=True)


def model_rollout(model, state, actions, delta):
    s = state.expand(len(delta), -1)
    xi = torch.zeros(len(delta), XI_DIM, device=s.device, dtype=s.dtype)
    xi[:, 0] = 1.
    result = []
    for i in range(HORIZON):
        action = actions[i].expand(len(delta), -1)
        if i == 0:
            action = action + delta
        s = model.step(s, action.clamp(-20., 20.), xi)
        if i + 1 in READOUTS:
            result.append(s)
    return torch.stack(result, dim=1)


def execute(args):
    plan_path = args.output / "plan.json"
    plan = json.loads(plan_path.read_text())
    if (args.output / "admission.json").exists():
        raise FileExistsError("This probe is already admitted")
    for item in [plan["source"], plan["checkpoint"], plan["evaluation"], plan["backend"], *plan["sources"]]:
        if identity(item["path"]) != item:
            raise ValueError("Frozen input changed")
    for sample in plan["samples"]:
        for key in ("reference", "trace"):
            if identity(sample[key]["path"]) != sample[key]:
                raise ValueError("Frozen sample changed")
    write_json(args.output / "admission.json", {"plan": identity(plan_path), "device": args.device})
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    receipt = {"status": "FAILED", "plan": identity(plan_path), "baselines": [],
               "samples": [], "control_steps": 0, "new_policy_inference": 0,
               "optimizer_updates": 0, "storage_status": "LOCAL_ONLY"}
    sims, prepared = [], []

    def native(sim, sample, initial, previous, actions):
        sim.reset(sim.ids, torch.tensor([sample["start"]]))
        sim.state.copy_(initial)
        sim.previous_action.copy_(previous)
        sim.steps.fill_(sample["start"])
        states, term = [], []
        for action in actions:
            if receipt["control_steps"] >= plan["max_control_steps"]:
                raise RuntimeError("Control budget exhausted")
            result = sim.step(action[None], auto_reset=False)
            receipt["control_steps"] += 1
            states.append(result["state"][0].numpy().copy())
            failed = bool(result["terminated"][0] or result["timeout"][0])
            term.append(failed)
            if failed:
                break
        return np.asarray(states), np.asarray(term)

    try:
        assets = Q1Assets(plan["root"])
        ck = torch.load(plan["checkpoint"]["path"], map_location=args.device, weights_only=False)
        if assets.manifest() != ck["manifest"]["assets"]:
            raise ValueError("R01 asset contract differs")
        cfg = dict(ck["manifest"]["world_model"])
        cfg["state_spec"] = StateSpecCfg(**cfg["state_spec"])
        model = WorldModel(WorldModelCfg(**cfg), assets.heads(args.device), args.device).eval()
        model.load_state_dict(ck["wm"], strict=True)
        model.requires_grad_(False)
        del ck
        double_model = copy.deepcopy(model).double()
        for index, sample in enumerate(plan["samples"]):
            lib = MotionLibrary([sample["reference"]["path"]], TRACKED, assets.q0,
                                device="cpu", active_indices=assets.active_indices)
            sim = Q1Sim(assets, lib, SimCfg(num_envs=1, threads=1,
                        domain_randomization=False, observation_noise=False, action_noise=0.,
                        episode_steps=lib.num_frames + 1), assets.heads(), 20261023)
            sims.append(sim)
            with np.load(sample["trace"]["path"], allow_pickle=False) as data:
                saved = torch.from_numpy(data["state"].copy())
                actions = torch.from_numpy(data["actions"].copy()).clamp(-20., 20.)
            t = sample["start"]
            initial, previous = saved[t - 1:t], actions[t - 1:t]
            window = actions[t:t + HORIZON]
            baseline, term = native(sim, sample, initial, previous, window)
            error = np.abs(baseline - saved[t:t + HORIZON].numpy()) if len(baseline) == HORIZON else None
            passed = error is not None and not term.any() and bool(
                np.all(error <= np.array([2e-6] * 29 + [2e-5] * 28)))
            row = {"sample": index, "clip": sample["clip"], "start": t, "passed": passed,
                   "max_abs_error": float(error.max()) if error is not None else None}
            receipt["baselines"].append(row)
            print(json.dumps({"baseline": row}), flush=True)
            if not passed:
                raise AssertionError("Saved trajectory baseline did not replay")
            prepared.append((sim, sample, initial, previous, window, baseline))
        for index, (sim, sample, initial, previous, actions, baseline) in enumerate(prepared):
            directions = np.asarray(sample["directions"])
            native_states = np.full((len(VARIANTS), HORIZON, STATE_DIM), np.nan, dtype=np.float32)
            native_states[0] = baseline
            lengths = np.zeros(len(VARIANTS), int)
            lengths[0] = HORIZON
            failures = np.zeros(len(VARIANTS), bool)
            termination_masks = np.zeros((len(VARIANTS), HORIZON), bool)
            delta = np.zeros((len(VARIANTS), ACT_DIM), np.float32)
            for vi, (direction, magnitude, sign) in enumerate(VARIANTS[1:], 1):
                delta[vi] = directions[direction] * magnitude * sign
                controls = actions.clone()
                controls[0] += torch.from_numpy(delta[vi])
                states, terminated = native(sim, sample, initial, previous, controls)
                native_states[vi, :len(states)] = states
                lengths[vi], failures[vi] = len(states), bool(terminated.any())
                termination_masks[vi, :len(states)] = terminated
            s = initial.to(args.device)
            a = actions.to(args.device)
            with torch.no_grad():
                predicted = model_rollout(model, s, a, torch.from_numpy(delta).to(args.device)).cpu().numpy()
            derivatives, numerical = [], []
            for direction in directions:
                vector = torch.tensor(direction[None], device=args.device, dtype=torch.float32)
                zero = torch.zeros_like(vector)
                fn = lambda value: model_rollout(model, s, a, value)
                _, jvp = torch.autograd.functional.jvp(fn, zero, vector)
                f64 = lambda value: model_rollout(double_model, s.double(), a.double(), value)
                z64, v64 = zero.double(), vector.double()
                _, j64 = torch.autograd.functional.jvp(f64, z64, v64)
                with torch.no_grad():
                    fd64 = (f64(z64 + 1e-4 * v64) - f64(z64 - 1e-4 * v64)) / 2e-4
                control = bool(torch.all((j64 - fd64).abs() <= 1e-7 + 1e-3 * j64.abs()))
                precision = bool(torch.all((jvp.double() - j64).abs() <= 1e-5 + .02 * j64.abs()))
                if not control or not precision:
                    raise AssertionError("Model automatic-differentiation numerical control failed")
                derivatives.append(jvp[0].cpu().numpy())
                numerical.append({"float64_jvp_fd_passed": control, "float32_float64_passed": precision,
                                  "float64_max_abs_error": float((j64 - fd64).abs().max())})
            derivatives = np.asarray(derivatives)
            comparisons = []
            for di in range(4):
                for magnitude in MAGNITUDES:
                    minus, plus = [i for i, item in enumerate(VARIANTS)
                                   if item[0] == di and item[1] == magnitude]
                    for hi, horizon in enumerate(READOUTS):
                        valid = (lengths[minus] >= horizon and lengths[plus] >= horizon
                                 and not termination_masks[[minus, plus], :horizon].any())
                        pfd = (predicted[plus, hi].astype(float) - predicted[minus, hi].astype(float)) / (2 * magnitude)
                        truth = ((native_states[plus, horizon - 1].astype(float)
                                  - native_states[minus, horizon - 1].astype(float)) / (2 * magnitude)) if valid else None
                        for group, ids, floor in (("root", np.arange(3), 1e-6),
                                                  ("joints", np.arange(7, 29), 1e-5)):
                            comparisons.append({"direction": di, "magnitude": magnitude, "horizon": horizon,
                                                "group": group, "valid": bool(valid),
                                                "model_fd": compare(truth[ids], pfd[ids], floor) if valid else None,
                                                "model_jvp": compare(truth[ids], derivatives[di, hi, ids], floor) if valid else None})
            raw_path = args.output / f"sample_{index:02d}.npz"
            np.savez_compressed(raw_path, native=native_states, model=predicted, jvp=derivatives,
                                lengths=lengths, failures=failures, termination_masks=termination_masks,
                                directions=directions, delta=delta)
            receipt["samples"].append({"sample": index, "clip": sample["clip"], "start": sample["start"],
                                       "raw": identity(raw_path), "numerical_controls": numerical,
                                       "responses": comparisons})
            print(json.dumps({"sample_finished": index, "control_steps": receipt["control_steps"]}), flush=True)
        summary = {}
        for h in READOUTS:
            for group in ("root", "joints"):
                rows = [r for sample in receipt["samples"] for r in sample["responses"]
                        if r["horizon"] == h and r["group"] == group]
                valid = [r["model_jvp"] for r in rows if r["valid"]]
                cos = [r["cosine"] for r in valid if r["cosine"] is not None]
                ratio = [r["norm_ratio"] for r in valid if r["norm_ratio"] is not None]
                summary[f"H{h}_{group}"] = {"total": len(rows), "valid": len(valid), "nondegenerate": len(cos),
                    "median_cosine": float(np.median(cos)) if cos else None,
                    "positive_cosine_fraction": float(np.mean(np.asarray(cos) > 0)) if cos else None,
                    "median_norm_ratio": float(np.median(ratio)) if ratio else None,
                    "median_relative_l2_error": float(np.median([r["relative_l2_error"] for r in valid])) if valid else None}
        receipt["summary"] = summary
        receipt["status"] = "Q1_ACTION_RESPONSE_DIAGNOSTIC_COMPLETED"
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for sim in sims:
            sim.close()
        receipt["seconds"] = time.monotonic() - started
        receipt["physics_substeps"] = receipt["control_steps"] * 20
        write_json(args.output / "receipt.json", receipt)
        print(json.dumps({key: value for key, value in receipt.items() if key not in ("samples", "baselines")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.execute:
        execute(args)
    else:
        if args.evaluation is None:
            parser.error("Preparation requires --evaluation")
        prepare(args)


if __name__ == "__main__":
    main()
