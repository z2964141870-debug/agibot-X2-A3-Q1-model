#!/usr/bin/env python3
"""Audit saved local reward responses, separating exact action-rate rewards."""

import argparse
import json
from pathlib import Path

import numpy as np

from compare_prefix import checked_file, identity
from diagnose_fullbody import write_json


def rate_rewards(actions, previous):
    history = np.concatenate([previous[None], actions[:-1]], axis=0)
    return -.002 * np.sum((actions.astype(float) - history.astype(float)) ** 2, axis=-1)


def audit(plan_path, receipt_path):
    plan = json.loads(plan_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if receipt["plan"] != identity(plan_path) or receipt["status"] != "LOCAL_OPEN_LOOP_REWARD_DIRECTION_PROBE_COMPLETED":
        raise ValueError("Probe receipt and frozen plan differ")
    checked_file(plan["checkpoint"])
    discount = plan["discount"] ** np.arange(32)
    variants = [(None, 0., 0)] + [(d, m, sign) for d in range(4)
                for m in plan["magnitudes"] for sign in (-1, 1)]
    if len(receipt["rows"]) != len(plan["samples"]) or receipt["control_steps"] > plan["max_probe_control_steps"]:
        raise ValueError("Registered sample/control budget changed")
    rows = []
    baseline_errors = []
    for sample, result in zip(plan["samples"], receipt["rows"]):
        if (sample["clip"], sample["start"]) != (result["clip"], result["start"]):
            raise ValueError("Frozen state selection changed")
        with np.load(checked_file(sample["trace"]), allow_pickle=False) as saved:
            saved_actions = saved["actions"].copy().clip(-20, 20)
            saved_states = saved["state"].copy()
        start = sample["start"]
        actions = saved_actions[start:start + 32]
        previous = saved_actions[start - 1]
        with np.load(checked_file(result["raw"]), allow_pickle=False) as raw:
            actual = raw["native_reward"].copy()
            predicted = raw["model_reward"].copy()
            states = raw["native_state"].copy()
            roots = raw["model_root_position"].copy()
            gradients = raw["gradients"].copy()
            delta = raw["delta"].copy()
            directions = raw["directions"].copy()
            lengths, failed = raw["lengths"].copy(), raw["failed"].copy()
        if (actual.shape != (17, 32) or predicted.shape != (17, 32)
                or not np.isfinite(actual).all() or not np.isfinite(predicted).all()
                or np.any(lengths != 32) or failed.any()):
            raise ValueError("This audit expects the observed complete finite perturbations")
        np.testing.assert_array_equal(states[0], saved_states[start:start + 32])
        rate = []
        for change in delta:
            controls = actions.copy()
            controls[0] += change
            controls = controls.clip(-20, 20)
            rate.append(rate_rewards(controls, previous))
        rate = np.asarray(rate)
        actual_returns = (actual.astype(float) * discount).cumsum(-1)
        predicted_returns = (predicted.astype(float) * discount).cumsum(-1)
        rate_returns = (rate * discount).cumsum(-1)
        for h in plan["horizons"]:
            native_return = float(actual_returns[0, h - 1])
            model_return = float(predicted_returns[0, h - 1])
            baseline_errors.append({"sample": result["sample"], "horizon": h,
                "reward_return_error": model_return - native_return,
                "root_position_error_m": float(np.linalg.norm(roots[0, h - 1] - states[0, h - 1, :3]))})
        for response in result["responses"]:
            if response["direction"] == "random":
                di = 3
            else:
                di = plan["horizons"].index(int(response["direction"].split("H")[1]))
            magnitude, h = response["magnitude"], response["horizon"]
            minus, plus = [i for i, v in enumerate(variants) if v[0] == di and v[1] == magnitude]
            real_gain = float(actual_returns[plus, h - 1] - actual_returns[0, h - 1])
            model_gain = float(predicted_returns[plus, h - 1] - predicted_returns[0, h - 1])
            real_fd = float((actual_returns[plus, h - 1] - actual_returns[minus, h - 1]) / (2 * magnitude))
            model_fd = float((predicted_returns[plus, h - 1] - predicted_returns[minus, h - 1]) / (2 * magnitude))
            for observed, expected in ((real_gain, response["positive_step_physics_return_gain"]),
                                       (model_gain, response["positive_step_model_return_gain"]),
                                       (real_fd, response["physics_central_fd"]),
                                       (model_fd, response["model_central_fd"])):
                if not np.isclose(observed, expected, atol=1e-10, rtol=1e-8):
                    raise AssertionError("Raw reward response differs from receipt")
            rate_grad = -.004 * (actions[0].astype(float) - previous.astype(float))
            if h > 1:
                rate_grad += .004 * plan["discount"] * (actions[1].astype(float) - actions[0].astype(float))
            rate_ad = float(np.dot(rate_grad, directions[di]))
            rate_fd = float((rate_returns[plus, h - 1] - rate_returns[minus, h - 1]) / (2 * magnitude))
            if not np.isclose(rate_fd, rate_ad, atol=2e-7, rtol=1e-4):
                raise AssertionError("Exact action-rate derivative numerical control failed")
            ad = float(np.dot(gradients[plan["horizons"].index(h)], directions[di]))
            state_real_fd, state_model_ad = real_fd - rate_fd, ad - rate_ad
            state_gain = real_gain - float(rate_returns[plus, h - 1] - rate_returns[0, h - 1])
            rows.append({"sample": result["sample"], "clip": sample["clip"], "start": start,
                "direction": response["direction"], "horizon": h, "magnitude": magnitude,
                "total_model_ad": ad, "total_model_fd": model_fd, "total_physics_fd": real_fd,
                "rate_ad": rate_ad, "rate_fraction_of_total_model_ad": rate_ad / ad if abs(ad) > 1e-10 else None,
                "state_dependent_model_ad": state_model_ad, "state_dependent_physics_fd": state_real_fd,
                "state_dependent_sign_matches": bool(state_model_ad * state_real_fd > 0)
                    if abs(state_model_ad) > 1e-7 and abs(state_real_fd) > 1e-7 else None,
                "positive_step_total_physics_return_gain": real_gain,
                "positive_step_state_dependent_physics_return_gain": state_gain,
                "positive_step_root_xy_difference_m": float(np.linalg.norm(states[plus, h - 1, :2] - states[0, h - 1, :2]))})
    summary = {}
    for h in plan["horizons"]:
        selected = [r for r in rows if r["horizon"] == h and r["direction"] == "gradient_H" + str(h)]
        informative = [r["state_dependent_sign_matches"] for r in selected if r["state_dependent_sign_matches"] is not None]
        baseline = [r for r in baseline_errors if r["horizon"] == h]
        summary["H" + str(h)] = {
            "states": len(plan["samples"]), "state_magnitude_pairs": len(selected),
            "median_rate_fraction_of_total_model_ad": float(np.median([r["rate_fraction_of_total_model_ad"] for r in selected])),
            "state_dependent_informative_pairs": len(informative), "state_dependent_correct_signs": sum(informative),
            "positive_steps_improving_state_dependent_reward": sum(r["positive_step_state_dependent_physics_return_gain"] > 1e-7 for r in selected),
            "median_state_dependent_reward_gain": float(np.median([r["positive_step_state_dependent_physics_return_gain"] for r in selected])),
            "median_positive_step_root_xy_difference_m": float(np.median([r["positive_step_root_xy_difference_m"] for r in selected])),
            "mean_baseline_root_prediction_error_m": float(np.mean([r["root_position_error_m"] for r in baseline])),
            "mean_absolute_baseline_return_prediction_error": float(np.mean([abs(r["reward_return_error"]) for r in baseline]))}
    return {"status": "RAW_REWARD_RESPONSES_AND_RATE_DECOMPOSITION_AUDITED", "source": identity(__file__),
            "plan": identity(plan_path), "probe": identity(receipt_path), "summary": summary,
            "rows": rows, "baseline_errors": baseline_errors,
            "new_physics_steps": 0, "new_model_forwards": 0, "new_optimizer_updates": 0,
            "limits": "State-dependent reward means 6 tracking terms plus joint-limit penalty after exact action-rate subtraction. Gradients were chosen for total raw reward, not reoptimized for this residual. Two magnitudes share each of 6 states; no independent-sample claim. No survival/critic/closed-loop actor validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.plan, args.probe)
    write_json(args.output, result)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
