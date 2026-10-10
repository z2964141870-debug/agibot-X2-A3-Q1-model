"""Produce explicit acceptance gates for conservative A3 update diagnostics."""

import json

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.corrected_trial import read
from script.a3.fullchain_support import ROOT


OUTPUT = ROOT / "data/experiments/a3_search_20261010_E06"
METRICS = ("joint_rmse_rad", "root_pos_error_m_mean", "wrists_anchor_local_mean_m", "legs_anchor_local_mean_m")


def gate(base, candidate):
    full = {metric: candidate["all"]["full_rollout"][metric] <= base["all"]["full_rollout"][metric]
            for metric in METRICS}
    hold = {metric: candidate["groups"]["heldout"]["full_rollout"][metric] <=
                    base["groups"]["heldout"]["full_rollout"][metric] for metric in METRICS}
    heldout_joint_improved = candidate["groups"]["heldout"]["full_rollout"]["joint_rmse_rad"] < \
                            base["groups"]["heldout"]["full_rollout"]["joint_rmse_rad"]
    no_additional_falls = all(candidate["groups"][role]["falls"] <= base["groups"][role]["falls"]
                             for role in ("train", "heldout"))
    return dict(no_additional_falls=no_additional_falls, heldout_joint_improved=heldout_joint_improved,
                all_motion_nonregression=full, validation_nonregression=hold,
                selected20_gate_passed=no_additional_falls and heldout_joint_improved and all(full.values()) and all(hold.values()),
                general_acceptance=False, general_acceptance_reason="NEEDS_UNSEEN_SESSIONS_AND_MULTIPLE_SEEDS")


def main():
    sources = [ROOT / "data/experiments/a3_search_20261010_E05/comparison.json",
               OUTPUT / "comparison.json"]
    reports = [read(path) for path in sources]
    tables = {}
    for report in reports:
        for row in report["rows"]:
            if row["label"] in tables and row != tables[row["label"]]:
                raise ValueError("Reused comparison result changed")
            tables[row["label"]] = row
    base = tables["official"]
    gates = {label: gate(base, row) for label, row in tables.items() if label != "official"}
    eligible = [label for label, receipt in gates.items() if receipt["selected20_gate_passed"]]
    selected = min(eligible, key=lambda label: tables[label]["groups"]["heldout"]["full_rollout"]["joint_rmse_rad"]) if eligible else "official"
    result = dict(rows=list(tables.values()), gates=gates, selected_reference=selected,
        new_policy_improvement_found=bool(eligible), criteria="STRICT_FALL_JOINT_ROOT_WRIST_LEG_NONREGRESSION_PLUS_VALIDATION_JOINT_IMPROVEMENT",
        input_reports=[dict(path=str(path.relative_to(ROOT)), sha256=sha256(path)) for path in sources],
        next_stage="REPEAT_CANDIDATE_WITH_OTHER_SEEDS_AND_UNSEEN_MOTIONS" if eligible else
                   "KEEP_OFFICIAL_AND_DIAGNOSE_UPDATE_DIRECTION_BEFORE_LONGER_TRAINING",
        scope="REUSABLE_DIAGNOSTIC_SELECTION_NOT_BLIND_FINAL_TEST_OR_DEPLOYMENT", backup_status="LOCAL_ONLY")
    atomic_json(OUTPUT / "selection.json", result)
    atomic_json(ROOT / "data/manifests/a3_search_selection_20261010.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
