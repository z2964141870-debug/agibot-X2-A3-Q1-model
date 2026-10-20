"""E55 protocol: paths, arm constants, grouped per_joint bands, gate rank + liveness verdict math."""

from pathlib import Path

OPS = Path("/opt/data/private/yuanqi_ops")
NS = "a3_zhy_residual_p1_20261019_E55"
EXPERIMENT = "A3_ZHY_RESIDUAL_P1_20261019_E55"
TRAIN_ROOT = OPS / "data/training" / NS

SOURCE_035 = "data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt"
ARMS = ("A0", "P", "Q")
MILESTONES = (8, 32, 64)
UPDATES = 64
CHECKPOINT_BY_MILESTONE = {m: "checkpoints/milestone_%d/model_step_%06d.pt" % (m, m) for m in MILESTONES}
CHANNELS = 6
SLOT_ANCHOR = 58
SLOT_WIDTH = 64
SLOTS = 10
ARMS_TRAINER = {a: "e55_trainer.E55ResidualTrainer" for a in ARMS}
FUNC_BY_ARM = {
    "A0": "e55_obs_terms:e55_root_channels_a0",
    "P": "e55_obs_terms:e55_root_channels_p",
    "Q": "e55_obs_terms:e55_root_channels_q",
}
ARMS_NOTES = {
    "A0": "Raw c_I channels (E53 arm-I math): sim-anchor displacement accumulation 3 + anchor-frame velocity 3; no normalization.",
    "P": "Per-motion standardized: clip((c_I - mu_m)/sigma_m, -5, +5); reuses E54 e54_channel_stats.json (sha pinned to E54 registration stats_sha256).",
    "Q": "Error-form: [disp_a(ref)-disp_a(sim), vel_a(ref)-vel_a(sim)] in the anchor heading frame; reset returns 6-wide zeros (E54 fix).",
}
HEAD = dict(hidden=(256, 256), activation="elu", slot_aggregation="sum",
            zero_init="last_layer_only", input="slot58_plus_6")
RUNTIME_LIMIT_TRAIN_SECONDS = 7200
RUNTIME_LIMIT_EVAL_SECONDS = 5400
MEMORY_RESERVE_GIB = 2
ENTRY_HEADROOM_GIB = 8
MAX_CPU_C = 90
MAX_GPU_C = 85

E53_NS = "a3_zhy_rootchannel_20261019_E53"
E54_NS = "a3_zhy_channel_probe_20261019_E54"
E54_OUT = OPS / "data/experiments" / E54_NS
ANCHOR_RECEIPT = E54_OUT.parent / E53_NS / "anchors_035.json"
ANCHOR_TARGETS = E54_OUT.parent / E53_NS / "anchors_035.pt"
STATS_NAME = "e54_channel_stats.json"
STATS = E54_OUT / STATS_NAME
E54_REG = E54_OUT / "registration.json"

WDR_ORDER = ("001_walk_front_slow", "029_rotate_place_slow",
             "048_balance_right_singleleg", "050_march_place_slow")
# N8 WDR reference table kept for README context only; gate_pass never reads it.
# v1.1 shutter bands (lead + code-mapper v1.2): cloth WDR / official WDR / grouped per_joint bff
BAND_CLOTH_WDR = 1.185
BAND_OFFICIAL_WDR = 1.640
PERJOINT_CLOTH_BASE = 0.234
PERJOINT_OFFICIAL_BASE = 0.283
PERJOINT_FACTOR = 1.2
# contract v1.2 literals (rounded from 0.234x1.2=0.2808 and 0.283x1.2=0.3396)
BAND_PERJOINT_CLOTH = 0.281
BAND_PERJOINT_OFFICIAL = 0.340
GUARD_TOLERANCE_FRACTION = 0.5   # liveness: degradation > half of guard tolerance => active

STATS_GUARD = {"cloth": BAND_PERJOINT_CLOTH, "official": BAND_PERJOINT_OFFICIAL}


def wdr(torso, wrists, legs):
    return max(wrists, legs) / max(torso, 1e-12)


def gate_pass(shutter):
    """shutter: {kind: {"wdr_max": float, "per_joint_bff": float}} for cloth+official readings."""
    checks = {
        "cloth_wdr": shutter["cloth"]["wdr_max"] <= BAND_CLOTH_WDR,
        "official_wdr": shutter["official"]["wdr_max"] <= BAND_OFFICIAL_WDR,
        "per_joint_cloth": shutter["cloth"]["per_joint_bff"] <= BAND_PERJOINT_CLOTH,
        "per_joint_official": shutter["official"]["per_joint_bff"] <= BAND_PERJOINT_OFFICIAL,
    }
    return all(checks.values()), checks


def rank_key(row):
    """(Z, H, -N) execution rank from REAL landing statistics (VerificationSpec v1.1).

    Z = zero-fall repeat count, H = min first-fall step, N = total fall count.
    WDR/per_joint are guard bands (binary) and the cloth6 tie-break layer only -
    they never enter the execution key (contract B.2).
    """
    m = row["rank_metrics"]
    return (m["Z"], m["H"], -m["N"])


def verdict(candidates, liveness):
    """candidates: {arm: {gate fields, cloth6_by_milestone}}; liveness: {arm: "ACTIVE"|"CHANNEL_INERT"|"PENDING"}.

    Health: cloth6(64) < cloth6(8). Selection: rank key (Z,H,-N), cloth6 tie-break;
    tie = (Z,H,-N) all equal AND cloth6 relative diff <=2% -> A0. Winner downgrade on CHANNEL_INERT.
    """
    out = {}
    for arm, row in candidates.items():
        passed, checks = gate_pass(row["shutter"])
        health = row["cloth6"][64] < row["cloth6"][8]
        out[arm] = dict(gate_pass=bool(passed), checks=checks, health=health,
                        liveness=liveness.get(arm, "PENDING"),
                        cloth6_by_milestone={str(m): row["cloth6"][m] for m in MILESTONES},
                        rank=list(rank_key(row)))
    passing = [a for a in ARMS if out[a]["gate_pass"]]
    if len(passing) >= 2:
        ordered = sorted(passing, key=lambda a: rank_key(candidates[a]), reverse=True)
        best, second = ordered[0], ordered[1]
        k1, k2 = rank_key(candidates[best]), rank_key(candidates[second])
        key_tie = tuple(round(float(v), 6) for v in k1) == tuple(round(float(v), 6) for v in k2)
        rel = abs(candidates[best]["cloth6"][64] - candidates[second]["cloth6"][64]) / max(
            abs(candidates[best]["cloth6"][64]), 1e-12)
        if key_tie and rel <= 0.02:
            winner, rule = "A0", "key (Z,H,-N) all equal & cloth6 rel diff<=2% -> A0"
        elif key_tie:
            winner = best if candidates[best]["cloth6"][64] <= candidates[second]["cloth6"][64] else second
            rule = "key all equal -> cloth6 tie-break (smaller cloth6 wins)"
        else:
            winner, rule = best, "key (Z,H,-N) lexicographic max wins"
    elif passing:
        winner, rule = passing[0], "single gate pass: record, no forced selection"
    else:
        winner, rule = None, "none pass gate: record + escalate lead"
    conclusion = None
    if winner is not None:
        lv = liveness.get(winner, "PENDING")
        if lv == "CHANNEL_INERT":
            rule = rule + "; WINNER CHANNEL_INERT -> form conclusion INCONCLUSIVE"
            conclusion = "INCONCLUSIVE"
        elif lv == "PENDING":
            rule = rule + "; WINNER LIVENESS PENDING -> no conclusion until three shutters run"
            conclusion = "PENDING"
        elif lv == "ACTIVE":
            conclusion = "CLEAN"
        else:
            conclusion = "INCONCLUSIVE"
    return out, dict(passing=passing, winner=winner, rule=rule, winner_conclusion=conclusion)
