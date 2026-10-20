"""E54 protocol: paths, arm constants, pinned appendix-B baselines and verdict math."""

from pathlib import Path

OPS = Path("/opt/data/private/yuanqi_ops")
NS = "a3_zhy_channel_probe_20261019_E54"
EXPERIMENT = "A3_ZHY_CHANNEL_PROBE_20261019_E54"
TRAIN_ROOT = OPS / "data/training" / NS

SOURCE_035 = "data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt"
ARMS = ("P", "Q")
TIERS = (16,)
UPDATES_BY_TIER = {16: 16}
CHECKPOINT_BY_TIER = {16: "checkpoints/milestone_16/model_step_000016.pt"}
CHANNELS = 6
BASE_OBS_DIM = 640
ARMS_TRAINER = {"P": "e54_callback.E54InjectedTrainer", "Q": "e54_callback.E54InjectedTrainer"}
FUNC_BY_ARM = {
    "P": "e54_obs_terms:e54_anchor_ori_b_mf_standardized",
    "Q": "e54_obs_terms:e54_anchor_ori_b_mf_errorform",
}
ARMS_NOTES = {
    "P": "Per-motion standardized channels: clip((c_I - mu_m)/sigma_m, -5, +5); mu/sigma are offline per-motion reference statistics (no learned parameters), manifest e54_channel_stats.json SHA-registered.",
    "Q": "Error-form channels: [disp_a(ref)-disp_a(sim), vel_a(ref)-vel_a(sim)] in the reference anchor heading frame; zero at perfect tracking, bounded by drift.",
}
RUNTIME_LIMIT_TRAIN_SECONDS = 7200
RUNTIME_LIMIT_EVAL_SECONDS = 5400
MEMORY_RESERVE_GIB = 2
ENTRY_HEADROOM_GIB = 8
MAX_CPU_C = 90
MAX_GPU_C = 85

E53_NS = "a3_zhy_rootchannel_20261019_E53"
E53_OUT = OPS / "data/experiments" / E53_NS
ANCHOR_RECEIPT = E53_OUT / "anchors_035.json"
ANCHOR_TARGETS = E53_OUT / "anchors_035.pt"

WDR_ORDER = ("001_walk_front_slow", "029_rotate_place_slow",
             "048_balance_right_singleleg", "050_march_place_slow")
WDR_BASELINES = {
    "N8": {"001_walk_front_slow": 0.999, "029_rotate_place_slow": 1.138,
           "048_balance_right_singleleg": 1.034, "050_march_place_slow": 0.993, "max": 1.138},
    "I8": {"001_walk_front_slow": 1.027, "029_rotate_place_slow": 1.116,
           "048_balance_right_singleleg": 1.019, "050_march_place_slow": 1.294, "max": 1.294},
    "I32": {"001_walk_front_slow": 1.015, "029_rotate_place_slow": 1.138,
            "048_balance_right_singleleg": 1.065, "050_march_place_slow": 1.531, "max": 1.531},
}
E53_I8_CLOTH6 = 0.174711
PRIMARY_DROP_FRAC = 0.10
EFF_MAX_BOUND = round(min(1.165, 1.138 * 1.02), 4)   # 1.1608, lead-ratified conjunction bound
CLOTH6_BOUND = round(E53_I8_CLOTH6 * 1.05, 6)        # 0.183447

STATS_NAME = "e54_channel_stats.json"


def wdr(torso, wrists, legs):
    return max(wrists, legs) / max(torso, 1e-12)


def verdict(candidates):
    """candidates: {arm: {"wdr": {motion: v, "max": m}, "cloth6": c}} -> per-arm passes + selection."""
    out = {}
    for arm, row in candidates.items():
        primary = row["wdr"]["max"] <= EFF_MAX_BOUND
        secondary = row["cloth6"] <= CLOTH6_BOUND
        out[arm] = dict(wdr_max=round(row["wdr"]["max"], 4), cloth6=round(row["cloth6"], 6),
                        primary_pass=bool(primary), secondary_pass=bool(secondary),
                        both_pass=bool(primary and secondary))
    passing = [a for a in ("P", "Q") if out[a]["both_pass"]]
    if len(passing) == 2:
        if out["P"]["wdr_max"] <= out["Q"]["wdr_max"] or abs(out["P"]["wdr_max"] - out["Q"]["wdr_max"]) <= 0.02:
            winner, rule = "P", "min wdr max; tie (<=2%) -> P (simpler, reference-statistics only)"
        else:
            winner, rule = "Q", "min wdr max"
        sel = dict(passing=passing, winner=winner, rule=rule)
    elif passing:
        sel = dict(passing=passing, winner=None, rule="single pass: record data, no forced selection")
    else:
        sel = dict(passing=[], winner=None, rule="none pass: E53 open item stays open, escalate lead")
    return out, sel
