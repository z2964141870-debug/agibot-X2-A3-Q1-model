"""E53 Exp-2b root-channel injection constants and verdict logic; no tensor imports."""

TIERS = (8, 32)
ARMS = ("N", "I")
CHANNELS = 6
G1_ENCODER_FIRST_LAYER_IN = 640
G1_TEMPORAL_REPEAT = 10
G1_FIRST_LAYER_IN_BUILT = 700
CHANNEL_GROUP = "e53_root_channels"
G1_ENCODER_NAME = "g1"
UPDATES_BY_TIER = {8: 8, 32: 32}
CHECKPOINT_BY_TIER = {8: "milestone_8/model_step_000008.pt", 32: "milestone_32/model_step_000032.pt"}
BASELINE_CLOTH_MEAN = 0.1413798
MECHANISM_THRESHOLD_PCT = 3.0

RUNTIME_LIMIT_TRAIN_SECONDS = 1800
RUNTIME_LIMIT_EVAL_SECONDS = 1800
MEMORY_RESERVE_GIB = 4
ENTRY_HEADROOM_GIB = 8
MAX_CPU_C = 90
MAX_GPU_C = 85

EXPERIMENT = "a3_zhy_rootchannel_20261019_E53"
SOURCE_035 = "data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt"

ARMS_TRAINER = {
    "N": "e53_trainer.E53PlainTrainer",
    "I": "e53_trainer.E53InjectedTrainer",
}
ARMS_NOTES = {
    "N": "035 warm start, stock g1 encoder inputs (command_multi_future_nonflat + motion_anchor_ori_b_mf_nonflat).",
    "I": "Same 035 warm start; the tokenizer term motion_anchor_ori_b_mf_nonflat is overridden to e53_anchor_ori_b_mf_with_channels: vendor anchor block + 6 root channels repeated over the 10 future slots, so g1 slot width is 580+120=700 and the first Linear auto-builds at 700. The 035 checkpoint weight (2048,640) is right-padded with 60 zero columns inside load (source-side, so the Finetune exact-weights guard holds); zero columns multiply the injected channels, t=0 outputs are identical to 035 by construction and gated by the E11 anchor check.",
}
CRITERIA = [
    "mechanism: within each budget tier, arm I six-cloth mean RMSE better than arm N; verdict requires BOTH tiers direction-consistent; both >= 3pp of N -> H2_INTERFACE_SUPPORTED; both consistent but < 3pp -> WEAK_POSITIVE_SIGNAL (one 64-update escalation allowed); single-tier-only or contradictory -> INCONCLUSIVE (no conclusion, then H5 if escalation also fails)",
    "task gates reported per arm-tier vs B1 (E26 six-cloth mean 0.1413798): change, fall rate (cloth >= 5/6 preferred), official 4-motion per-joint <= +2% and world root/wrist <= +5% (informational for mechanism, mandatory reporting)",
    "teacher MSE recorded only; no interception, no early stop (E52 conclusion: the 0.05 gate does not protect task performance)",
    "channel sanity at tier 8 arm I: t=0 outputs equal 035 within 1e-5 (zero-column identity at encoders.g1 first Linear 640->646) and injected channel variance > 0",
    "single run per (arm, tier); no re-run after terminal status; launch-failure residue archived and re-launched once",
]


def mechanism_verdict_with_means(means, threshold_pct=MECHANISM_THRESHOLD_PCT):
    """means: {(arm, tier): six_cloth_mean_rad}. Returns (verdict, detail)."""
    gaps, pcts = {}, {}
    for tier in TIERS:
        n, i = means[("N", tier)], means[("I", tier)]
        gaps[tier] = n - i
        pcts[tier] = 100.0 * (n - i) / n if n else float("nan")
    consistent = all(g > 0 for g in gaps.values())
    detail = dict(gaps_rad={str(t): gaps[t] for t in TIERS},
                  improvement_pct={str(t): pcts[t] for t in TIERS})
    if not consistent:
        return "INCONCLUSIVE", detail
    if all(p >= threshold_pct for p in pcts.values()):
        return "H2_INTERFACE_SUPPORTED", detail
    return "WEAK_POSITIVE_SIGNAL", detail
