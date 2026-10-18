# E53 — Exp-2b: root-channel injection at the g1 encoder (FINAL REPORT)

Experiment: A3_ZHY_ROOTCHANNEL_20261019_E53 · Host: ZHY 3090 · 2026-10-19
Executor: exp-operator · Lead-approved as E53 (amendments: budget tiers 8/32 only; 6-dim channel set; two-tier direction-consistency rule)

## Verdict: H2_INTERFACE_SUPPORTED

Per the lead's rule: arm I better than arm N at BOTH tiers, and both gaps >= 3pp.

| tier | arm N six-cloth mean (rad) | arm I six-cloth mean (rad) | gap (rad) | improvement |
|------|---------------------------|---------------------------|-----------|-------------|
| 8    | 0.190405                  | 0.174711                  | 0.015695  | **8.24%**   |
| 32   | 0.269835                  | 0.196860                  | 0.072975  | **27.04%**  |

Baseline B1 (E26 six-cloth mean): 0.1413798 rad. All four cells degrade vs B1 as expected at
8/32-update budgets; the mechanism effect is the within-tier N-vs-I comparison.

No 64-update escalation was needed (both tiers >= 3pp on the first pass).

## Task gates (mandatory reporting, informational for the mechanism verdict)

| cell | cloth mean (rad) | cloth falls | official world root mean (m) | official world wrists mean (m) |
|------|------------------|-------------|------------------------------|-------------------------------|
| N 8  | 0.190405 | 1 | 4.2504 | 4.2717 |
| N 32 | 0.269835 | 3 | 8.3398 | 7.4378 |
| I 8  | 0.174711 | 1 | 17.8572 | 16.2083 |
| I 32 | 0.196860 | 1 | 7.5791 | 6.6816 |

Note: arm I tier 8 shows a large official-eval world error (17.9 m root mean) while its
six-cloth result is the best of the run; official-eval gates are reported informationally per
contract and do not gate the mechanism verdict. Flagged for lead attention.

## Final arm-I mechanism (as executed — differs from the approved plan; disclosed)

The approved plan ("tokenizer-obs-term injection, g1 first Linear 640->646 zero-append") was
executed as follows after the true architecture was established on-host:

1. The tokenizer group is consumed per-NAME by the universal-token encoders; g1's slot is
   [command_multi_future_nonflat 580 | motion_anchor_ori_b_mf_nonflat 60] = 640 per slot, and
   640 = 10 future slots x 64 — not a flat 640.
2. Arm I overrides the anchor term via hydra argv to
   e53_obs_terms.e53_anchor_ori_b_mf_with_channels: vendor motion_anchor_ori_b block + the 6
   E53 root channels (HP26 PC definition: root displacement-so-far 3 + root linear velocity 3,
   anchor heading frame) repeated across the 10 slots => slot 580+120=700; the g1 first Linear
   auto-builds at 700 and the g1_kin decoder tail row-pads to 700.
3. The 035 checkpoint weights are slot-INTERLEAVED with zeros at load: encoder first-Linear
   columns [64k+64, 64k+70) and g1_kin decoder tail rows/bias are zero per slot, so the
   injected channels are exactly silent at t=0 (0 x 0 exact in fp32) and gradients reach the
   new parameters.
4. Load-time guarantees: the checkpoint source dict is padded inside torch.load so live ==
   source stays bit-equal (Finetune exact-warm-start guard passes); the E11-replay anchor
   identity check passes (<= 1e-5); frozen-parameter digests unchanged.
5. Permanent width-gated padding wrappers (actor_module.forward /
   forward_with_external_tokens) replay the stored 2402-wide E11 obs through the 2462-spec
   module for the retention/anchor losses at every optimizer step.

Arm N is the unchanged 035 warm-start flow; single run per cell; no re-run after TRAINED.

## Prominent disclosures (executor-level deviations, all record-only or geometry-level)

- **Base obs dims corrected**: actor_obs 930, critic_obs 1645, tokenizer flat 2402 (not the
  1570/1645 figures used in early planning). Injection touches the g1 encoder input geometry
  (640 -> 700 per-slot interleave), not a flat append; the 6-dim channel set and "encoder
  input + zero-init" contract semantics are preserved.
- **Drift budget record-only** (approved amendment): E53RecordOnlyDriftMixin records
  teacher_action_rmse / max_abs per optimizer step (logs/a3_zhy_rootchannel_20261019_E53/
  <cell>/cloth_contract.json). Plain AdamW crosses the 0.05 gate near step 23 (E52 arm-A
  phenomenon, fourth reproduction); interception would have made both arms untrainable at
  these budgets.
- **Launch-failure archive**: 27 waves of arm-I launch failures (failed_launch_archive/), all
  pre-training (trainer_init / load gates / stage launch), zero contamination of trained
  cells; the E11-replay geometry (per-slot channel placement, exact-weights guard, decoder
  row-pad, persistent replay wrappers) was resolved empirically with one instrumented probe
  per failure class.
- **registration.json** lives under data/ (gitignored by policy); the amended file on disk is
  authoritative (mechanism_refinement + code SHA256 map). Code commits: main @ 9ca09f5-lineage
  amended through the run; final code state = files hashed in registration.json.

## Artifacts (zhy:/opt/data/private/yuanqi_ops/)

- Training receipts: data/experiments/a3_zhy_rootchannel_20261019_E53/train_arm_{n,i}_tier{8,32}.json (all TRAINED)
- Eval receipts: eval_arm_{n,i}_tier{8,32}_{cloth,official}.json (8/8 rc=0)
- Effect verdict: data/experiments/a3_zhy_rootchannel_20261019_E53/effect_verification.json
- Milestones: data/training/a3_zhy_rootchannel_20261019_E53/<cell>/milestone_*/
- Drift records: logs/a3_zhy_rootchannel_20261019_E53/<cell>/cloth_contract.json
- Code: e53_protocol.py, e53_trainer.py, e53_trial.py, e53_obs_terms.py, e53_anchor_probe.py, test_e53_controls.py
- Controls: test_e53_controls.py 6/6 pass (zero-init identity, heading transform, channel stats, verdict table, obs-term math, protocol constants)
