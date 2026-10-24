# PHUMA P10 supplemental walking evaluation

Date: 2026-10-25, Beijing time. Status: implementation and four local source-exclusion controls passed; server selection, conversion and actual dynamics pending.

## Goal and reason

Use existing PHUMA resources while BONES acquisition remains with the user's colleague. The existing three `test_locomotion` references have fitness names, and the original label means horizontal displacement above a threshold, not certified walking semantics. Preserve R02D's registered 340/6/7 inputs and seven-stage evaluation. Prepare a separate walking test to assess the final movement goal more directly.

## Selection and provenance

`script/supermimic_q1_sources/select_walking_holdout.py` joins the fixed HumanML metadata to the full 73076-row PHUMA scan. It excludes every previous P02/P04/additional selection by source group, exact bytes, known original event and directory-subject candidate; the sixteen AMASS diagnostic candidates also contribute event/directory exclusions. Mirror aliases and collection rename candidates are conservatively grouped for exclusion.

Walking names exclude treadmill and listed interactions. A fixed seed selects one chunk per unused directory candidate, before Q1 geometry or policy scores. Original scan rows and source splits remain byte-for-byte JSON-equivalent provenance. A separate role reserves all new references for `SUPPLEMENTARY_TEST_ONLY_NEVER_TRAIN`; a source originally labelled train is not relabelled as an official test item.

Unknown identities in other PHUMA namespaces and cross-collection identities remain unresolved. This is a supplementary test with documented conservative exclusions, not proof of complete source or biological-subject independence. Names alone do not certify walking semantics.

## Preparation and budgets

`prepare_walking_holdout.py` reuses the original full-index/ZIP reconciliation, native 30-to-50Hz retargeting and all nineteen independent geometric checks. It does not alter their thresholds or R02D's frozen files. The original geometry receipt field `qualified_for_training_pilot` is not permission to train on these reserved references.

One CPU worker, nice10; shared memory 28.75GiB, free disk 100GiB, output 256MiB; no CUDA or optimizer. The owned worker group is cleaned up on completion/failure. Initial and final source identities and geometry receipts remain on huanyu; only small receipts enter Git. All large assets remain LOCAL_ONLY.

## Reproduction and outputs

Root: `huanyu:/opt/data/private/YuanQi_Q1`. Python: `/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`. Publish and independently verify Q1, deploy new files by SHA, then:

```bash
"$PY" script/supermimic_q1_sources/select_walking_holdout.py \
  --root "$ROOT" \
  --output "$ROOT/data/supermimic_for_q1/PHUMA_P10_walking_selection_20261025.json"
"$PY" script/supermimic_q1_sources/prepare_walking_holdout.py \
  --root "$ROOT" \
  --selection "$ROOT/data/supermimic_for_q1/PHUMA_P10_walking_selection_20261025.json" \
  --output "$ROOT/data/supermimic_for_q1/PHUMA_P10_walking_holdout_20261025" \
  --expected-oom-kill 14
```

The OOM baseline must be rechecked before execution. Main outputs: selection; `batch/{scan_check,reference_check,completion}.json`; `batch/references/manifest.json` and NPZ; `guard.json`; same-name `logs/` stdout. No new reference or successful walking result exists at this implementation stage.

## Next step

Run focused source-exclusion controls locally and on the server, then select/convert/check the fixed pool. Review admitted references and their semantic/physical boundaries. A separate evaluation adapter must bind both the original checkpoint contract and the new reference receipt; the existing evaluator requires its original training library and must not be tricked with a replaced manifest. R02D remains the sole training run and retains its fixed 7000-new-round budget.
