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

## 05:25真实来源选择完成

源码发布`afb29d9bdf272de4bcbe0accdb70c2a1ddea62f5`，Q1独立远端与本地HEAD一致。服务器三份新文件部署前确认不存在且不在R02D冻结列表，部署后逐SHA与本地一致；真实四项控制全部通过、0跳过。[控制原样收据](../data/manifests/supermimic_q1_P10_walking_controls_20261025.json)及[控制日志](../logs/session_records/supermimic_q1_P10_walking_controls_20261025.log)已取回。

真实选择原生退出0。[原样选择](../data/manifests/supermimic_q1_P10_walking_selection_20261025.json)30228字节/SHA`1b23939b731dcdfedc8b27732e4f22f92a4faf07412dff0dbea8529e6d969ba9`；未用52块、17个原事件候选、15个目录候选，按固定seed每目录一块得到15候选。额外排除treadmill后与初步只读调查的54/19/17不同，不代表转换准入或语义认证。全部15目录已预留补充test，三份旧PHUMA选择和16个AMASS诊断来源参与排除；其余命名空间仍有未解身份，全源独立保持false。

下一独立新目录执行15片原速Q1转换及十九门，不将原source split改写为官方test，也不修改R02D输入；当前0新Q1准入/物理/optimizer。训练部署时实读R02D950新增轮，盘余948.50GB，仍按原保护运行。
