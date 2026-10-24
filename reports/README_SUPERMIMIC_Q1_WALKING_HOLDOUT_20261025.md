# PHUMA P10 supplemental walking evaluation

Date: 2026-10-25, Beijing time. Status: P10B independent checking completed, zero of fifteen references admitted; supplementary dynamics unavailable. Original P10 wrapper failure retained.

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

## P10首次转换完成、检查入口失败

05:26:20至05:26:57首次guard执行，15片转换完成、原转换器初检13片通过；调用独立检查时误导入`supermimic_for_q1/verify_reference.py`同名模块，`AttributeError: module 'verify_reference' has no attribute 'verify'`，guard为WORKER_FAILED、worker原生退出1且回收。没有独立十九门或completion，不把13称准入。原失败目录`PHUMA_P10_walking_holdout_20261025`保留，转换manifest56320字节/SHA`e315b2240e369d4d5a72780c7827f921cf3698fd249e6d865e3fbe0375ada26b`；来源/轨迹约9.2MB，内存15.47GB/OOM14/盘余935.82GB，R02D继续1000新增轮。

修正仅对新wrapper按绝对路径独立加载原PHUMA检查器，沿用旧prepare_expansion_batch的同名模块隔离做法。新增`--verify-existing-from`要求旧guard失败且worker已回收、转换完整、15条源记录及selection完全一致、没有原独立check/completion；新目录重新做原扫描核对及十九门，保留旧失败，不重复IK或复制大型工件。新执行须使用`PHUMA_P10B_walking_check_20261025`并显式引用原失败目录；实际补检尚待。

## 05:30独立补检完成：0/15准入

修正发布`b7ad46cd24308fbc60fb3988b49fbe200004ab2d`已推送并独立核Q1远端，wrapper逐SHA部署一致。真实导入预检刻意把同名旧目录放搜索首位，仍正确加载PHUMA检查器、verify/check可调用且无CUDA上下文。05:29:45至05:30:14新P10B guard/worker均原生退出0并回收，重新扫描核对15原始成员、对原转换逐片执行全部十九门，0重复IK/积分/optimizer。原P10 WORKER_FAILED保持。

[补检完成收据](../data/manifests/supermimic_q1_P10B_completion_20261025.json)、[完整十九门](../data/manifests/supermimic_q1_P10B_reference_check_20261025.json)、[扫描核对](../data/manifests/supermimic_q1_P10B_scan_check_20261025.json)及[guard](../data/manifests/supermimic_q1_P10B_guard_20261025.json)已原样取回。十九门收据38300字节/SHA`b9a819074a6c28356247b479532d01bec62534e619e0544e1fb813440fcfc2fb`；准入0/15，足底15mm内比例失败15、最大30mm间隙失败14、自穿透3mm失败6、转换初检失败2，计数重叠。没有转换错误；足底15mm内比例3.51%至45.64%，而既有要求90%。

该固定池包含circle_walk、normal_walk、walking1等名字，但零准入意味着仍没有可交付的补充步行动力学测试，不能把原fitness移动测试升格为步行证明。也不能仅由此断言限速是唯一原因；源足底、形态映射、硬root和速度限制都须用train样本区分。接触缺口已在D02C/D04发现，本次不重复其相同样本诊断、不放宽门或逐帧根高硬投影。

下一以已使用训练划分的明确walking来源做有限映射/限速归因，排除本15个预留目录及所有dev/test来源候选；不根据这15片选修补参数。R02D保持原340输入与固定7000新轮，主目标未达。补检结束内存14.25GB/OOM14、盘余918.30GB；P10/B新资产合计不足10MB，所有大型原件LOCAL_ONLY。
