# PHUMA扩展库P08组装与来源核对

记录日期：2026-10-25，北京时间。阶段P08，状态：完整组库/保存时轴重算/原trainer准入通过；来源审计完成，新增4条同开发受试者目录候选需独立排除后再用于新实验。

## 目标与输入

利用现有PHUMA完整处理后的固定来源构造后续独立实验用库，先达到30分钟工程目标，再评估接触质量和扩至1小时的价值。BONES由师兄处理，AMASS未准入候选不混入。原R02C仍用340训练片，固定7000新增轮，不修改其496冻结或原FAILED终态。

输入为原P04固定1861候选及额外固定1024训练来源，共2885。只使用成功的B005至B014和B015/B016/B017B/B018/B019，原B017及其他失败输出不参与。原名单SHA`c4dbc3054ed1a99c67a05667e1dc085af187e802d9740f331036b619baa88f4f`，额外名单SHA`631cd06bd0272e279ef587abead63e8849357d34e234ee3c7dfa74af60703a20`。

## 实际执行与验证

沿用已发布的`script/supermimic_q1_expanded/assemble_library.py`，SHA`7ac8118041b6e0fc6590a03767069fa59e33c0e52fe437e45bd7f18765a1da24`，两端一致；记录前发布`90005a88ad7c401d9f2aa9dd882e5d2b839fb624`已独立核远端Q1，无新组库代码。03:46:00至20通过原MemoryBoundBudget/guarded_run执行，28.75GiB/OOM14、至少100GiB盘余、输出64MiB、nice10/单线程/无CUDA及900秒无进展保护。guard836449/worker836450，原生退出0、worker_reaped=true/return_code0。

全部2885候选按两份完整固定名单各覆盖一次；每批来源/原字节/模型/19门/收据绑定一致，合格NPZ逐文件SHA与50Hz保存时轴独立重算，完整源ZIP及划分重核、原trainer准入通过。629份合格参考只建符号链接，不重新转换或搬动原NPZ；明确`geometric_checks_rerun=false`，不把继承19门记录说成重新做几何验证。

| 划分 | 合格片数 | 保存时长 | 位移候选时长 |
| --- | --- | --- | --- |
| train | 616 | 2203.58秒 / 36.73分钟 | 366.02秒 |
| development | 6 | 21.26秒 | 3.96秒 |
| test | 7 | 27.56秒 | 11.72秒 |

额外独立核验原340训练片全部保留、development/test逐clip列表与原库完全相同，五份关键绑定大小/SHA稳定。本次没有优化器更新、物理积分或新策略。30分钟保存时长已实际重算达到；1小时尚未达到，来源独立性和动态跟随不能由时长证明。

## 命令、产物与存储

主机`huanyu:/opt/data/private/YuanQi_Q1`，沿用Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。实际完整命令向量保存在guard原件中；producer调用如下，输出需为全新路径：

```bash
DATA=/opt/data/private/YuanQi_Q1/data/supermimic_for_q1
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" script/supermimic_q1_expanded/assemble_library.py \
  --selection "$DATA/PHUMA_P04_selection_20261024.json" \
  --additional-selection "$DATA/PHUMA_P04_additional_selection_20261024.json" \
  --batches "$DATA"/PHUMA_P04_B{005..014}_20261024 \
    "$DATA/PHUMA_P06_B015_20261024/batch" "$DATA/PHUMA_P06_B016_20261024/batch" \
    "$DATA/PHUMA_P06_B017B_20261025/batch" "$DATA/PHUMA_P06_B018_20261025/batch" \
    "$DATA/PHUMA_P06_B019_20261025/batch" \
  --output "$DATA/PHUMA_P08_expanded_library_20261025/library" --expected-oom-kill 14
```

库根`data/supermimic_for_q1/PHUMA_P08_expanded_library_20261025/library/`，日志`logs/PHUMA_P08_expanded_library_20261025.log`。末内存17803849728字节/OOM14、盘余1463563833344字节，组库目录实际分配22700032字节，629软链接未重复复制数据。原件与模型仍LOCAL_ONLY，未新增网盘备份。

[完成小收据](../data/manifests/supermimic_q1_phuma_P08_library_20261025.json)26855字节/SHA`ae259e7bd185f1162c280a5f13793e20c28b251c0437c8853d0c239ce6f2b94b`；[guard](../data/manifests/supermimic_q1_phuma_P08_guard_20261025.json)3271字节/SHA`0c9423a337c757b9a356df9eda45c799a3590f265e5f6f290aa0ee7c973ca122`。manifest9297398字节/SHA`181f28896b93be7525986a89f85c5dfadbb6b2d16b79cc873e13699926eafd8b`；reference_check7116510字节/SHA`19eff6bb340a58e5acaf738b64ddb0804669bbbd60aeeaa568804995841b51c4`。两份小收据已取回，SHA核对待最终保存检查。

## 来源边界与下一步

旧P07只覆盖原353参考，不能沿用其95段HumanML统计证明本629参考隔离。下一用相同固定HumanML/AMASS索引和已发布`audit_humanml_aliases.py`审计新库，独立输出`SOURCE_P08_aliases_20261025`；发现跨split原动作或目录受试者候选则先登记、后续保守排除，原R02C不热改。

[PHUMA-D04](README_SUPERMIMIC_Q1_PHUMA_REFERENCE_DYNAMICS_20261025.md)已发现静态19门过关不等于真实接触/无滑动，故本库只能称几何准入库。先来源核对，再独立训练样本接触/速度一致性改进与原R02C配对结果；不把更多数据或已完成组库当作移动策略成功。本阶段记录待审核发布及本地恢复镜像同步。

## 03:48来源审计真实完成

原`audit_humanml_aliases.py`在新库首次执行退出0，输出`SOURCE_P08_aliases_20261025/receipt.json`，四份输入前后身份一致、OOM14，0新参考/积分/optimizer。小原件已取回：[P08别名收据](../data/manifests/supermimic_q1_source_P08_receipt_20261025.json)206609字节/SHA`ab9a344b4f9868144bf729d0aa2925a5a3ee814211afa5dc47d4b572c9cb823a`。

629合格参考中211段HumanML（train210/dev1），172个相对事件名精确匹配、13个集合更名候选、26个未解。原动作跨split为0，但开发`CMU/137/137_07`与新增4条训练片共享`CMU/137`受试者目录候选。对应训练源为humanml/001457、011773、013549、007921，总12.24秒（其中位移候选2.96秒）。目录候选不是生物身份认证，保守排除仍有必要；不能把“不是同一原动作”当作来源完全隔离。

另418段其他命名空间以及跨集合身份尚未解决，`source_isolation_certified=false`保持。P08原库及检查原件保持不改，下一独立P08B排除上述训练片并重新验证trainer准入及HumanML别名；预计612训练片/2191.34秒为排除算术，实际视图尚未创建，不预先写成完成或全源独立。

## P08B已知别名冲突排除视图准备

新增`script/supermimic_q1_sources/filter_phuma_alias_conflicts.py`，从原库的实际trainer准入名单与完整HumanML审计连接，重算held-out原动作和目录受试者候选集合，只排除有明确候选关系的训练片。原19项几何门不改；新派生manifest/check明确记录`qualification_before_source_exclusion`、排除理由及原始几何/来源收据。训练资格合并已知来源排除，不伪称原参考突然几何失败；原P08全部文件保留。

新视图只为保留参考建软链接，逐片重读保存50Hz时轴、字节与19门，再调用原trainer准入；要求dev/test列表完全不变，训练差集只等于已知排除集。`source_isolation_certified=false`保持，未解来源不会因无候选就成为认证通过。资源沿用28.75GiB/OOM14、至少100GiB盘余、源/输出64MiB，构造每32片检查一次，nice10/单线程/无CUDA；该有限元数据/链接操作没有外层0.25秒worker轮询。

Mac三项控制及语法通过：精确原动作/同受试者候选只从train排除、输入不变；缺失/重复别名或split/source变化拒绝；未解候选不自动推断关系。新producer SHA`3aa0f5b666f9d3312a02879e1feb8ef1ebe45798bce70956535e88165b09f169`，测试SHA`fe949658ad60bc1b73ba1a034f577d3ba6109914f277911bf56b4f46633aa2a2`。服务器控制与实际构造待发布部署后执行，不修改R02C冻结代码。

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  script/supermimic_q1_sources/filter_phuma_alias_conflicts.py \
  --root /opt/data/private/YuanQi_Q1 \
  --library /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_P08_expanded_library_20261025/library \
  --aliases /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/SOURCE_P08_aliases_20261025/receipt.json \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_P08B_subject_excluded_library_20261025 \
  --expected-oom-kill 14
```

实际输出/日志预定上述新目录及`logs/PHUMA_P08B_subject_excluded_library_20261025.log`。完成后再对新view运行原别名审计，独立`SOURCE_P08B_aliases_20261025`，不把排除算法自身结果当新审计。P08两份完成小原件及P08别名小收据的本地SHA已与服务器一致；本阶段新增代码/记录待审核发布。
