# PHUMA 与 AMASS 原动作别名审计

记录日期：2026-10-25。阶段P07，状态：首次内存预检拒绝保留；资源恢复后正式审计完成，全源隔离仍未证明。

## 目标与范围

使用现有原始索引和已接收HumanML元数据，识别当前PHUMA R02输入与AMASS的原动作、裁片/镜像和目录受试者候选关系，防止后续扩库把不同文件名当作未见测试。此次仅审计现有合格参考，不重分R02、不执行动作下载、FK、训练或动力学。

## 实现与决定

新增`script/supermimic_q1_sources/audit_humanml_aliases.py`和4项控制。先读取独立reference_check的qualified_clips，与转换manifest完整连接；不能直接使用manifest中较宽的converter候选标志。原库1861尝试中转换初门有1431片，而独立几何后仅353片（train340/dev6/test7）。

严格解析`humanml/000000`和镜像`humanml/M000000`，映射到固定index.csv的原始source_path；不同裁片及镜像按完整原动作归并。`_poses.npy`到`_stageii.npz`的相对事件名一致单独记录，不能据此称拟合文件字节相同。集合更名只登记候选排除依据，不能用于通过准入。目录受试者候选只用于保守排除，不证明生物身份或跨集合身份。

原CSV帧区间按原文字段保存，不直接当成AMASS120Hz原始帧编号；其他PHUMA来源、隐含重复及最终划分未解决前，`source_isolation_certified=false`。AMASS原始匹配只读既有头部索引，不重新扫描180GB动作文件。

## 版本与验证

工作区`/tmp/yuanqi-supermimic-q1-20261023`，Q1，父发布`a64824ec541b5e0e5506614339471dac99443096`。HumanML固定提交`9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23`，index SHA`80226910d7259655ab5f411fd807b7651037341b4806fde025946a74d62b1065`；AMASS头部索引SHA`501e723765a71167e7186f90df0791b506191cb83fbe98a103810d86caf2a233`，入口强制匹配二者。

Mac4项控制通过：真实官方000000路径/镜像身份，非法路径/身份拒绝，集合更名候选标记，跨split不同裁片/镜像仍检出共同原动作。服务器4项控制与正式输出待。轻量只读探查95个已合格HumanML来源均有CSV对应，未发现其中原动作/目录受试者候选跨split；开发移动009636对应CMU/137/137_07。该探查没有完成全源隔离证明，正式小收据将覆盖这些计数。

## 可复现命令与产物

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  script/supermimic_q1_sources/audit_humanml_aliases.py \
  --root /opt/data/private/YuanQi_Q1 \
  --library /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_P05_base_library_20261024 \
  --humanml-index /opt/data/private/YuanQi_Q1/data/references/humanml3d_metadata_9176e8fb_B_20261025/index.csv \
  --amass-index /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P01_inventory_20261024/all_files.jsonl \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/SOURCE_P07_aliases_20261025 \
  --expected-oom-kill 14
```

正式输出预定上述目录`receipt.json`，输入四份大小/SHA、逐片对应、匹配状态和未解来源计数均保存。只用标准库，输入全读前后核验并检查28.75GiB/100GiB余量/OOM14，输出16MiB上限；不启动GPU/训练worker。小收据审核后进入Git，大原件仍服务器LOCAL_ONLY。准备记录待审核提交推送并核远端。

## 下一步

准备发布`0a8dd461f1b3aa68f2073d2dcd7e6092760aa4a3`已独立核远端Q1；服务器入口SHA`c7c6b6d79dd88af7f3d66beba9957f97a67be8b2996737dda04f8b3b8d8e359c`、控制SHA`515b42835701c48c5bed1f8b67c0c58f5b0830985126e1be49dec1f8e3510c9f`与Mac一致，4项控制通过。正式运行在读取数据前触发MemoryLimit：共享34359721984字节超过28.75GiB，OOM14；原生退出1。02:20确认输出目录未创建，因此没有正式审计receipt，不把先前轻量探查当本入口完成。

同次实核原R02已在02:08:57内存保护FAILED至5300轮，随后SSH路由超时，详见[训练故障记录](README_SUPERMIMIC_Q1_EXISTING_RESOURCES_20261024.md)。预检拒绝发生在R02终止之后，不把本次未读取输入的审计归因为此前训练内存上涨；真实进程归因待连接恢复。没有放宽任何保护或盲目重试。

恢复原SSH路由并核资源后，仍可在本未创建的输出路径首次完成审计；若路径已由外部创建则用独立新目录。后续AMASS合库/测试候选应保守排除全部已使用原动作与受试者候选，并继续查其他PHUMA来源。原R02 FAILED不重启；先独立5000轮权重核验/配对开发评测，再登记新训练。AMASS P06新映射0/4不采用，PHUMA B017失败保留，移动有效性仍待实测。

## 02:57正式运行完成

[资源恢复](README_SUPERMIMIC_Q1_RESOURCE_RECOVERY_20261025.md)回收六个已失败旧G1残留后，按上述原命令/原保护线执行，原生退出0，首次正式输出`SOURCE_P07_aliases_20261025/receipt.json`，已保存[原样小收据](../data/manifests/supermimic_q1_source_P07_receipt_20261025.json)。开始13.513GB、结束13.691GB/OOM14，输入四份前后SHA未变，0新物理/参考/optimizer。

353合格参考中95段HumanML（train94/development1）；74个AMASS相对事件名精确匹配、9个集合更名候选、12个未解。95段内部原动作及目录受试者候选跨split均0，开发移动`CMU/137/137_07`保持保守排除。另258段属于GRAB、idea400、game_motion等十个来源命名空间，跨集合生物身份、动作重复与未解匹配未完成；`source_isolation_certified=false`保持，不能声称已通过全源隔离。
