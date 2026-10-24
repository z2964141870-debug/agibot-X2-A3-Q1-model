# PHUMA 与 AMASS 原动作别名审计

记录日期：2026-10-25。阶段P07，状态：实现/本地控制完成，正式服务器审计待运行。

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

实际执行本审计；后续AMASS合库/测试候选应保守排除全部已使用原动作与受试者候选，并继续查其他PHUMA来源。AMASS P06新映射0/4不采用；并行PHUMA B017因内存保护仍暂停，先定位资源或待R02释放，不盲目重试。R02继续原12000轮预算和终态配对评测，移动有效性仍待实测。
