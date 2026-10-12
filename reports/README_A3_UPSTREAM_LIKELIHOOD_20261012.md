# A3 官方训练说明与零更新偏差核验

记录日期：2026-10-12。阶段：只读官方来源核验。状态：完成所列公开来源检查，未找到当前零更新偏差的直接说明或修复；原因仍未定位。

## 目标与范围

回答用户“SONIC A3官方是否说明过这种情况”。这里的情况指ZHY E04/E05中尚未执行optimizer时，真实首训练batch与采样的动作分布已有偏差，不是原hp3090整机重启。

只读检查智元官方 `AgibotTech/sonic_for_a3` 的README、035训练合同、训练到部署指南、公开issue及维护者回复、训练曲线入口和当前main与固定vendor的差异。没有启动训练、评测、Isaac或CUDA，也没有更新vendor、依赖或驱动；没有联系维护者。此范围不涵盖私有讨论或所有NVIDIA上游issue。

## 查到的官方说明

| 来源 | 官方明确内容 | 与本地问题的关系 |
| --- | --- | --- |
| 035训练合同 | 默认每rank4096env、24步、5epochs、4minibatches；公开selected20 CSV不用于发布的035模型训练，只支持转换/冒烟/warm-start等，不能替代大训练语料 | 本地16env诊断与官方默认不同，不能直接宣称已复现官方缺陷；数据限制不直接解释尚未更新时的概率偏差 |
| 035训练合同 | 发布PT只warm-start actor/critic；resume=false时optimizer、scheduler、sampler、环境和计数重置，不是完整状态续训；配置和资产须与权重匹配 | 是官方明确的使用边界，但没有说明同一固定Actor在采样/首训练前向应发生当前偏差 |
| 公开issue #1 | 提问者报告MPJPE-L/G未按NVIDIA指南收敛，询问训练曲线；维护者2026-09-25回复并提供曲线页面 | 有训练不收敛的讨论；所读回复没有分析本地零更新ratio异常或给出修复 |
| 官方训练曲线 | 发布024/035的TensorBoard标量快照；页面说明不同run设置/奖励可能不同，不是共同评测基准，也不包含其他run或文本日志 | 可以用于理解官方训练过程；曲线存在不能证明我们的接入路径正确 |

2026-10-12的GitHub API `issues?state=all&per_page=100` 返回1条closed issue，无PR条目；issue #1有2条维护者回复。检查这些内容和所列文档，未找到直接解释“零optimizer更新的真实首batch与采样不一致”的说明。这个检索结果不等于证明官方所有渠道都没有提过，也不等于官方代码有bug。

## 版本与源码核对

固定vendor为 `fe6868ba37034f89b912f0fb851bce19f120266d`（2026-09-25），只读Git副本保持不变。GitHub API和 `git ls-remote` 核验当前main为 `c9df753d31ef63c930b825b1b6767cabd37c1549`，提交时间UTC2026-10-09 03:33:38 / 北京时间11:33:38。

main仅比固定版本多1个提交，主题是Hugging Face下载统计/发布元数据。改动仅 `.gitignore`、`a3_hf_manifest.json`、`docs/a3_training2sim2deploy.md`、`download_from_hf.py`、`tests/test_download_from_hf.py`；没有Actor、PPO或量化器训练代码改动。因此没有证据将“升级main”作为本问题修复。

固定入口 `train_a3_035_fromscratch.sh:242` 明确 `--mixed_precision=no`；`gear_sonic/train_agent_trl.py:456-457` 开启matmul/cudnn TF32。这些是配置事实，不是官方承认或解释当前偏差。E05已统一关闭TF32而真实4环境batch仍失败，不能再次把TF32当成已定位原因。

静态阅读 `tests/test_ppo_rollout_syncs.py` 和 `tests/test_universal_token_rollout_fast_path.py`：前者主要检查同步/统计/梯度裁剪，后者对玩具模块验证动作快路径等价。后者构造 `quantizer=None`，并不覆盖发布035真实权重、FSQ、真实训练首batch的概率一致性。本轮未运行这些测试，不能把源码存在当成本环境测试通过。

## 实际证据与下一步

本地[E05](README_A3_ZHY_PRECISION_20261012.md)真实首训练路径仍有ratio约0.53–1.94；[E06](README_A3_ZHY_BATCH_20261012.md)直接小批次回放和[E07](README_A3_ZHY_WRAPPER_20261012.md)原包装器/Accelerate及梯度模式回放均复现采样，未复现真实首训练mean。离线重建正常，尚缺真实Actor入口的输入、运行状态及即时前向证据。

当前决定：保留官方模型和固定vendor，先审计我们实际训练接入、输入与运行状态；不根据不收敛issue、语料限制或默认环境数直接归因。下一工程实验应在真实首batch逐键捕获Actor输入/参数和buffer/元数据/精度/调用选项，并在同Actor即时对照，仍保持0optimizer。本轮仅查询说明，未登记或启动E08。

## 来源与可复现检查

- [官方035训练合同](https://github.com/AgibotTech/sonic_for_a3/blob/fe6868ba37034f89b912f0fb851bce19f120266d/docs/a3_035_training_contract.md)
- [官方README](https://github.com/AgibotTech/sonic_for_a3/blob/fe6868ba37034f89b912f0fb851bce19f120266d/README.md)及[训练到部署指南](https://github.com/AgibotTech/sonic_for_a3/blob/fe6868ba37034f89b912f0fb851bce19f120266d/docs/a3_training2sim2deploy.md)
- [公开issue #1和回复](https://github.com/AgibotTech/sonic_for_a3/issues/1)
- [官方训练曲线页面](https://agibottech.github.io/sonic_for_a3/training/)
- [当前main API](https://api.github.com/repos/AgibotTech/sonic_for_a3/commits/main)及[固定版本与当前main对比](https://github.com/AgibotTech/sonic_for_a3/compare/fe6868ba37034f89b912f0fb851bce19f120266d...c9df753d31ef63c930b825b1b6767cabd37c1549)

```bash
git ls-remote https://github.com/AgibotTech/sonic_for_a3.git refs/heads/main
curl -fLsS 'https://api.github.com/repos/AgibotTech/sonic_for_a3/issues?state=all&per_page=100'
curl -fLsS https://api.github.com/repos/AgibotTech/sonic_for_a3/issues/1/comments
curl -fLsS https://api.github.com/repos/AgibotTech/sonic_for_a3/compare/fe6868ba37034f89b912f0fb851bce19f120266d...c9df753d31ef63c930b825b1b6767cabd37c1549
```

## 保存与恢复

小型版本/来源/SHA清单：`data/manifests/a3_upstream_likelihood_20261012.json`。本README、索引/current state和清单按main scoped审计提交推送，实际同步结果以Git历史和远端核验为准。同次提交号不写入自身文件。本轮无模型、张量或原始训练日志新工件；既有工件仍为新机LOCAL_ONLY。
