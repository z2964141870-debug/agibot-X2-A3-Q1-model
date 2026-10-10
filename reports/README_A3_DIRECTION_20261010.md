# A3 保持预训练能力的方法筛选与梯度诊断

记录日期：2026-10-10。E07失败保留，E08零更新梯度诊断完成；尚无新策略提升证据。服务器 `hp3090:/media/yu/FAFF-E9771/YUANQI_A3`，main。旧R01-R09账本不修改。

## 目标与研究依据

用户授权继续结合论文找可长期优化方向。官方预训练权重是已验证基线，公开一个权重不能证明它已经最优，也不能证明改善必须来自新算法。当前官方20动作0跌倒、关节RMSE0.0590597rad；最近R09缩小更新后仍差2.00%。先排除更新合同问题，再检验保留基础能力的适配方法，不用更多更新代替效果验收。

2026-10-10检索原始来源；以下为阅读与本项目假设，不是A3算法复现：

| 来源与版本 | 可借鉴内容 | 本项目适用边界与优先级 |
| --- | --- | --- |
| [ADEPT，arXiv:2608.19182v1](https://arxiv.org/html/2608.19182v1)，2026-08稿，本文不是当天首发 | 直接RL微调预训练策略会退化；使用行为约束、价值预热和保守更新的组合 | 灵巧手而非A3；各机制分别做对照，不能直接照搬组合或推断提高幅度。优先候选为固定官方参考策略约束；当前价值解释度并非所有批次失效，不先认定必须重训critic |
| [SONIC第三方跨本体迁移项目](https://sonic-agibot-x2.github.io/sonic-transfer/)及其v0.2论文，2026-09-12 | 冻结原策略、解析codec与decoder LoRA、目标动作适配 | A3同本体无需跨机器人codec；少参数适配列第二优先。此前公开包无完整训练配方，项目与论文留出描述冲突；不是智元官方或已复现证据，详情见README_A3_TRANSFER_20261009.md |
| [MimicX，arXiv:2610.09055v1](https://arxiv.org/html/2610.09055v1)，此前10/10简报已登记 | 依据失败的时间/身体部位调整参考、奖励与reset候选，并以闭环反馈选择 | 本轮重读方法作为失败驱动流程候选；不能改变动作后仍当同一基线，也不把多候选额外预算藏掉。当前先查PPO一致性，未引入自动改奖励 |

不把LoRA+PPO+A3、参考KL或失败驱动调参本身称为原创。未来研究贡献需定位实际遥操限制、可靠目标数据、新动作/会话验证和等预算朴素对照。已反复筛选的4留出仅诊断验证，非新的最终测试。

## 实际完成

新增独立 `GradientProbeTrainer` 与受控入口，官方初始化、16环境、24策略步、4批，epochs1、R09的LR尺度与原16/4划分。拦截第一次loss，在任何optimizer.step前分别计算policy surrogate、价值、熵和aux梯度；预钩子禁止optimizer.step。记录各网络梯度范数、夹角、优势/价值解释度和初始概率比；不训练新策略、不追加200/10000。

E07严格初始参数等价检查失败，失败job/state/console保留。E08独立复核发现唯一差异为`policy.std`，严格等于官方前向中的原地std限幅；代码 `actor_critic_modules.py:get_std`。不把这步限幅归为优化器更新，且不能声称从加载至结束所有参数完全不变。梯度诊断前后完整参数SHA相同，global_step=0、optimizer state为空、真实optimizer步记录为空。

E08共4批、16环境×24步新回放，未执行参数更新。保存 `rollout.pt` 7791609字节，SHA `1c0556a64ef046831f80db8324aeb6b4a03ee3b5b1c1fac5cce23243537dc74c`。保存初始PT并独立CPU重载、大小/SHA、网络/优化器有限性与计数验证；捕获数据全有限。3项CPU契约覆盖参数修改检测、梯度冲突及断开的critic、禁止optimizer更新。大型工件均LOCAL_ONLY。

## 诊断结果与下一决定

| 批次 | decoder PPO/aux梯度范数 | encoder PPO/aux范数 | 价值解释度 | 未更新初始ratio范围 |
| --- | --- | --- | --- | --- |
| 0 | 153.897 / 0.03371 | 38.713 / 0.12871 | 0.69980 | 0.52963-1.37392 |
| 1 | 36.480 / 0.06070 | 8.475 / 0.05747 | 0.91773 | 0.69441-1.51637 |
| 2 | 18.599 / 0.01430 | 5.095 / 0.04142 | -0.49196 | 0.56467-1.96424 |
| 3 | 25.027 / 0.05773 | 8.631 / 0.05538 | 0.78257 | 0.74723-1.71412 |

这批数据不支持“aux梯度主导actor更新”，也不支持“所有critic初始估计完全失效”。范数是原始梯度而非Adam后的参数步长；不能据此证明aux无作用或历史退化的因果原因。value梯度不直接连到actor；价值估计通过优势影响策略。仅一批初始rollout，未覆盖后续状态。

新线索优先于直接加LoRA：参数未更新时训练前向与采样保存的动作概率比明显偏离1。可能包括批量前向/量化舍入、训练与采样的输入或模式差异、padding等；本轮尚未归因。下一项在同一捕获回放上比较eval/train、单帧/批量和量化前后行为，先验证PPO概率一致性。若一致性合同成立仍退化，再先测试单一参考策略约束，与R09同预算对照；少参数decoder适配排第二。不混入奖励/数据/seed变化。

## 复现、版本与证据

起始main `d28561b966309401a57432b53e73fbe7e50b1b33`；vendor固定`fe6868ba37034f89b912f0fb851bce19f120266d`只读。官方PT SHA、split SHA及本轮代码SHA见job与Git清单，E07/E08独立账本。相同boot `6a7552b6-3c5a-4a51-929f-0a8d2d52dc79`。服务CPUQuota100%，既有CPU90/GPU85连续2次5秒保护；结束时E07 failed、E08 inactive，无新训练。

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
env PYTHONPATH=.:script/vendor/sonic_for_a3 OMP_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m unittest script.a3.test_gradient_probe -v
data/environments/a3-sonic/bin/python -m script.a3.run_gradient_probe --experiment E08
data/environments/a3-sonic/bin/python -m script.a3.gradient_report
```

完成的E08入口直接跳过；未完成的目录不覆盖重跑。原始证据 `data/experiments/a3_gradient_20261010_E07/`与E08，日志 `logs/a3_gradient_20261010_E07/`与E08；清单 `data/manifests/a3_gradient_20261010.json`。首次失败也有价值，不能从报告删掉。

## 保存与未完成项

阶段README、索引/current state与小型清单按main审计、scoped提交推送并比较远端；提交号以Git历史为准。研究没有安装外部训练代码、修改vendor/Q1、恢复旧训练或操作机器人。Isaac独立metrics、RKNN、动捕质量和现场时钟仍按全链路README搁置，不重复安装。尚无正向效果、新动作泛化或真机通过。
