# A3两次教师早停后的优化与保留约束审查

记录日期：2026-10-13晚。状态：静态源码/已保存记录审查完成；下一离线梯度试验待实现，未登记、0新模型加载/forward/仿真/optimizer。恢复先读[E45停止](README_A3_CLOTH_FAILURE_20261013.md)。官方035仍基线，尚无新效果改善。

## 目标与触发

E44原三段和E45加入zzx的四段训练，都在第19次optimizer超过同一0.05教师动作RMSE门。遵照预定决策，结束语料/曝光分支。问题改为：在既定冻结范围内，衣服PPO的更新方向与旧动作保留项如何相互作用？旧能力保持不能依靠“encoder已冻结”或“teacher loss非零”推定。

本轮没有重复论文筛选。沿用[10/13原文与源码研究](README_A3_OUTCOME_RESEARCH_20261013.md)的候选范围；也读了[E11旧梯度结论](README_A3_ZHY_GRADIENT_20261012.md)，它是较早官方参考、广泛可训练Actor、没有现有teacher项，不能替代当前冻结codec/衣服输入的诊断。

## 核实到的实现结构

| 项目 | 实际源码/配置 | 支持的结论 |
| --- | --- | --- |
| teacher训练项 | `cloth_constrained_trainer._compute_loss`每步只取4环境×24帧，四组轮转；系数1 | 当前优化的是96帧teacher MSE；每步保护却检查全部384帧，二者覆盖不同 |
| teacher保护 | `_after_optimizer_step`先调用基类记录已返回的optimizer，再全384帧无梯度forward，最后越门抛异常 | 这是更新后的停止保护，不是提交更新前的可行性约束；不应把触发保护称为训练完成 |
| 初始teacher | E45初始anchor最大差0、第一步retention_loss=0 | 精确匹配时该平方误差对参数的一阶梯度为0；不能用它在第一步提供非零反向拉力 |
| PPO学习率 | 固定vendor `TRLPPOTrainer._compute_ppo_loss`用衣服当前batch对本轮old_mu的KL调整LR | 该KL不是对官方384帧锚点的偏移约束；LR较低也不保证多步累计满足teacher门 |
| 实际LR | E45末步两个optimizer组均约1.0125e-5；初始配置2e-6、上下界1e-6/2e-5 | 这是原自适应规则的结果，不能把全程当作固定2e-6；未修改LR或重跑 |
| 梯度裁剪 | vendor `_gradient_clipping`对`model.parameters()`全局范数裁剪，实际max_grad_norm0.1 | 不能直接用未裁剪Actor单项范数解释Adam最终步长；Critic可影响共同缩放，不代表有Critic→Actor梯度 |
| 辅助项 | 已启用`TRLAuxLossPPOTrainer`；当前g1_recon系数0.01读g1_kin输出，a3_fast_g1_latent系数1读编码latent，reencoded系数0 | 静态依赖未见这些有效项直接读取g1_dyn动作输出；这些模块又已冻结，应先测当前dyn的aux梯度是否为0，不能先假定aux在竞争 |
| 旧修复 | `VerifiedFineTuneTrainer`保留已修正自适应LR调度器及辅助loss继承 | 不重新诊断或恢复旧“辅助项遗漏/constant scheduler覆盖LR”缺陷；当前以固定代码和实际记录为准 |

路径/方法行号与SHA见[静态证据及下一提案](../data/manifests/a3_retention_conflict_plan_20261013.json)。读取仅源码/配置/JSON及文件大小SHA，无新张量计算。

## 尚不能作出的结论

- 两次19步早停不能证明衣服PPO与teacher梯度夹角为负，也不能证明Adam动量是唯一原因。
- teacher动作RMSE是固定开发状态上的原始动作输出差，不是关节rad、摔倒率或闭环旧能力保证。
- 当前日志有标量损失/更新编号/LR，没有停止前每个batch的完整独立梯度和原始Adam步；不能重建第19步的唯一历史因果。
- 不能将较早E11的“PPO梯度强于aux”直接迁移成E45的测量结论；本次只有静态依赖推断。

## 下一固定问题与有限试验草案

计划独立E46（编号先检查未占用，尚无账本），只做保存状态的局部梯度分解，**不启动新训练或闭环效果评测**。主问题：E45第4完整更新保存状态上，同一已保存衣服batch的PPO下降方向，会使全384帧teacher MSE在一阶近似下增大还是减小？这与原E11的无teacher/不同训练范围问题不同。

1. 固定E45 step0作初始合同控制、step4作诊断状态；复用已保存`initial_token_repair.pt`中的同一衣服batch/真实采样tokens、原384teacher输入和targets、E45配置。输入SHA先登记，不能重新采样选择有利batch。
2. 只分解同一状态/输入下dyn decoder的PPO、有效aux、轮转96帧teacher、全384帧teacher及总梯度；输出范数、点积/夹角、`g_teacher · (-g_PPO)`和`g_teacher · (-g_total)`。零范数标为未定义，不伪造余弦。
3. 先用实际vendor loss方法核对分解加和等于总目标及梯度；初始cached概率门沿用1e-5/0.001。Actor参数、buffer、保存optimizer状态前后SHA必须不变；禁止调用optimizer.step或写入可部署checkpoint。
4. 草案预算最多2个策略状态、24次Actor forward、12次autograd.grad、0新仿真/rollout/optimizer，单阶段900秒。代码实现时须计数控制可覆盖上述诊断，否则先修订未登记草案并说明原因，不运行超预算。原3次5秒/8GiB入口及4GiB、温度保护不变。
5. 使用第一batch在step4上是**固定输入局部反事实**，不是第17–19步真实batch；不声称复现早停历史，也不用于效果选点或继续E45。step0用于合同控制，不重复旧效果试验。
6. 若点积支持原始梯度冲突，再单独设计对实际更新的保留约束；若不支持，下一检查Adam状态/非线性累积。两种情况都不先增加teacher系数、不先关aux、不改奖励/语料、不做权重搜索。局部改善仍必须由后续独立训练和完整匹配效果证明。

这一草案不是已实现或已授权绕开保护的训练入口。新源码、预算、输入、判据全部固定并通过资源门后才可执行；任何异常停止新分支，不续跑旧分支。

## 保存与恢复

E45停止材料已推送核验`4875463124d3a921295f22bc77a73db2edec90c7`并同步ZHY。此静态结论、当前状态、索引和提案清单随后scoped审计/提交推送，实际号以Git历史和远端核验为准。模型/原始日志均ZHY `LOCAL_ONLY`，未核验网盘。

下一从本README实现有计数保护的离线诊断。先核验Git/实际作业/boot/GPU/容器额度和新编号未占用；不重做本次静态阅读、原E11或E44/E45训练。资源不满足保存待启动并静默，工程heartbeat继续ACTIVE；首真实机器故障或达到验收目标才暂停。
