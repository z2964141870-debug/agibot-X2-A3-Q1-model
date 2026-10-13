# A3实际候选位移的能力保留约束：E49未登记草案

记录：2026-10-14。状态：方案已记录，尚未实现/登记/启动；编号需要恢复时重查。官方035仍baseline。

## 目标与依据

E48固定新batch已见原始总Actor梯度下降向内、保存Adam位移向外，且有限teacher MSE增加7.63463e-5。候选尚未超过原.05/.5保护，所以不能称触发了历史停止。下一单变量是对**实际Adam位移**加入完整teacher的一阶非增约束，而非改梯度权重或增训练次数。

## 固定草案

只读同E45 step4/moments和E48唯一保存fresh_rollout.pt内首batch/实际tokens，0新rollout/optimizer/checkpoint。完整Actor+Critic梯度、原0.1clip和原KL调LR复用，先核验原E48候选的标量和方向；不从另一个batch开始。

记g为完整384teacher MSE梯度，delta为原Adam实际位移。仅若g·delta>0，计算一次`delta_projected = delta - (g·delta)/(g·g) * g`。g范数为0或输入非有限即拒绝，不用极小数掩盖。FP32舍入后的真实位移重新核验一阶条件；容差须实现前固定。

随后完整384teacher有限步检查原RMSE<=.05/maxabs<=.5，另外要求MSE不增、固定同batch Actor目标不增。这是新候选诊断的更严格判据，不改变原训练效果门。投影到切平面不保证非线性有限步合格；若拒绝就记录，不自动缩步/扫LR/换权重。通过也不生成部署checkpoint或宣称闭环改善。

预算草案：1保存状态、最多16Actor/decoder forward（原2+4+4，投影teacher4和衣服2）、1Critic train forward、3autograd.grad、原候选1/投影候选1，900秒。没有仿真App，无关快速退出问题不会带入；最终progress/源文件核验需在正常退出前持久化。实现前静态核实预算，未登记时才可修订，运行中不扩展。

## 输入、保护与恢复

小清单`data/manifests/a3_candidate_retention_plan_20261014.json`包含源SHA与E48 fresh_rollout SHA `30a05c7fdad3c2cd7d022c567e33cd8a07b939b3344143127c9f9486dc8818c0`。缓存维护以本计划保护下一输入，勿将待用E48数据当闲置缓存处理。

所有计算仅ZHY既有环境，登记/启动各三次5秒原可用估计>=8GiB、无其他大进程/计算、GPU空闲/可靠遥测；4GiB运行保留、CPU90/GPU85两次5秒、900秒，首次异常停分支不复跑。E44/E45不恢复/不评测，E46–E48禁止复跑。0worker。下一完成实现与固定协议、发布同步后才登记。

## 保存与边界

这是后续计划，不是执行结果。原始工件ZHY LOCAL_ONLY，网盘未核验。本报告与清单按scoped审计提交推送；当前执行证据入口README_A3_FRESH_BATCH_20261013.md。若局部约束不合格先记录原因，不再连续堆同类零更新实验；若合格，另设短训练及完整匹配效果验证，官方基线及未见测试要求不变。
