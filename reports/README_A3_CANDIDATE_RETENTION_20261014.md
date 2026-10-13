# A3实际候选位移的能力保留约束：E49实现与独立检验

记录：2026-10-14。状态：实现完成，尚未登记/启动；编号已核实空闲，登记再次拒绝已有目录。官方035仍baseline。

## 目标与依据

E48固定新batch已见原始总Actor梯度下降向内、保存Adam位移向外，且有限teacher MSE增加7.63463e-5。候选尚未超过原.05/.5保护，所以不能称触发了历史停止。下一单变量是对**实际Adam位移**加入完整teacher的一阶非增约束，而非改梯度权重或增训练次数。

## 固定实现与预算

只读同E45 step4/moments和E48唯一保存fresh_rollout.pt内首batch/实际tokens，0新rollout/optimizer/checkpoint。完整Actor+Critic梯度、原0.1clip和原KL调LR复用，先核验原E48候选的标量和方向；不从另一个batch开始。

记g为完整384teacher MSE梯度，delta为原Adam实际位移。仅若g·delta>0，计算一次`delta_projected = delta - (g·delta)/(g·g) * g`。g范数为0或输入非有限即拒绝，不用极小数掩盖。FP32舍入后的真实位移重新核验一阶条件；登记前容差已固定，见下文。

随后完整384teacher有限步检查原RMSE<=.05/maxabs<=.5，另外要求MSE不增、固定同batch Actor目标不增。这是新候选诊断的更严格判据，不改变原训练效果门。投影到切平面不保证非线性有限步合格；若拒绝就记录，不自动缩步/扫LR/换权重。通过也不生成部署checkpoint或宣称闭环改善。

预算：1保存状态、最多16Actor/decoder forward（原2+4+4，投影teacher4和衣服2）、1Critic train forward、3autograd.grad、原候选1/投影候选1，900秒。没有仿真App，无关快速退出问题不会带入；最终progress/源文件核验需在正常退出前持久化。静态核实计数不变；运行中不扩展。

## 输入、保护与恢复

小清单`data/manifests/a3_candidate_retention_plan_20261014.json`包含源SHA与E48 fresh_rollout SHA `30a05c7fdad3c2cd7d022c567e33cd8a07b939b3344143127c9f9486dc8818c0`。缓存维护以本计划保护下一输入，勿将待用E48数据当闲置缓存处理。

所有计算仅ZHY既有环境，登记/启动各三次5秒原可用估计>=8GiB、无其他大进程/计算、GPU空闲/可靠遥测；4GiB运行保留、CPU90/GPU85两次5秒、900秒，首次异常停分支不复跑。E44/E45不恢复/不评测，E46–E48禁止复跑。0worker。下一发布同步实现、两轮原三次预检后登记/启动一次。

## 保存与边界

实现/5项标准库控制与AST检查已通过，不是模型执行结果。原始工件ZHY LOCAL_ONLY，网盘未核验。本报告与清单按scoped审计提交推送；当前执行证据入口README_A3_FRESH_BATCH_20261013.md。若局部约束不合格先记录原因，不再连续堆同类零更新实验；若合格，另设短训练及完整匹配效果验证，官方基线及未见测试要求不变。

## 登记前固定的实现、容差与验证

代码为`script/a3/candidate_retention_{trial,probe,protocol}.py`，单独复用旧loss/映射/监控函数；未修改E46/E47/E48已登记源码。调用装饰器计数16Actor/decoder+1Critic，3次autograd.grad；禁止optimizer构造/step和torch.save。不创建仿真App，result/progress/退出前源文件核验正常落盘。

原E48对照：标量atol1e-7/rtol1e-5、方向atol1e-8/rtol1e-4；核验同Actor/批次/teacher/保存optimizer/原Critic buffers SHA、完整映射/28参数、loss/梯度norm/clip/KL/LR/原候选方向与有限误差。只有原对照通过才计算唯一投影。

投影用FP64计算，再一次舍入为FP32参数；实际舍入后一阶变化<=1e-8，有限teacher MSE变化<=1e-9，固定Actor目标变化<=1e-7；原RMSE<=.05/maxabs<=.5保持。固定Actor目标是原cached-token PPO/entropy/aux加首96teacher，Critic值在两次目标比较中取同保存常数；另报不含teacher项的衣服loss。所有门和容差在结果前固定，不扫参数。

五项标准库控制包括：去除外向分量、内向不改、切平面通过但非线性仍增的反例必须拒绝、舍入/目标各自拒绝、零范数/非有限/控制方向翻转拒绝、预算不允许第二次投影（其中部分合并为一测试）。Mac/原hp3090均0张量/模型执行。

```bash
# 仅标准库验证；没有加载模型
/usr/bin/python3 -m unittest script.a3.test_candidate_retention_protocol -v
# 以下仅ZHY；完成发布/同步与原预检后各执行一次，不作为重跑入口
/usr/bin/python3 -m script.a3.candidate_retention_trial register
/usr/bin/python3 -m script.a3.candidate_retention_trial run
```

源SHA/输入SHA/预算及容差已写`data/manifests/a3_candidate_retention_plan_20261014.json`，登记后不改此计划。预期产物`data/experiments/a3_zhy_candidate_retention_20261014_E49/`，日志`logs/a3_zhy_candidate_retention_20261014_E49/`，均ZHY LOCAL_ONLY。实现待scoped审计推送并核对，提交号由Git历史与registration记录，不预称执行成功。
