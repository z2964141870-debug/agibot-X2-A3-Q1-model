# A3当前策略新批次方向诊断：E48未登记草案

记录：2026-10-13晚。状态：源码路径静态核验/方案固定，尚未实现、尚未登记、0worker。E48目录目前未发现；启动时仍须重查。E47已完成并禁止复跑。

## 目标与依据

E46/E47同一E45首batch在step4上原始/Adam/有限teacher都向内，不能解释历史新rollout第19步超门。保存日志核实：真实第17步LR6.75e-6，18/19步1.0125e-5；E47旧batch的KL使同一规则降至3e-6。E45目录只有初始rollout/token/batch张量，没有第5rollout完整张量，不能做真正历史17–19步重演。已读固定vendor的_rollout_step、_get_rollout_data、_get_mb_rollout_data、train：原流程在24步后用带末帧的轨迹value计算GAE，再由环境索引切batch。

本轮不再在旧batch上追加权重/LR试探。下一主变量固定为：**同一个保存step4 Actor/Critic/Adam状态，诊断batch从旧官方初始策略数据换成当前step4策略新采数据**。新采不等于历史失败样本；若不能复现向外，结论仅为当前采样未复现，不能换seed/batch反复挑结果。

## 未登记实现合同

1. 新编号E48，单一独立任务，绝不调用E44/E45 run/verify/evaluation，不加载其训练恢复入口。加载E45 milestone4同一SHA，只读其Actor/Critic/optimizer；四衣服语料、原原生reset分布、16环境、24步、seed0，1个新rollout/384交互。奖励、路由、teacher、梯度裁剪和自适应LR规则都不改。
2. 在新环境reset后由当前step4策略采样，用原rollout/GAE/索引切片逻辑，保存实际token、动作、logprob、old_mu/sigma、obs、dones、returns/advantages及采样初末状态摘要。固定取原randperm产生的第一个4环境batch，不看结果再选batch。采样/诊断严格串行，Actor frozen codec/std/参数buffer保持不变，禁止optimizer.step或保存新策略checkpoint。
3. 需要明确Critic normalization范围：采样沿原train/rollout规则，记录所有临时buffer变化；方向诊断开始前将临时Critic恢复到E47相同保存step4 buffer，再做一次原train前向。由此对比固定Actor/Critic/moments下的batch来源，**不是无缝续训上下文**。不可将eval/常数value替代完整Critic梯度。源checkpoint不动。
4. 用同一原vendor目标、first96teacher软项和完整384teacher，沿E47已核验完整参数映射/原0.1全局clip/原KL调LR/保存Adam公式，只算1个数学候选。保存原始下降、g_teacher·delta、有限teacher变化与原.05/.5门。Actor偏导和常数value控制需逐张量吻合，源Actor/输入/保存optimizer前后SHA不变。
5. 新采首概率门action RMSE<=1e-5、ratio最大偏离<=.001、token为实际采样而非重编码。首次任何合同/计算失败即STOPPED，不自动复跑，不追加微批次。候选超过teacher门只记方向拒绝，不部署或进一步评测。
6. 预算草案：1保存Actor+Critic状态，最多40Actor/decoder forward（24采样+10诊断，余量仅供事先核验的初始化调用）、8Critic forward（轨迹分块+诊断）、3autograd.grad、1数学候选、384交互、0optimizer/新checkpoint；单阶段900秒。实现前必须核实初始化/价值chunk的真实次数，并固定计数拦截；若不够，只能在未登记时说明并修订，不运行中加预算。若采用capture/probe分阶段，先统一登记整体预算、单次阶段和状态机，不能重复capture。

## 资源与内存

登记/启动前各三次5秒快照，原available_estimate>=8GiB，无其他RSS>1GiB或计算/GPU空闲/遥测可靠；CPU90/GPU85连续两次5秒、保留4GiB。首boot/OOM/内核/计算异常停止留证，不自动恢复。用户允许排查与适当释放确认属于本项目的不用缓存；已执行一次421.16MiB定向释放，不能反复运行旧释放脚本，不作全宿主drop_caches或改记账阈值，不能停他人进程。

## 判据、版本与下一步

- 若新batch原始方向已向外，定位到该新采样目标与teacher的局部冲突；若原始向内而Adam向外，才支持此新输入的变换冲突；若一阶向内有限向外，记录非线性。不得从一次结果推断全部历史原因。
- 若方向仍向内，则保存不能复现，不盲目另取batch；下一需讨论真实训练逐步捕获或改用有保留约束的候选接受机制，单独登记，不直接续训。
- 这不是效果评测。六衣服均RMSE改善>=1%、全部不倒、旧不倒保留、官方关节<=+2%、世界根腕<=+5%原门不变；官方035仍baseline。
- 输入/源码初始身份在[草案清单](../data/manifests/a3_fresh_batch_plan_20261013.json)。固定vendor静态SHA沿E47，尚未有E48实现版本或执行命令，不能把草案当已开始。下一实现前先核实最新README/目录/boot/资源，预算固定后scoped发布同步再登记。
- 原始数据/模型/日志ZHY LOCAL_ONLY；Mac仅小报告/代码/清单，不安装新框架或下载新模型。记录按项目audit/scoped提交推送并核验，不覆盖并行工作。
