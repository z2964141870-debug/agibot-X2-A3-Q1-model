# A3当前策略新批次方向诊断：E48实现与固定合同

记录：2026-10-13晚。状态：2026-10-14北京时间00:02:36单次COMPLETE，0worker；E48禁止复跑，E47保持COMPLETE。编号与文件名沿登记时固定20261013，实际跨日时间明确记录。

## 目标与依据

E46/E47同一E45首batch在step4上原始/Adam/有限teacher都向内，不能解释历史新rollout第19步超门。保存日志核实：真实第17步LR6.75e-6，18/19步1.0125e-5；E47旧batch的KL使同一规则降至3e-6。E45目录只有初始rollout/token/batch张量，没有第5rollout完整张量，不能做真正历史17–19步重演。已读固定vendor的_rollout_step、_get_rollout_data、_get_mb_rollout_data、train：原流程在24步后用带末帧的轨迹value计算GAE，再由环境索引切batch。

本轮不再在旧batch上追加权重/LR试探。下一主变量固定为：**同一个保存step4 Actor/Critic/Adam状态，诊断batch从旧官方初始策略数据换成当前step4策略新采数据**。新采不等于历史失败样本；若不能复现向外，结论仅为当前采样未复现，不能换seed/batch反复挑结果。

## 固定实现合同（已执行，历史登记条件保留）

1. 新编号E48，单一独立任务，绝不调用E44/E45 run/verify/evaluation，不加载其训练恢复入口。加载E45 milestone4同一SHA，只读其Actor/Critic/optimizer；四衣服语料、原原生reset分布、16环境、24步、seed0，1个新rollout/384交互。奖励、路由、teacher、梯度裁剪和自适应LR规则都不改。
2. 在新环境reset后由当前step4策略采样，用原rollout/GAE/索引切片逻辑，保存实际token、动作、logprob、old_mu/sigma、obs、dones、returns/advantages及采样初末状态摘要。固定取原randperm产生的第一个4环境batch，不看结果再选batch。采样/诊断严格串行，Actor frozen codec/std/参数buffer保持不变，禁止optimizer.step或保存新策略checkpoint。
3. 需要明确Critic normalization范围：采样沿原train/rollout规则，记录所有临时buffer变化；方向诊断开始前将临时Critic恢复到E47相同保存step4 buffer，再做一次原train前向。由此对比固定Actor/Critic/moments下的batch来源，**不是无缝续训上下文**。不可将eval/常数value替代完整Critic梯度。源checkpoint不动。
4. 用同一原vendor目标、first96teacher软项和完整384teacher，沿E47已核验完整参数映射/原0.1全局clip/原KL调LR/保存Adam公式，只算1个数学候选。保存原始下降、g_teacher·delta、有限teacher变化与原.05/.5门。Actor偏导和常数value控制需逐张量吻合，源Actor/输入/保存optimizer前后SHA不变。
5. 新采首概率门action RMSE<=1e-5、ratio最大偏离<=.001、token为实际采样而非重编码。首次任何合同/计算失败即STOPPED，不自动复跑，不追加微批次。候选超过teacher门只记方向拒绝，不部署或进一步评测。
6. 预算：1保存Actor+Critic状态，最多40Actor/decoder forward（24采样+10诊断，余量仅供事先核验的初始化调用）、8Critic forward（轨迹分块+诊断）、3autograd.grad、1数学候选、384交互、0optimizer/新checkpoint；单阶段900秒。实现前必须核实初始化/价值chunk的真实次数，并固定计数拦截；若不够，只能在未登记时说明并修订，不运行中加预算。若采用capture/probe分阶段，先统一登记整体预算、单次阶段和状态机，不能重复capture。

## 资源与内存

登记/启动前各三次5秒快照，原available_estimate>=8GiB，无其他RSS>1GiB或计算/GPU空闲/遥测可靠；CPU90/GPU85连续两次5秒、保留4GiB。首boot/OOM/内核/计算异常停止留证，不自动恢复。用户允许排查与适当释放确认属于本项目的不用缓存；已执行一次421.16MiB定向释放，不能反复运行旧释放脚本，不作全宿主drop_caches或改记账阈值，不能停他人进程。

## 判据、版本与下一步

- 若新batch原始方向已向外，定位到该新采样目标与teacher的局部冲突；若原始向内而Adam向外，才支持此新输入的变换冲突；若一阶向内有限向外，记录非线性。不得从一次结果推断全部历史原因。
- 若方向仍向内，则保存不能复现，不盲目另取batch；下一需讨论真实训练逐步捕获或改用有保留约束的候选接受机制，单独登记，不直接续训。
- 这不是效果评测。六衣服均RMSE改善>=1%、全部不倒、旧不倒保留、官方关节<=+2%、世界根腕<=+5%原门不变；官方035仍baseline。
- 输入/源码初始身份在[草案清单](../data/manifests/a3_fresh_batch_plan_20261013.json)。固定vendor静态SHA沿E47并追加入口、wrapper、GAE存储相关源码；新增5文件SHA已写入清单。本提交发布同步后才登记，不把实现当执行通过。
- 原始数据/模型/日志ZHY LOCAL_ONLY；Mac仅小报告/代码/清单，不安装新框架或下载新模型。记录按项目audit/scoped提交推送并核验，不覆盖并行工作。

## 本次实现、检查与可执行入口

- 独立 `fresh_batch_trial/probe/collect/protocol`，不改E46/E47任何已登记源码。采集器仅初始化原TRLPPOTrainer的storage/rollout/GAE所需属性，从未调用Trainer构造、train、resume或load_checkpoint入口。禁止Optimizer构造与AdamW.step；只读一次step4 payload。
- 静态核实原`_train_rollout_mode`为模型eval/环境train，24步后对16×25轨迹一次Critic evaluate（chunk1024）；方向计算前恢复保存Critic buffers并train。Actor无Lazy层才允许继续；预期24+10=34Actor/decoder、1+1=2Critic，自动拦截计数，原40/8上限无需扩展。
- 环境使用原create_manager_env、原配置、原空间发现reset和reset_all；工作目录沿原vendor以解析机器人资产。仅E48输出路径改变，不运行旧callback/调度/训练入口。seed0固定，但未复原历史第五rollout或DataLoader RNG，不能称历史重演。
- 输出先保存initial_environment.pt、fresh_rollout.pt（初末状态/obs、完整rollout、timeout、实际token、唯一randperm及首batch），再做方向诊断。完整映射、全模型梯度clip和Adam纯数学公式复用E47固定实现。原初概率门、保存teacher控制、全/常数value Actor梯度比对、源状态不变检查均保留。
- Mac仅AST及标准库检查：`python3 -m unittest script.a3.test_fresh_batch_protocol script.a3.test_adam_direction_protocol -v`，8项通过；`fresh_batch_trial plan`通过，无torch导入。尚未实际启动模型或仿真，因此不能宣称运行合同通过。
- 仅在ZHY根目录用系统Python登记/监护：`/usr/bin/python3 -m script.a3.fresh_batch_trial register`，然后`/usr/bin/python3 -m script.a3.fresh_batch_trial run`。登记与启动各三次5秒原资源门；目录存在/状态非REGISTERED禁止重复。原始输出`data/experiments/a3_zhy_fresh_batch_20261013_E48/`，日志`logs/a3_zhy_fresh_batch_20261013_E48/probe/console.log`。
- 本次恢复缓存维护返回COOLDOWN，未重复释放；可用估计约8.598GiB、OOM0、boot未变，但启动须重新三次核验。Git发布状态以提交与远端核验为准，原始工件仍LOCAL_ONLY。

## 单次结果（2026-10-14北京时间00:02:36）

代码固定提交`a049156cae1aa0fd26357506011feae3d94abf6e`，登记/启动前分别三次5秒预检通过；阶段00:01:48–00:02:36，共48.07秒。34Actor/decoder、2Critic、3autograd.grad、1rollout/384交互、1数学候选；0Optimizer构造/更新、0可部署checkpoint。实际四段MotionLib已加载，唯一首batch环境索引[7,10,9,12]。全部原始证据在ZHY LOCAL_ONLY，小型清单见`data/manifests/a3_fresh_batch_result_20261014.json`。

| 同一step4状态下的比较 | E47保存旧batch | E48当前策略新batch |
| --- | ---: | ---: |
| 原KL规则的KL | 0.347235 | 0.000290394 |
| 同一规则后的LR | 3e-6 | 6.75e-6 |
| 沿负总Actor原始梯度的teacher一阶变化 | -2.590118 | -0.613754 |
| 沿Adam候选位移的teacher一阶变化 | -2.732338e-5 | +6.256352e-5 |
| 有限候选teacher MSE变化 | -2.417056e-5 | +7.634633e-5 |
| teacher RMSE | .046055388→.045792229 | .046055388→.046876915 |

**该固定新batch上，原始总Actor下降方向向内，而保存Adam变换后的候选向外。**有限步也增大teacher偏差，因此这次支持研究实际参数位移的保留约束。候选RMSE仍小于原.05、maxabs .239271小于.5，未触发保护。不能将其称为历史第19步复现，不能把Adam历史矩、逐坐标缩放、完整clip和自适应LR各自贡献拆开归因；LR是同一规则对新KL的结果，不是另扫的变量。总Actor含原teacher96软项，不把它误称为仅PPO梯度。

首概率RMSE2.9845e-7、ratio最大偏离1.1444e-5，通过原1e-5/.001门。完整Actor+Critic梯度范数28.25528、clip .00353916；Actor与常数value控制梯度最大差0。完整teacher起点与E47一致。Actor参数/buffer、保存Actor/Critic/optimizer、输入均未变；采样Critic eval buffers未变，恢复后train只改变临时running_mean。最终结论`RAW_DESCENT_INWARD_ADAM_OUTWARD`。**这仍是局部诊断，没有新闭环收益，官方035仍baseline。**

11健康样本同boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`，OOM0，无新严重内核/遥测错误；CPU最高43°C、GPU33°C/127.36W，最低原可用估计4.513GiB。登记到启动存在49.86秒间隙，不能声称全程连续监控。采集期间MotionLib短暂子进程属于本次作业，未与其他计算任务并行。

## 退出核验与未解决项

`result.json`已在app.close前原子写入；退出码0。IsaacSim默认fast_shutdown使app.close之后的打印/finally及worker末尾SHA检查未执行，`progress.json`缺失，不捏造补写。已用独立标准库在退出后核对全部登记源码/输入SHA和最终计数，保存`postexit_verification.json`；无模型重放，既有result保留。下个独立实验必须把最终progress/文件核验放在app.close之前或父监护结束后，而非修改或重跑E48。

计算结束后的缓存维护在可见进程检查处拒绝（`Unresolved process visibility`），没有释放/删除文件；只读复核当前为176个僵尸及原3个小基础休眠进程、无RSS>1GiB进程，GPU26°C/0%/13MiB、同boot/OOM0，可用估计8.225GiB。旧错误未记录PID，所以不声称查明当时身份；先补充维护失败的PID/comm/阶段留证，按现有范围再处理，不能放宽检查或盲复跑。

## 下一单变量

不再重复E46–E48，不恢复E44/E45。下一独立设计E49：固定E48保存batch与E45 step4/moments，只改变数学候选位移是否满足完整teacher一阶非增约束；不重采样、不扫LR/权重、不部署。原Adam方向向外才做一次半空间投影，再以完整384teacher有限步原门和固定batch目标检查；有限步不合格即拒绝，不连续试缩放。研究这个约束能否修正局部位移后，再另登记短训练与原六衣服/四官方效果评测，不能把投影通过直接算baseline收益。
