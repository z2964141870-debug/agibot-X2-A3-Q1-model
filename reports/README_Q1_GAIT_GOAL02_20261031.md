# Q1步态目标第二轮

日期：2026-10-31，北京时间。用户明确“设定新目标，继续解决”，应用目标GAIT_GOAL02已实际创建并保持ACTIVE。当前是新方案准备，尚无本轮实训或步态改善结果。GAIT_GOAL01的901.178392秒和physical_regression负面结果完整保留。

## 验收目标

继续在SuperMimic内同时推进底座和PPO残差。固定起走、停走、抬手各三重复完整；两走各自摆脚脚底Z误差中位数降至少20%、承重滑移降至少15%，逐重复姿态和上肢/抬手误差不退超过10%。同片核实真实离地、左右交替和前进，并交付原330库评测、原值和本地原速视频。总reward、loss、存活和参考相位不替代物理验收；用户采纳保留null。

## 第一配方

独立目录script/supermimic_q1_gait_goal02，配方POSTURE_PAIR_COUPLED_V1。原跟踪奖励、控制、终止、输入特征、训练上肢混合、探索噪声及0.15rad残差范围保持。

- 底座：原额外脚部相位奖励乘躯干朝向因子和左右脚相对高度因子；仍为可微FK几何，额外奖励上限0.01/20ms、总上限0.11。
- PPO：原额外支撑/摆脚两项乘同一因子，其他奖励和成本保留，总上限0.15。
- 朝向因子使用实际与参考根旋转第三行差的平方，尺度0.15；近小角时才约等于rad，不是独立roll干预。
- 相对高度因子尺度2cm，仅在声明单支撑且参考摆脚比支撑脚高至少1cm时启用；双支撑和模糊相位不编造摆脚。

这是检验姿态和双脚几何协调的假设，不预称已补足实际承重动力学。底座该项仍无实测负载输入。PPO的末次solve力早于post姿态5ms，新增诊断始终标为delayed，不拼成同刻卸载真值。

## PPO学习链路

新增被动记录：逐相位原始/归一化优势、V/return和有限条确定性选样，实际minibatch ratio裁剪比例、KL、裁剪前后梯度。沿用现有RSL PPO损失和采样；CPU反例须证明开关记录后的损失、梯度、参数和RNG一致。原本缺少的历史优势不能由新试验补写为旧证据。

## 首轮范围

先CPU数学/真实奖励组合及真实RSL观察等价测试，再每线16和4096环境各一次零更新预检，通过后串行启动新底座和PPO各一次。两条均从原RF14 stage02开始，分别比较各自initial；不把两条独立结果或旧ACTIVITY混成未经实测的组合。

每线实训硬限1200秒、软终点1100秒，总至多2400秒；300/600/1100秒物理门，1150秒外停与10秒回收，沿用原24GiB启动准入/30GiB守卫及共享GPU锁。每线预检16外限600秒、4096外限900秒。严重物理退步立即按门停止，不自动扩时或故障重试，不恢复旧训练。

新根、配方、基线SHA及验收见[清单](../data/manifests/supermimic_q1_gait_goal02_20261031.json)。原子步观测不重跑。只有新代码必要检查和新目标明示范围可执行；CPU故障修复不构成步态收益。

## 当前阶段

独立代码、本地语法/空白检查和服务器13项CPU测试已通过。真实RSL PPO开启记录后的损失、梯度、参数、优势和Torch RNG逐值一致；20个minibatch与40次梯度裁剪记录齐，优化器step被阻断。真实CPU奖励契约51参考姿态、四种扰动共204姿态通过，根Z梯度与有限差分最大差7.57e-5；特征、原成本和终止保持。底座/PPO无环境imports及两个调度入口help均通过。以上均无新物理积分、策略推理或优化更新，不是步态改善。

原生会话94632因Windows超时在传输前断开，未写服务器。按用户一次自恢复授权先登记新故障，随后原生两段式97816恢复；2026-10-31 20:56:30核原boot/GPU/唯一根，GPU15MiB/0%、无计算、容器4447211520B、OOM0。20:57:30核准备提交a398e7a531cbe88a53653609954e0191d3947d7c的16文件386459B三端大小SHA一致，未重启宿主机/容器或使用转发。

21:06:55已生成[不可变执行来源plan](../data/manifests/supermimic_q1_gait_goal02_source_plan_20261031.json)，1671份，414004B/SHA25631e9772f6db73c20bbc613b3a29434047519122e1fe9e8d35be108f6bae95235。原1564依赖仍匹配，并补当前MuJoCo Warp Python及MuJoCo共享库；旧预检仅供来源身份参照，不当本轮通过结果。[CPU证据](../data/manifests/supermimic_q1_gait_goal02_cpu_20261031.json)。CPU与冻结提交4b1aae428f71ee62634061e3e5744af2f1dc9dc2已独立远端核对，8份Git记录776584B及1份1933B可变连接状态完成Mac/25386大小SHA，详见[旁收据](../data/manifests/supermimic_q1_gait_goal02_cpu_publication_20261031.json)。

## 四项新预检

底座16/4096于21:16:07、21:17:53，PPO16/4096于21:20:46、21:23:58分别正常COMPLETED，四个guard均WORKER_COMPLETED/reaped0，1671来源前后不变。底座真实采集272/69632转移、同份replay末帧与H32起点一致、有限非零梯度、模型参数不变；wm/pi/v各1/1/20次step调用被阻断，实际optimizer均0。PPO96/24576 minibatch概率差均0，奖励/终止/状态RNG与冻结教师保护通过，0optimizer。不是跨环境动力学等价或步态收益。

21:27:19四controller已退出，原boot/GPU一致，GPU15MiB/0%、无计算、容器4863082496B、OOM0。48份工件2598404B及407884B结果清单已Mac/25386逐大小SHA；权重和大NPZ未传回。[完整预检证据](../data/manifests/supermimic_q1_gait_goal02_preflight_results_20261031.json)。正式实训仍底座0秒、PPO0秒。下一步发布本阶段后，先底座一次新独立训练及完整交付，再PPO一次；每线原1100软/1200硬、300/600/1100物理门和串行GPU保护不变，不故障重试或补预算。

## 底座已启动

预检发布2ea0ccf0502790ca7211a3ac4668e1c3a997a5aa已独立远端及7份Git记录762352B三端核验，另1933B可变连接状态同步；[发布旁收据](../data/manifests/supermimic_q1_gait_goal02_preflight_publication_20261031.json)。唯一底座根Q1_GAIT_GOAL02_BASE_COUPLED_E4096_MAX20M_20261031已启动，controller344446/start36735544、worker344484/start36736788，原boot/GPU一致，1677冻结来源。21:37:06仍initial repeat1/frame400、实训0秒/optimizer0，GPU5171MiB/38度、容器8392724480B、OOM0；这是启动现场，不是改善结果。import_guard正常WORKER_COMPLETED，既有一次新底座尝试已用，只监督原进程，不重试扩时。完整[启动收据](../data/manifests/supermimic_q1_gait_goal02_base_launch_20261031.json)。PPO正式新训仍0，待底座训练及保存交付完成/GPU释放后按原独立首次授权执行。

## 底座首门结果

21:45:26底座在301.575546404696秒/127轮按physical_regression正常停，WM1270/actor117/critic2540更新，未补剩余预算。两普通走和抬手各三重复全部完整，验证失败0/0/0；停走躯干倾斜RMS为初始1.3105/1.3968/1.3051倍，触发严重退步门，抬手倾斜亦约增加14.7%。完整/存活不替代步态验收。

| 固定动作 | 摆脚脚底Z误差中位数 | 承重滑移中位数 |
| --- | --- | --- |
| 停走 | 10.753→10.500mm，降2.35% | .11389→.10932m/s，降4.01% |
| 起走 | 11.561→11.493mm，降0.59% | .14066→.10902m/s，降22.49% |

额外脚部奖励乘姿态/双脚高度因子仍未让两走脚底Z过门，起走滑移的局部收益伴停走姿态代价，不能采纳为自然步态成功，也不能据此隔离单一训练原因。[首门完整原值](../data/manifests/supermimic_q1_gait_goal02_base_gate01_20261031.json)。训练结束后原pipeline继续保存媒体，Mac解码和保存审计尚待，不预称全流程完成；PPO首次独立试验仍待底座交付后执行。渲染器图/视频标题继承GAIT_GOAL01通用前缀，实验身份以本轮GOAL02目录、2ea0ccf冻结来源和收据为准，不混同旧GOAL01结果。

CPU复现使用既有Python及activate.py，执行本目录test_goal02.py与check_candidates.py；后者的runtime/scene/library取主清单baseline。训练入口train.py加--check-imports只加载真实模块，不建环境。新预检使用本目录supervise.py，--frozen-plan指上述plan，--mode分别preflight16/preflight4096，--line base/ppo；其余唯一根、原boot/GPU/OOM0及checkpoint/library/calibration均固定在主清单。

先显式payload审计/commit/push origin Q1和独立40位远端，再镜像Mac与原生25386并核大小SHA。测试和实训结果随实际证据更新，未测试不称通过；最终媒体与保存审计完成后才称交付完成。权重/大NPZ只留25386，不改认证、环境、其他自动任务或启动agent。实际发布以publication旁收据为准。
