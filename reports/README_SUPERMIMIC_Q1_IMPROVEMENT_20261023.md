# Q1改进方案筛选与验证

记录日期：2026-10-23，北京时间。阶段：SM-Q1-S01/P07/J01/T01/B01/C01。状态：方案筛选与入口预检完成；尚未启动新训练或取得新策略效果。

修正适配器通过后端验证；J01四状态2176控制步与独立重算、T01边界力矩和B01批量资源测量完成，C01独立入口绑定预检通过。准备代码提交`73febe1`已推送Q1；本阶段结果和后续入口另行提交。原R01冻结文件未修改。

## 目标与范围

用户要求寻找提升和解决方案。保持SuperMimic世界模型、k8/H32/BPTT/critic方法复现，优先修复已有配置问题，基于Q1实际证据决定补模型精度、训练预算或控制适配。不恢复A3，不把额外观测、PPO、教师策略或新奖励混入主线；原R01数据、权重、评测和视频通道保留。

## 原始来源核对

web检索/原站入口未返回可用内容，改用直接HTTP获取作者原文。原文`https://arxiv.org/pdf/2610.28480v1`保存于服务器`data/references/supermimic_2610.28480v1_S01.pdf`，3524904字节/SHA`fe213017ab062f4acc3965da89abd3c29969163e144a2628a662b8ce46691009`，与先前原文身份一致。已读第13–16页完整相关正文，第16页已渲染PNG并实际目视核对表格、公式和脚注。第三方固定源码`CuboiLeo/SuperMimic-Reproduced` d9be385的README/PAPER_SPEC只用于交叉核查，不替代原文或视为作者官方成绩。

| 项目 | 原文Table A.5 | Q1 R01实际 |
| --- | --- | --- |
| 每卡仿真环境 | 4096 | 16 |
| 每次actor想象轨迹 | 4096 | 32 |
| WM每次更新 | 10批×2048窗口 | 10批×64窗口 |
| 训练长度 | 320000 iterations | 3000采集轮/1099 actor更新 |
| 学习率 / H / k | 2e-4 / 32 / 8 | 相同初始值 / 32 / 8 |

论文iteration与采集/更新的精确计数口径有未指明项，不能把上述长度直接换成匹配交互倍数或训练耗时。16/4096、32/4096、64/2048分别为本轮环境、actor batch、WM batch相对论文数值，不声称同等数据难度/吞吐/学习效率。原文第13页强调低PD增益下设定点是力矩代理；因此“直接播放参考角度跟不上”不证明参考物理不可行。默认关节角/动作尺度/力矩范围仍需Q1核验。

固定独立复现README报告，论文观测下e2根跟踪较差，加入9维锚点相对位姿的e3效果更好；其结果不与原文同配置，不能直接作为我们加观测的复现成功依据。主线先保留论文观测，相关观测增强只列备选诊断，不作为当前方案实现。

## 关键决定与预登记

P07：新增`contact_protocol.ContactCorrectedAssets`，仅提高足碰撞priority到floor+1，沿用原PD/动作/资产。旧`supermimic/q1_assets.py`等冻结模块不改；R01仍由原适配器重现。新`verify_contact_protocol.py`在实际Q1Sim三次随机重置（两环境）核对context与contact，确定性重置核对真实摩擦1，随后82个原生控制步（另20显式比较子步）和FK/积分/冻结H32图验证，0optimizer。诊断用64宽网络只验证图和后端，不是完整策略成绩。登记时未接入训练入口，后续C01已独立接入。

J01：新增`probe_action_response.py`，只接纳R01最终两段训练轨迹，固定每段有效H32区间25%/65%各一状态，PCG64 seed20261023，每状态4个单位动作方向、首动作±0.05/±0.10，其余31动作固定。先四个无扰动基线全部复现，再做扰动；上限4×17×32=2176控制步/43520子步，0策略推理/optimizer。保持R01实际摩擦0.6，模型context仍为旧请求参数1，不混入P07物理改变。读取H1/H8/H32的根/关节响应，与模型有限差分/JVP比较；float64模型1e-4中心差分验证JVP，float32/float64再比较。失败样本保留、无替换或自动重试；弱响应不报cosine。所有结果为描述性局部响应，不是全闭环actor梯度正确性或策略成功门。

## 命令与证据

服务器`huanyu:/opt/data/private/YuanQi_Q1`。Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。入场资源快照3090 24576MiB、已用6348MiB、GPU0%/39°C；容器usage27942223872/34359738368字节、历史OOM14、boot d994e71f-bae0-46ba-a5f7-6c2163771c6c。其他历史进程保留，不能以GPU空闲视为资源独占。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' $PY -u script/supermimic_for_q1/verify_contact_protocol.py --root "$PWD" --output data/supermimic_for_q1/P07_backend_20261023
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' $PY -u script/supermimic_for_q1/probe_action_response.py --root "$PWD" --evaluation data/supermimic_for_q1/R01_eval_final_20261023/receipt.json --output data/supermimic_for_q1/J01_action_response_20261023
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 $PY -u script/supermimic_for_q1/probe_action_response.py --root "$PWD" --output data/supermimic_for_q1/J01_action_response_20261023 --execute --device cuda:0
```

以上命令本轮已执行，输出拒绝覆盖，恢复时不要直接重跑。源码以Git和准备plan中的SHA绑定。大文件仍LOCAL_ONLY，原始PDF不进Git或Mac。

准备验证：三个新增脚本AST通过；服务器计分控制原生退出0，反向模型必须给负cosine、零物理响应抑制cosine/ratio、完全匹配响应误差为零三项通过。上述不是实际物理/学习结果；日志`logs/supermimic_q1_S01_metric_controls_20261023.log`。

## 下一步与恢复

本阶段检查已完成；下一以本文件末节和`improvement_plan_S01.json`建立修正摩擦下的batch32/128匹配对照，冻结发布版本/输入并按当时资源登记新实验。当前R02未登记或启动，不把草案当运行账本。恢复读取本文件、D02和Q1主阶段，并重新核对实际作业/资源；V01播放服务保留。报告索引和current state随结果更新，提交推送Q1并独立核验远端。

## P07实际结果

`verify_contact_protocol.py`原生退出0、`BACKEND_AND_GRADIENT_CONTRACT_PASSED`，收据统计82个Q1Sim控制步/1640原生子步，另外有20个显式`mj_step`比较子步（合计1660物理子步），0optimizer；收据physics_control_steps字段只计原生Q1Sim，不能把它当全部积分调用数。实际两环境×三次随机重置、各14个脚地接触，六个context0.896847/0.881839/0.981361/1.119740/0.441778/0.837581与各自有效摩擦逐个相同；确定性重置实际摩擦1.0。原生rollout与显式20子步PD状态最大差8.055e-9；FK/积分检查通过，H32小fixture动作梯度有限差分差5.242e-12、actor梯度非零、WM参数梯度0。与旧R01分别记录，不把修正后的摩擦追记进旧结果。

服务器收据`data/supermimic_for_q1/P07_backend_20261023/receipt.json`，日志`logs/supermimic_q1_P07_backend_20261023.log`，Git小副本`data/manifests/supermimic_q1_backend_P07_20261023.json`。它支持后续选择修正适配器，不能证明跟踪提升、完整全宽训练成功或真机参数通过。J01使用旧R01实际摩擦，确保比较对象匹配。

## J01实际结果

原生退出0、`Q1_ACTION_RESPONSE_DIAGNOSTIC_COMPLETED`。四个无扰动基线与已保存状态完全一致（最大误差0），2176控制步/43520物理子步，61.7474秒，0策略推理/optimizer。16个方向的float64 JVP/中心差分与float32/float64数值控制全部通过。192项根部/关节响应比较均满足对应时域有效掩码；每个时域/部位32项，样本来自四状态、四方向和两扰动幅度，彼此不独立。

| 时域 | 部位 | JVP方向余弦中位数 | 正余弦比例 | 模型/物理范数中位数 | 相对L2误差中位数 |
| --- | --- | --- | --- | --- | --- |
| H1 | 根位置 | 0.76128 | 0.9375 | 1.04641 | 0.73674 |
| H1 | 关节角 | 0.98540 | 1.0000 | 0.98707 | 0.17474 |
| H8 | 根位置 | 0.93999 | 1.0000 | 0.91270 | 0.49628 |
| H8 | 关节角 | 0.95591 | 1.0000 | 1.03390 | 0.30681 |
| H32 | 根位置 | 0.98248 | 0.8125 | 0.99819 | 0.42950 |
| H32 | 关节角 | 0.95459 | 0.8750 | 0.87983 | 0.40192 |

证据降低了动作梯度链整体失效的嫌疑，但H32仍有错误方向及较大个别误差；它只覆盖R01访问状态的开环局部响应，不能证明闭环actor梯度、奖励目标或新摩擦下模型正确。暂保留H32，不凭该结果缩短想象时域或宣称策略改善。

服务器目录`data/supermimic_for_q1/J01_action_response_20261023/`，plan SHA `98acb6fd5c76752650987d75e8b1f38fcdc06f5c9095264769a8c3ee173b8eeb`，原始NPZ保持服务器LOCAL_ONLY。Git收据/计划为`data/manifests/supermimic_q1_J01_action_response_20261023.json`与`data/manifests/supermimic_q1_J01_plan_20261023.json`，执行日志在`logs/session_records/supermimic_q1_J01_*_20261023.log`。独立保存数据重算原生退出0、`SAVED_ACTION_RESPONSE_AUDIT_PASSED`：4基线精确、192项响应、掩码/计数/汇总一致，最大标量差2.22045e-16，0新模型或物理调用。小收据`data/manifests/supermimic_q1_J01_audit_20261023.json`；不复跑动力学。

## T01/B01执行前登记

新增`audit_action_response.py`独立从J01保存数组重算基线、有效掩码、响应与汇总，不导入原计分函数，不调用模型或动力学。新增`inspect_torque_limits.py`读取R01 step0/final六条既有轨迹，在每个控制边界之前的状态计算PD需求力矩，与原Q1Sim的实际`mj_forward`限幅actuator force逐关节核对；0积分/策略调用/optimizer。该统计不覆盖控制区间内的饱和，也不证明参考动作所需的真实力矩或硬件能力。

新增`benchmark_actor_batch.py`：R01全宽WM/actor/critic，H32/论文观测/critic bootstrap、原观测噪声与TF32；固定seed20261023，在保存走路轨迹抽128个有效起点，比较batch32/128，各一次预热、两次计时。仅actor前后向，WM/critic冻结，0物理/optimizer；前后state_dict内容SHA一致并检查有限非零actor梯度。容器30GiB/新OOM、GPU86°C、可用显存不足2GiB触发停止。该测试排除replay、WM/critic优化及Adam状态，只作为资源可行性证据，不作为训练提效结论。

三个新增入口运行前AST检查及两端源码SHA相同；实际结果见下文。入场资源快照cgroup usage27940626432/34359738368字节、OOM14、3090显存6348MiB/GPU0%/40°C，与P07同boot。huanyu部署根是源码展开目录而非Git工作副本；本地Q1 checkout负责提交/推送，服务器使用文件SHA对应版本。

## T01实际结果

原生退出0、`SAVED_BOUNDARY_TORQUE_AUDIT_COMPLETED`。六条既有轨迹共6974个控制边界，解析PD限幅与实际`mj_forward`actuator force最大差1.77636e-15Nm；0新积分/策略调用/optimizer。

| 最终策略片段 | 全关节边界样本饱和比例 | 腿部边界样本饱和比例 | 存在任一关节饱和的控制步比例 |
| --- | --- | --- | --- |
| 走路 | 0.08805% | 0.15133% | 1.33172% |
| 浅蹲 | 0.08643% | 0.13685% | 1.12360% |
| 出拳 | 0.18101% | 0.30728% | 3.09735% |

全部六条轨迹动作clip比例0；最终右踝pitch饱和比例分别1.0291%/0.8643%/2.9499%，为主要饱和关节。当前策略没有频繁在控制边界触碰力矩上限，因此不优先提高cap或PD增益。该低饱和结果来自主要站立的现有策略，不能证明行走有足够力矩或参考动态可行；区间内部峰值未测。下一轮仍保持PD50/2和25%cap，以免同时引入控制变量。

服务器收据`data/supermimic_for_q1/T01_boundary_torque_20261023.json`，Git小副本`data/manifests/supermimic_q1_T01_boundary_torque_20261023.json`，stdout `logs/session_records/supermimic_q1_T01_boundary_torque_20261023.log`。

## B01实际结果

原生退出0、`ACTOR_BATCH_RESOURCE_BENCHMARK_COMPLETED`，总38.193秒，6次全宽H32 actor前后向（每批量1次预热+2次计时），0物理/optimizer。三模块前后state_dict SHA逐一相同；各次actor梯度有限非零、WM/critic参数梯度均缺省。

| actor batch | 两次计时秒 | 均值秒 | 峰值allocated MiB | 峰值reserved MiB |
| --- | --- | --- | --- | --- |
| 32 | 3.73742 / 3.40788 | 3.57265 | 239.15 | 290 |
| 128 | 4.12161 / 3.73111 | 3.92636 | 526.35 | 628 |

四倍想象轨迹在这次小样本测量中，actor前后向耗时均值约增加9.90%；只证明当前3090可容纳该actor批量，不是完整训练加速比或效果结论。未测试batch4096，不能将128称为论文规模。计时顺序固定、试次数少、观测噪声重新采样；不根据loss数值差声称更好。未分配replay/Adam状态或执行WM/critic训练，完整运行仍有资源和稳定性门。

入/出容器内存28225482752/29885272064字节，OOM始终14，GPU40°C，未触发保护。服务器收据`data/supermimic_for_q1/B01_actor_batch_20261023.json`，小副本`data/manifests/supermimic_q1_B01_actor_batch_20261023.json`，stdout `logs/session_records/supermimic_q1_B01_actor_batch_20261023.log`。接下来只接入新接触协议入口并准备批量单变量对照；尚未R02训练。

## C01新入口与下一对照草案

`run_contact_corrected.py`显式将冻结train/evaluate驱动的资产构造器绑定到ContactCorrectedAssets，不修改R01核心或驱动文件。运行前核对P07收据资产与全部验证源码；评测额外拒绝旧接触协议checkpoint。新训练仍使用原采集/WM/BPTT/critic流程，manifest的source_files含该包装入口，training_entry指向实际调用的冻结驱动。`--check-only --preflight-output <新路径>`只检查绑定与身份，0物理/策略/optimizer；实际完整训练生命周期尚未测试。

下一对照草案`script/supermimic_for_q1/improvement_plan_S01.json`，状态PROPOSED_NOT_REGISTERED_OR_STARTED：两臂均从零初始化、修正摩擦、原16环境/3000轮=48000新交互、WM64×10、H32/k8、相同参考和seed；仅actor batch32与128不同。PD/奖励/学习率/观测不变。这样先得到修正接触的匹配基线，不能用“旧R01对修正+大batch”把两因素混成单变量因果结论。

每臂保留step0/final独立整片动力学评测与保存轨迹审计，主看腿部/全身RMSE、根跟踪/实际位移、完整率，不按总回报选赢家；出拳已用于开发，不能称未触碰测试集。0.15rad门不放宽。完整训练的replay/Adam/WM/critic峰值和稳定性仍待实际验证，新轮无自动恢复；两臂结果之后再登记独立长预算比较，当前不臆估训练耗时或同时增大WM batch。

C01实际：入口AST/计划JSON及两端SHA通过。新P07训练绑定预检原生退出0、`CONTACT_PROTOCOL_PREFLIGHT_PASSED`；传入旧P04收据的阴性控制原生退出1，准确报`Backend receipt is not for this corrected contact protocol`，未进入驱动或创建训练产物。预检与阴性控制均0物理/策略/optimizer，不等于新训练完整生命周期通过。新协议checkpoint尚不存在，evaluate路径仅实现，尚无其正例实际评测。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' $PY -u script/supermimic_for_q1/audit_action_response.py --input data/supermimic_for_q1/J01_action_response_20261023/receipt.json --output data/supermimic_for_q1/J01_action_response_20261023/independent_audit.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' $PY -u script/supermimic_for_q1/inspect_torque_limits.py --root "$PWD" --evaluations data/supermimic_for_q1/R01_eval_initial_20261023/receipt.json data/supermimic_for_q1/R01_eval_final_20261023/receipt.json --output data/supermimic_for_q1/T01_boundary_torque_20261023.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 $PY -u script/supermimic_for_q1/benchmark_actor_batch.py --root "$PWD" --evaluation data/supermimic_for_q1/R01_eval_final_20261023/receipt.json --output data/supermimic_for_q1/B01_actor_batch_20261023.json --device cuda:0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' $PY -u script/supermimic_for_q1/run_contact_corrected.py train --root "$PWD" --backend-receipt data/supermimic_for_q1/P07_backend_20261023/receipt.json --check-only --preflight-output data/supermimic_for_q1/C01_contact_entry_20261023.json
```

以上已运行，输出拒绝覆盖；恢复优先读保存的结果，不重复诊断。C01小收据`data/manifests/supermimic_q1_C01_contact_entry_20261023.json`，日志`logs/session_records/supermimic_q1_C01_{contact_entry,expected_rejection}_20261023.log`。

## 方案选择与保存状态

已确认的修复是接触优先级；已验证可行的下一变量是actor batch32→128。保留H32/k8、论文观测、学习率、奖励和PD/力矩设置。训练预算偏小是有依据的嫌疑，仍须匹配长短预算实测；原策略回报改善多数来自限位惩罚，不能以继续优化同一总回报代替移动/腿部效果验收。

R01冻结核心、train.py、evaluate.py与训练提交`cc3ecb61f8ab00cab112ed52a6c40147d30f5eb8`的Git差异为零。新增代码/计划、小收据和审核后日志由本次Q1阶段提交保存，大型工件及论文原件仍LOCAL_ONLY。同步流程为暂存审查、payload audit、提交、显式push origin Q1，再独立比较HEAD与远端refs/heads/Q1；完整提交号由Git历史及发布核验输出提供。本地报告镜像同步到Mac工作区。结束实核无`python.*script/supermimic_for_q1`作业，V01 HTTP返回200；未启动R02，未恢复A3。
