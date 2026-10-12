# YUANQI 当前状态与下一步

更新：2026-10-12（E14源标签结构7/7通过、A3配对0/7质量通过；优先定位转换，未训练人体分支或超过官方）。此文件用于恢复工作；只保留最新摘要，详细过程见阶段 README。旧机器/进程信息不是实时状态，执行前重新核实。

## 用户目标与固定约定

10/12用户确认CLOTHO为自主采集衣服模型训练数据，授权必要时找外部数据。E14于20:28:32完成、20:28:41独立验收VERIFIED_QUALITY_REJECTION：6CPU测试通过，7序列13,526源帧内部FK/骨长/米尺度/根速度全部通过（FK max≤5.72e-7m）；2100新IK帧限位/速度通过，但位置P95 0.403–0.661m，6/7脚穿地、6/7支撑滑动失败，合格pair0。0optimizer/0策略回放/无CUDA，6健康样本同boot/CPU39/GPU26°C/OOM0，任务已结束；原机只复制14个标签（7.81MB），Mac无原始数据。旧TEST_ONLY不改，时间沿用30Hz工程合同、现场时钟未验收。优先拆校准/逐链几何，七段现已用于质量诊断，不能充当未见最终测试；恢复见[E14结果](README_A3_SMPL_CORPUS_20261012.md)与[清单](../data/manifests/a3_zhy_smpl_corpus_20261012_E14.json)。

从官方权重取得可验证提升一直是目标。10/12已实现自制SMPL 10×90→64学生分支与冻结官方035 teacher/FSQ/decoder桥；11CPU契约及ZHY E13的128真实窗口/16旧官方动作独立核验通过，q/v ABI与人体参考重构误差0、桥动作最大差2.86e-6、官方及学生权重不变。E13于19:35:59结束、19:37:43独立验收为VERIFIED_INTERFACE_ONLY，0真实optimizer/0新回放/无CUDA初始化；现有四段仍TEST_ONLY，训练门全部拒绝、合格训练会话0，学生PT明确未训练/不可部署。5健康样本同boot/CPU40/GPU26°C/OOM0，阶段间最大103.68秒空档不算连续覆盖。CLOTHO原始标签206序列/17会话可作下一数据候选，尚缺来源范围、合格重定向及会话划分；原机停载，不重置旧实验。恢复见[SMPL实现与E13](README_A3_SMPL_ENCODER_20261012.md)和[E13清单](../data/manifests/a3_zhy_smpl_20261012_E13.json)，接口通过不代表已具备SMPL跟随能力或超过官方。

E12已VERIFIED：3项CPU和6条件/384固定帧前向及独立CPU复算通过，0optimizer/0新回放。更新2原始动作mean RMSE0.03803；encoder-only0.01931并改变378/384帧token，decoder-only0.02780且token精确不变，std-only确定性mean不变。两部分均独立产生漂移，不能只归因量化或一个模块，更不能当作闭环退化贡献百分比。6健康样本同boot、CPU41/GPU31°C、OOM0，计算已结束；下一项优先官方行为约束/冻结的单变量对照，仍须明确真实目标及保留已有能力，见[模块结果](README_A3_ZHY_MODULES_20261012.md)。原机停载，官方仍基线。

10/12 E11已VERIFIED：7项CPU、真实四批FP32零更新/独立CPU损失概率核验通过，0optimizer与梯度前后权重SHA一致。PPO主梯度大于aux，value不直接连到Actor，Critic三批解释度正/一批负，归一化buffer变化单列。R12保存点0/4/8optimizer，59个参数张量跨encoder/decoder/Critic/std改变；实际LR2e-6～4.5e-6（已更正旧概括）。16健康样本同boot、CPU40/GPU34°C、OOM0，任务结束GPU空闲。用户此前问题已对应证据表；下一项固定同批观测隔离encoder/decoder与token变化，见[排查结果](README_A3_ZHY_GRADIENT_20261012.md)。原机不计算，不重置旧实验或自动长训。

E10/R12训练17:07:03–17:08:06、4完整回放17:15:20–17:17:20均独立核验；8optimizer/3保存点/官方辅助loss及实际精度通过，5775策略步/0跌倒，但四诊断动作等权平均关节误差0.057795→0.060468rad（+4.63%），3/4退化，效果门槛失败，不升级R12。复用E02同新机同boot官方基线，未重跑20基线。39健康样本同boot、CPU46/GPU35°C、OOM0，无新关键内核错误；最大441秒间隔为阶段间空档，任务已结束/GPU无计算进程。原机不计算、无恢复/长训；下一项新FP32入口损失/梯度与模块更新诊断，先明确要改善的真实缺陷，见[最终结果](README_A3_ZHY_FP32_20261012.md)。

E09于16:40:41–16:41:38完成，8项CPU回归与真实四batch概率验收通过、0optimizer；显式algo.trl.bf16/fp16/tf32=false及ACCELERATE_MIXED_PRECISION=no，Actor31次调用均FP32，首batch mean RMSE4.328e-7、四批ratio约0.999987–1.000017，原门槛不变。E08主体权重/输入稳定但真实BF16；本环境TRL默认BF16经Transformers覆盖启动no，已定位并修正这次概率偏差，后续E10证明仍有策略退化，见[修正结果](README_A3_ZHY_CONSISTENCY_20261012.md)。

10/12用户再次要求持续写README并思考反复排查背后的问题。工作理解是建立能自主判断失败与改进的完整实验链；尚未当作用户确认的选题。真实遥操中哪项缺陷值得微调、验收指标是什么仍需证据，见[实验依据与问题](README_A3_EXPERIMENT_DECISIONS_20261012.md)。

10/12只读核验官方A3说明/issue/曲线和版本：有训练不收敛讨论及语料/warm-start限制，未找到零更新偏差直接说明；main `c9df753`只新增下载元数据，无训练修复，未更新vendor。后续E08/E09在本环境定位精度覆盖并修正，不扩大为官方所有环境均有同一问题，见[来源核验](README_A3_UPSTREAM_LIKELIHOOD_20261012.md)。

E04–E07历史诊断保留：E05实际4批ratio约0.53–1.94；E06/E07离线FP32回放复现采样，但未重建真实BF16精度。旧失败账本不改写，后续E08/E09更正此前完整精度推断并修复计算不一致，见[E07更正](README_A3_ZHY_WRAPPER_20261012.md)与[E05更正](README_A3_ZHY_PRECISION_20261012.md)。

10/12最新授权：迁移及R11完成后，在新服务器ZHY继续A3探索与微调。E02 64完整回放+3冒烟核验通过、0跌倒，adapter/R11全部效果门槛失败，官方仍为基线。E03采集器TensorDict错误已保留，修正后4项CPU回归通过；E04有效完成0更新诊断，采样输入/mean与存储精确一致、12概率行CPU复算通过，实际首batch ratio0.52963–1.37392，见[诊断](README_A3_ZHY_LIKELIHOOD_20261012.md)。尚未定位具体层或历史退化唯一根因。原机不计算、旧账本保留，见[E02结果](README_A3_ZHY_SEARCH_20261012.md)。

- 10/12 记录问题：既有驱动 README 的一处凭据说明已从当前文件删除，Git 历史仍保留旧内容；对应主机密码轮换待用户，见[脱敏记录](README_RECORD_REDACTION_20261012.md)。只通知一次，不尝试验证密码或运行工程计算。

- 10/12 教学：本人答对PD位置误差0.25-0.20=0.05rad；静止时也计算控制、PD输出力矩及速度阻尼已讲解，理解仍需核验。下一项“位置已到但仍在转动”判断待回答；学习任务未运行工程计算，见[当天学习记录](README_LEARNING_SESSION_20261012.md)与[任务卡](README_LEARNING_AGENT_TASK_20261011.md)。

10/12：ZHY A3数据8440文件/21.07GiB、运行环境173862文件/22.35GB与补充4147文件均SHA通过，路径修正/CPU导入/9项保护契约及16个motion入口通过。用户授权的新机R11于11:49:54–11:51:26完成：官方035初始化、16env/epochs1/LR尺度0.1/2更新，3个checkpoint独立SHA/CPU重载/有限性、真实optimizer8步/辅助loss/初始权重与计数核验通过；18样本CPU最高40/GPU35°C、功率采样最高131.09W、RSS4.83GiB/可用最低8.03GiB，同boot/0OOM/无新关键内核错误。新机28 CPU等效/16GiB、RTX3090/570.153.02，GPU已空闲；原机A3服务inactive/disabled，旧账本保留，不自动长训。仅短训练完整性通过，效果/长期稳定/旧机具体根因未验收。恢复见[阶段记录](README_ZHY_RUNTIME_20261012.md)、[验收清单](../data/manifests/zhy_runtime_20261012.json)与[实验清单](../data/manifests/a3_zhy_20261012_E01.json)。

10/11最新：用户授权先尝试可改的软件层。已将官方 `nvidia-driver-595-open` 595.84 切换为 apt 的 `nvidia-driver-580` 580.178.04；DKMS 模块在6.8.0-138上安装成功，计划内重启后新 boot `560a7f8b-e70c-45a7-b3d7-5388308339cb`，GPU 29°C/23.28W/0%利用率，A3自动服务仍 inactive/disabled，无计算进程。尚未在新驱动上复测负载，见[驱动切换记录](README_DRIVER_SWITCH_20261011.md)。

10/11最新故障证据：580.178.04 下 D01 普通 CUDA 冒烟完成导入、初始化、1024 矩阵分配/乘法/同步，持久化日志停在 `CHECK_BEGIN`；随后确认 boot `fbae4671-acc0-470e-8e39-60d8340b5da2`，状态 `STOPPED_REBOOT`，不重试。用户现场 BIOS 同时提示“上次开机检测到系统电源发生异常供电现象，已启动电源保护机制”。前一 boot 内核只读记录未见 Xid/MCE/AER/panic；当前首要排查对象是 PSU/主板供电/显卡辅助供电线/PCIe 插接与供电路径。全部 GPU/A3/YOLO 负载保持停止，详见[驱动切换记录](README_DRIVER_SWITCH_20261011.md)。

10/11最新：YOLO普通训练排查 E01 已执行并确认新boot：Ultralytics 8.4.175 的 YOLOv10n 架构、1800/200合成数据、batch=32 只写到 TRAIN_BEGIN，0个epoch；重启前CPU40–47°C、GPU30–33°C、利用率采样0%，不属于已观测热保护。状态STOPPED_REBOOT，未重试、不恢复A3；普通YOLO初始化也能触发主机重启。见[YOLO阶段记录](README_YOLO_PROBE_20261011.md)。

10/11最新：E30已用既有twist(torch1.13.1+cu117/CUDA11.7)完成同一单次1024矩阵与60秒观察，18:29:30–18:30:35，同boot34cc7c88，56阶段记录无异常，CPU49/GPU33°C，实时kernel follow可读。E31随后以同环境30秒循环运行，UTC10:36:19.326968完成MATMUL_SYNCHRONIZE后在ALLCLOSE BEGIN处失去持久化完成记录；UTC10:38:00.216198确认新boot c99ddfa9，状态STOPPED_REBOOT。重启前仅2个健康样本（CPU最高48/GPU30°C），没有可用旧boot内核错误记录；重启后只读快照，无GPU计算。E28（cu128）和E31（cu117）都出现“单次通过、循环重启”，但具体API/驱动/硬件根因仍UNKNOWN。随后只读检查未见Xid/MCE/AER/panic，GPU Recovery Action为None、PCIe replay为0；当前PCIe协商Gen1是待硬件核对线索，不能单独视为根因。E31首故障后不再增加GPU负载，不启动E32，不恢复A3。见[阶段记录](README_GPU_RUNTIME_COMPARE_20261011.md)。

10/11最新：E27关闭blocking单次及60秒观察通过；E28于18:01:04启动30秒循环计划，前2轮完整，第3轮matmul及同步/allclose通过后18:01:05.596最后VALUE=true，下一同步BEGIN未保存；18:02:45确认新boot34cc7c88，STOPPED_REBOOT，全链停止/E29未登记启动。94服务器阶段日志，Mac镜像仅8/94阶段覆盖，不能按旧镜像尾认定初始化失败。仅2健康采样CPU最高46/GPU30°C，无故障附近内核覆盖，根因UNKNOWN。再次在无A3/Isaac/RL的普通GPU计算中观察重启；不复跑/恢复训练。只读METADATA发现既有twist环境torch1.13.1+cu117可作下一轮候选，当前2.8.0+cu128；尚未导入/验证旧环境，先改善崩溃证据再同程序对照。unit无残留GPU计算、旧A3自动恢复disabled，heartbeat不启用。见[最终矩阵与停止证据](README_GPU_OPERATOR_COMPARE_20261011.md)。

- 10/9 用户确认六项能力正式纳入学习主线，见[最新教学入口](README_LEARNING_MAINLINE_20261009.md)。教学从关节控制链与必要 Python 开始，结合样本 / 训练代码逐步补坐标时间、RL、诊断与实验；每阶段亲自检查并总结，尚未宣称掌握。
- 10/9 最新优先级：10/30 前确定 A3 或 Q1 遥操模型的训练方法、资源需求与本人能力要求；先形成有核验依据的路线。当前以[选型方案](README_MODEL_ROUTE_20261009.md)为准，不把旧完整人形训练 OKR 当成本次硬性截止，团队正式 OKR 尚未修改。
- 长期目标：衣服与头环自然遥操 X2 / A3 / Q1，并形成自主策略训练能力和科研课题。
- 近期目标：独立完成一轮机器人策略训练，并逐步形成可验证的论文工作。10/2 用户明确不必以底层算法创新为目标；教学改为实践先行、数学按需，见[最新安排](README_LEARNING_ROUTE_20261002.md)，仍按[训练工作包](训练工作包.md)从 P0/P1 开始。
- 训练位置：`ssh hp3090`，`/media/yu/FAFF-E9771/YUANQI`。
- 10/10 A3专用位置：`/media/yu/FAFF-E9771/YUANQI_A3`；通用研究/教学与历史记录保留YUANQI，Q1位置不变。A3新副本同仓库main，后续A3数据/模型/评测写新目录。
- 目录：`data/` 产生的数据，`logs/` 纯执行记录，`script/` 代码，`reports/` 重要 README。
- 关键记录与代码：`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`；大模型和关键大型数据：百度网盘。
- 每完成重要任务立即写阶段 README，更新此文件和[报告索引](README.md)，提交并同步 GitHub。规范见 [AGENTS.md](../AGENTS.md)。
- 教学需定期阶段总结；默认每 2–3 个关键概念或遇到理解卡点回顾一次，按反馈调整，区分已讲解与已理解。

## 已完成，可从证据继续

10/11当前：E24普通GPU也确认重启，STOPPED_REBOOT，全链停止。256矩阵30/120/600三档通过，冷却后1024档15:39:53.797最后MATRIX_FIRST_OPERATION_BEGIN，无首操作完成；15:42:19确认新boot c7805e85，开机推算约15:40:20（非故障精确时刻）。221样本CPU最高50/GPU46°C、末次44/34°C，无保存热停；无A3/Isaac/RL导入，重启无需A3专有链路即可发生，根因UNKNOWN。3通过/1中断/11未做；unit无残留GPU进程，旧自动恢复disabled、hp3090/yuanqi-a3 PAUSED、日研保留。见[最后区间与证据](README_A3_ISOLATION_20261011.md)。用户要求的[A3细日志版本](README_A3_PHASE_TRACE_20261011.md)已实现、5无CUDA契约/目标静态核验通过，但E25未登记/启动；不绕过E24失败继续计算。

10/11最新结果：用户授权“继续A3训练，若重启就停”已执行。E22/R10于14:51:33启动官方035保守2更新试跑，第一更新4optimizer步及step0/1大小SHA证据存在，无step2；14:53:09确认新boot740972e7，开机时间推算约14:52:29（非故障精确时刻）。仅5健康采样，CPU最高60/GPU34摄氏度，无保存的热保护事件；缺失间隔内峰值未排除。E22已STOPPED_REBOOT、详细state blocked，无训练进程，不再加载模型，旧自动恢复disabled/两个heartbeatPAUSED，日研保留。优先怀疑A3运行链路关联，尚不能定位代码/Isaac/CUDA或硬件根因；普通GPU/网络对照未做。后续计算需用户新指示，先分离执行环节，不继续整链盲试。见[本轮完整记录](README_A3_REBOOT_TRIAL_20261011.md)及[证据清单](../data/manifests/a3_reboot_trial_result_20261011_E22.json)。

10/11最新结果：HOST_LOADCHECK_E01在14:25:16因CPU温度保护STOPPED，四worker600秒档运行40.377秒后连续两次采样90摄氏度，GPU28摄氏度/空闲；最终909采样、14通过、1保护停止、6内存项未运行，15份stdout SHA及注册执行器SHA核验。全部同boot、无重启或关键内核事件；失败退出来自保护终止，未认定算术错误。14:27停载后CPU55摄氏度。unit failed/Restart=no且无残留计算；内存/GPU/小CNN/YOLO搁置，不重复恢复E01、不自动升档。hp3090 heartbeat已PAUSED，原A3训练/评测及yuanqi-a3保持暂停，每日研究保留。下一步用户检查CPU散热运行/安装情况后再决定独立新实验；历史重启根因仍UNKNOWN。见[最终测试矩阵](README_HOST_LOADCHECK_20261011.md)及[停止证据清单](../data/manifests/host_loadcheck_stop_20261011_E01.json)。

10/11最新结果：北京时间10:44:35–12:44:35固定两小时空闲观察完成，122采样、同boot、0boot转换、无计算进程/采样错误，CPU37–49°C/GPU30–31°C，可用RAM29.55–29.62GiB。监测服务停止并disabled，A3 heartbeat到期暂停，训练自动启动入口仍disabled；结束后不恢复计算，每日研究保留。只证明该窗口未观察到重启，负载相关性仍是假设、根因未定位。E21两组各200/44保存点核验通过；GPU闭环3动作完成、050中断，效果仍UNKNOWN。见[最终观察记录](README_A3_IDLE_WATCH_20261011.md)，以下恢复建议暂停，继续计算须用户新指示。

10/11恢复结果：E20 GPU分支搁置；E21完成两组各200更新，44保存点/导出、21,878标签窗口及96个独立重算标签核验通过。官方policy更新0、旧E10 blocked保持，效果未完成验收；详细证据见[当天恢复记录](README_A3_RECOVERY_20261011.md)。

10/10最新执行E20：同16训练动作/两档噪声/seed/零初始化CV残差适配器，比较角度MSE与冻结teacher量化前latent目标，各2→200监督更新。官方policy更新0、R10未分配、E10 blocked保持；四重复诊断动作不参与训练、不当作最终验收，先独立checkpoint/export/标签核验再闭环。门槛与恢复见[独立阶段](README_A3_CONTROLLER_ADAPTER_20261010.md)，当前预登记，未声称改善。

10/10 A3 E16-E19完成60完整回放均无跌倒，但预测器未过效果门槛；E17独立44指标复算，E18完整23,100实际策略参考因果核验，参考年龄0-40ms（官方离线初速度边界保留，非端到端实时证据）。E19在相同机器人状态800动作重放差0，干净E12位置/速度更准却编码/动作偏差更大，部署round使动作损失输入梯度0、量化前latent梯度有效。24独立CPU契约通过；直接PPO长训基础就绪但概率/改善/变量/预算未满足。下一E20同架构/数据/预算对照角度与teacher latent目标，仅训练参考残差适配器；官方controller冻结、旧E10 blocked保持、R10未分配。见[闭环与长训准备](README_A3_CLOSED_LOOP_20261010.md)。

10/10 A3方向筛选E07-E15完成当前有限诊断：E08零更新梯度和E09离线概率通过限定检查，初始ratio差异未归因；E10两次初始化重启无保存进展blocked，保留后继续CPU。E12仅相对历史的未来关节预测在4重复诊断验证动作均优于CV，整体-17.14%，但E13含噪失败；E14噪声感知/E15在std0.005/0.01人工噪声×3seed均逐动作优于CV、整体约-41%/-60%，干净单腿及简单保持基线局部仍退化，全部条件门槛失败。官方A3policy本阶段0更新，未证明策略或真机提升。3个模型各20动作独立插值/CPU误差/训练归一化验证、10契约通过，模型LOCAL_ONLY；boot f6e86267，作业结束。优先候选是冻结官方低层，优化因果参考及不确定性；下一需补完整root/朝向/可靠新会话与闭环，详见[方法、测试矩阵与待办](README_A3_DIRECTION_20261010.md)。

10/10 A3 E05/E06筛选完成：epochs1关节RMSE0.07730；再LR尺度0.1后0.06024rad，接近但仍高于官方0.05906（+2.00%），4验证0.06783也更差，根/腕/腿整体仍退化。两轮各2更新、8真实optimizer步，step0/1/2独立CPU/有限性/计数/实际LR/aux核验，40新完整回放、0跌倒，6新契约与190工件/代码/日志SHA引用通过。R09有4动作关节略改善、其余16更差，右单腿站立为事后局部候选，非独立泛化证据。选择工具保留官方；低步长短更新→全指标门槛→回退流程可复用，正向提升尚未验证。所有训练/评测inactive/disabled、同boot、GPU空闲，LOCAL_ONLY。下一项同批rollout梯度/优势方向诊断，见[结果与长期流程](README_A3_SEARCH_20261010.md)。

10/10 A3 E04/R07阶段完成：修正Accelerate钩子兼容，10 CPU契约、官方初始化2更新、step0/1/2独立模型/优化器有限性与40实际step/aux目标核验通过。step1/2各完整selected20无跌倒；关节RMSE官方0.05906、旧step2 0.09153、新step1 0.07991、新step2 0.08314rad。新step2较旧误差低9.16%但比官方高40.77%，20/20动作关节误差均高于官方，留出同样退化，尚无正向提升。首轮回合长度空统计NaN单列。训练/评测inactive/disabled、同boot、GPU空闲；E03/R06失败账本保留，LOCAL_ONLY。下一变量建议epochs5→1独立2更新，本轮未执行；见[完整结果](README_A3_CORRECTED_20261010.md)。

10/10 A3优化器诊断E02完成：官方trainer为AuxLoss PPO，R05包装器继承基础PPO绕过辅助目标，哨兵梯度检查确认；constant scheduler与KL双重控制及恢复args LR变更均CPU复现。新增显式 `script.a3.verified_finetune.VerifiedFineTuneTrainer` 接回官方aux loss/stats，以KL唯一控LR并记录每个实际step，9项CPU契约通过。旧job/service/vendor/R05与E01不改，没有启动新训练。修正尚未证明策略改善，下一项为完整Isaac入口集成和独立短试验对照；见[最终诊断与证据](README_A3_OPTIMIZER_20261010.md)。

10/10 A3退化定位E01完成：官方及0/2/10/20/40/80/120/200各20动作完整核验，共180回放。step0与官方完全一致；step2无跌倒但关节RMSE0.05906→0.09153rad，最早已观测退化在0→2；首次跌倒在2→10，20/40再次无跌倒。所有更新阶段共同跌倒前与留出总体误差仍高于官方；没有验证到改善方向。学习率日志/保存状态冲突、固定输入两部件变化仅为线索。模型最终CPU/有限性与产物SHA核验通过，诊断服务inactive/disabled，无新增训练，旧作业仍暂停，LOCAL_ONLY。下一轮先验证实际LR控制，再讨论单变量短试验；入口[最终诊断](README_A3_REGRESSION_20261010.md)。

10/10 11:43 A3官方全链路当前可执行工作结束：24任务18通过（按结构/链路范围）、2失败、4搁置，无待做/运行中；65项检查通过。官方20动作0跌倒，R05完成200且独立CPU/优化器有限性通过，但微调5/20跌倒、4留出1/4，效果退化。4动捕离线3/4跌倒、缓冲2/4，仅诊断；官方20动作因果缓冲仍0跌倒，37694窗口未读未来源帧、实际年龄200–240ms，非现场延迟。官方/微调ONNX各100真实输入通过，LOCAL_ONLY。服务均退出/disable，旧长训与巡检继续暂停，每日研究保留；Isaac指标、RKNN、动捕质量等待后续处理。见[完整测试矩阵/用户待办](README_A3_OFFICIAL_TEST_20261010.md)与[SHA清单](../data/manifests/a3_official_stage_20261010.json)。

10/10 A3目录划分与官方包导入完成：`YUANQI_A3` 独立Git工作副本和四目录已建立，vendor/Python环境只读复用，历史全链路仅data/references引用。PT 402050931字节及4个匹配小文件均核对官方SHA；CPU重载、网络/优化器有限性通过，bundle_integrity_verified=true、LOCAL_ONLY。4文件来自用户Mac Downloads并SCP传入，配置缺失卡点解除；未启动策略/训练。原PT与旧账本保留；目录创建时boot变化只登记恢复。入口见[目录与完整包记录](README_A3_WORKSPACE_20261010.md)，下方模型不可得为历史状态。

10/10 02:31 A3全链路独立可执行项已完成：22项台账为11通过/1质量失败/10搁置，无待做/运行中。56项项目测试及6项官方RKNN契约测试通过；四段SMPL结构/运动学、26CSV、24PKL、16/4整动作划分及输入preflight通过，均保留限定范围。参考TEST_ONLY、IK/脚穿地质量失败；因果输入年龄180–220ms，非现场/策略效果。旧2950仅100步诊断，1.88秒跌倒；其ONNX 100真实输入误差8.35e-7通过。官方PT下载超时，正式基线/2→200微调实际0更新；Isaac指标与RKNN工具链缺失。R03仍inactive/disabled、GPU空闲，同boot。最新任务/148项产物SHA及迁移待办见[Git清单](../data/manifests/a3_fullchain_20261010.json)，详情/恢复顺序见[执行入口](README_A3_FULLCHAIN_20261010.md)。

10/10主线调整：用户赞成“官方预训练PT + 动捕服数据适配与微调”。先验证官方样例链路，再测转换后的衣服录制，以短板决定微调；会话/动作序列隔离验证，未来参考先建议缓冲并计延迟。Q1目录已有FGP导出的SMPL测试样本，优先复用并验收根轨迹/接触/坐标时序，不重复索取已有文件；现场协议仍待核验，官方PT未下载/评测。实时服务inactive、MainPID0、disabled，继续暂停；下一步为已有样本质量与数据合同、官方基线准备，见[动捕适配路线](README_A3_MOCAP_ROUTE_20261010.md)。

10/10最新A3：用户要求先停并核对官方HF模型；R03已01:09停止，service inactive/dead/disabled、无GPU计算进程，巡检yuanqi-a3 PAUSED，每日研究yuanqi ACTIVE。最近保存2950独立大小/SHA/CPU重载/有限性校验通过，LOCAL_ONLY。期间01:03出现新boot40497006，根因未知；账本running字段为历史持久值。官方035 step200000有PT/ONNX/RKNN，RKNN用于板端推理，PT支持评测/权重微调；未下载/评测，不含SMPL编码器。最新入口[官方模型与暂停](README_A3_PRETRAINED_20261010.md)，下方运行记录为历史证据。

10/10 01:00:52 R03首次定时巡检已实际触发：service active/running/enabled、同boot/run、R03范围与CPUQuota100%核验，最新持久sidecar2900/大小匹配，本轮不重复哈希/重载；CPU73°C/GPU42°C，无新退出/热停，尚未到3000通知节点。日志 `logs/a3_watch_20261010/scheduled_20261009T165952Z/`，详情见[持续续训README](README_A3_CONTINUOUS_20261010.md)。下方为启动时独立模型验证证据。

10/10 00:53:50最新A3：用户授权重启并取消3次启动上限；R03已00:52从2700完整恢复，新2725模型独立大小/SHA/CPU重载/有限性校验通过，LOCAL_ONLY。`yuanqi-a3-longtrain.service` active/running/enabled、CPUQuota100%，run `E008_cont_01_s2700`；CPU66°C/GPU37°C、同boot。无累计启动上限/到期时间，温度、连续2次无进度及异常停止保护保留；39项测试通过。原heartbeat ACTIVE/每10分钟已更新R03，到10000后停止并暂停巡检。R01/R02账本不修改，重启根因仍UNKNOWN；见[持续续训README](README_A3_CONTINUOUS_20261010.md)，下方R02停训为历史状态。

10/9 20:28最新A3：R02反复整机重启后在18:40:13因3/3启动预算用完停止，服务failed/exit2、总账本blocked/attempt_limit、GPU无计算进程；当前boot `a95a89b9…`。2700模型独立大小/SHA/CPU重载/网络和优化器有限性校验通过，LOCAL_ONLY，距10000尚差7300。三个run CPU采样最高76/67/67°C、GPU44/35/37°C，无temperature_stop证据，重启根因UNKNOWN；不能认定电源故障。本轮只诊断与记录，未重启训练或重置预算。最新入口见[长训故障与恢复](README_A3_LONGTRAIN_20261009.md)及[故障清单](../data/manifests/a3_R02_stop_20261009.json)。下面17:50运行状态为历史证据。

10/9 17:50最新A3：用户授权测试失败后续训至累计10000；新R02已17:47从2000完整恢复，独立校验2050通过，`yuanqi-a3-longtrain.service` active/running/enabled、CPUQuota100%，第1/3次run `E007_long_01_s2000`。CPU69°C/GPU40°C、同boot，无新热停/重启证据。R02有效至10/10 17:47:35，温度CPU90/GPU85保护保持，普通异常/热停不自动解锁。原heartbeat yuanqi-a3已恢复ACTIVE/每10分钟、范围更新R02；R01complete及其3/3账本保留。见[累计10000续训](README_A3_LONGTRAIN_20261009.md)和[启动证据](../data/manifests/a3_longtrain_R02_20261009.json)。

续训依据MuJoCo E01的20/20跌倒结果。Isaac E02输出路径冲突，E03加载模型和20动作后缺smpl_sim指标依赖，官方validate因metrics缺失返回1；原生效果UNKNOWN，不能采用退出0/manifest complete作为通过依据。包装器已补强制validate。35项恢复/保存检查通过；模型LOCAL_ONLY，更新数不证明策略效果。下面R01完成与旧巡检暂停为历史证据。

10/9 15:37后完成A3 E01 MuJoCo sim2sim：相同20个训练动作、完整长度，各模型24304策略步；step500和step2000均20/20触发跌倒，首次跌倒时间中位1.90→2.40秒，前向慢走最终1.96秒即触发。两进程退出0、数值/覆盖核验通过，策略稳定跟随未通过，不可据此部署。CPU保护采样最高82°C，无新热停/重启。见[模型效果评测](README_A3_EVALUATION_20261009.md)和[逐动作摘要](../data/manifests/a3_mujoco_E01_20261009.json)。下一步训练仿真Isaac同条件评测以区分策略失败与sim2sim差异，尚未做该评测。

10/9 15:22 最新A3：第三次 `E006_auto_03_s1800` 已于15:20:33正常完成1800→2000，ledger complete、service inactive/success/exit0，同boot `91918243…`。最终2000模型独立SHA/CPU重载/有限性通过，13项训练标量有限；模型LOCAL_ONLY。CPUQuota100%期间100条保护采样CPU最高71°C，独立巡检曾73°C，没有再次触发保护；不能认定长期稳定、故障根因或策略效果通过。见[最终续训记录](README_A3_CPU_RESUME_20261009.md)。最终记录已保存；yuanqi-a3已暂停且回读PAUSED，每日研究yuanqi仍ACTIVE；3/3预算与原保护保留，不自动开新训练。

E001 / E002 / E005 的中断与 E003 / E004 续训历史保留在[续训 README](README_A3_RESUME_20261009.md)。当前任务为从零学习 / 对照，不是成熟预训练策略；累计2000更新完成，固定条件效果未验收。后续工作前重新检查service、ledger与最新有效模型。Q1副本未修改。

10/9 选型资料核验：A3 公开 SONIC 工程有随机初始化 / PT 权重起点训练说明，但缺 SMPL encoder，使用 0–180 ms 未来参考窗口；建议优先审计 A3 精确资产 / 接口兼容，Q1 小任务训练备选，尚未最终选型或复现。RTX 3090 24GB 可见，不据此保证人形训练可运行。通用 main 已从干净 `a82be9b` 快进至 `d086898` 再更新记录；Q1 并行工作没有修改，见[本轮 README](README_MODEL_ROUTE_20261009.md)。

Q1 后续迁移、PD / 重定向与输入测试在专用 `/media/yu/FAFF-E977/YuanQi_Q1`、`Q1` 分支推进，本轮只读核验。下面 Q1 描述是 main 保留的历史阶段，不表示平台当前仍未开始仿真；恢复 Q1 应先读专用副本的当前状态。

Q1新工作：用户于10/8指定 `/media/yu/FAFF-E977/YuanQi_Q1`，最终目标为SMPL全身跟随（腿部和移动），动捕具体格式暂不清楚。已阅读核心资料、确认22DOF资产与底层关节接口；现有`SetMcMotion`明确不支持外部模型/轨迹，不能当成实时SMPL入口。文档与模型限位/力矩冲突、真机IMU接口缺失待澄清。专用服务器工作副本已clone，资源迁移因连接反复中断尚未完成；仿真/训练未开始，见[Q1阶段README](README_Q1_SMPL_20261008.md)。原始Mac附件保留；旧研究记录不能据本轮SSH成功而视为已同步。

1. 学习路线、第一课、训练工作包、实验模板与 EgoLocate/头环静态分析已保存；10/2 新完成官方倒立摆 PPO 基础仿真训练/保存/重载，见[第二课实验](README_LESSON02_CARTPOLE_20261002.md)。尚未完成 X2/人形策略训练。
2. hp3090 上四目录项目已建好，确认 RTX 3090 24GB 与可写 ext4 磁盘；初始化前可用空间约 634 GB，后续使用前重新检查。
3. 初始化提交 `c7b64a7515b551a63830d0e040a203ba543346e5` 已推送 `origin/main`，该阶段结束时本地/远端一致且工作区干净。见[初始化记录](README_SETUP_20260929.md)。
4. 17 份参考节选已同步到项目 `data/references/` 并核验 SHA-256；原始附件仍在 Mac，未复制全部 ZIP。见[清单](../data/manifests/reference_excerpts_20260929.json)。
5. 已加入工件登记、Git 暂存检查和只读环境盘点脚本；见 [script README](../script/README.md)。
6. 本轮把重要阶段落盘要求写入项目规则，建立当前状态和阶段模板。见[记录规范阶段 README](README_RECORDING_POLICY_20260929.md)。
7. 每日研究最新为 10/12 [静默检查](README_RESEARCH_CHECK_20261012.md)：三类公告与已查版本/代码未确认实质变化；HF 当前资产未核验。SSH 已恢复，以服务器最新入口补同步 10/11 与本次研究记录；[发布回执](../data/manifests/research_watch_sync_20261012.json)核到 `d1fd2a8` 与远端 main 一致。最后有内容的[简报为 10/10](research/2026-10-10/README.md)；没有运行计算或新增假设。
8. 10/2 Agent 基线后，用户已按说明完成 `lesson02_my_first_run`；新目录、训练与重载完成标记及 23 个产物已核验。1024 并行环境、150 次采集/更新循环，平均维持 0.6446→4.9833 秒，达到时限比例 0→100%，重载动作探针误差 0。两次使用相同种子，不算多种子结果；操作已完成，代码理解与自主设计仍待确认。见[第二课跟进](README_LESSON02_CARTPOLE_20261002.md)及[复跑清单](../data/manifests/lesson02_user_run_20261002.json)。
9. 基础 Isaac Lab 仿真及 RSL-RL 训练/保存/重载已通过；人形 X2 的 P0 未完成。原有训练 sandbox 的资产路径/配置与未提交改动仍待处理，见[静态核验](README_P0_INSPECTION_20260930.md)；本轮未修改该工程。
10. 已讲解单步倾斜惩罚，入口新增 `--pole-angle-weight`（默认 1），准备权重 3 的单变量对照；参数与 CPU 数值检查通过，未启动新训练/重载。新版驱动增加权重记录与重载核对，旧实验精确重载继续用保存的驱动快照。见[奖励项与对照准备](README_LESSON02_REWARD_20261002.md)。

## 尚未验证 / 尚未完成

- PyTorch/CUDA、Isaac Sim、官方倒立摆资产、PPO 训练与 checkpoint 重载已验证；X2 精确资产/配置、X2 训练、模型导出与 sim-to-sim 尚未完成验证。
- 百度网盘官方 Linux 客户端 8.7.0 已发现，入口 `/opt/baidunetdisk/baidunetdisk`；登录、上传/下载未验证，未上传模型。记录状态仍是 `LOCAL_ONLY`；如现有入口不合适，再与用户讨论替代方案。
- 历史 X2 支撑站立/A3 上肢里程碑不能当作当前完整遥操验收；当前工作没有启动机器人控制。
- EgoLocate 的 Stage-A 推理实现已找到，所引用完整外部训练项目与 `best.pth` 不在已检查的 ZIP 中。

## 下一项具体工作

后续A3在 `YUANQI_A3` 执行。先读[方向筛选E07-E15](README_A3_DIRECTION_20261010.md)，不重跑已完成诊断、不解除E10 blocked或恢复旧长训。本轮局部收益在未来参考模块，官方policy仍为基线；下一阶段把关节候选扩展为完整因果参考（root/旋转/速度），加入因果平滑+CV/保持对照，用可靠新会话和相同到达时序比较oracle/已验证缓冲/预测，独立报告噪声、参考年龄、跌倒和身体误差。新会话时钟/根轨迹/地面对照和现有动捕质量仍需用户处理；Isaac指标/RKNN/现场接口继续见[待办](README_A3_OFFICIAL_TEST_20261010.md)。参考约束/少参数策略微调列独立对照，不自动追加200/10000。

下一课：同一个关节的 Actor 输出 -> 目标角 -> PD -> 实际反馈，由本人解释并判断，再回到对应字段 / 代码。A3 公开工程来源本轮已确认智元官方组织发布、基于 NVIDIA SONIC；型号兼容和实际训练仍按资源审计推进，见[主线与来源记录](README_LEARNING_MAINLINE_20261009.md)。

当前第一项：审计 A3 同平台预训练包、配置 / 资产、普通 A3 T2D5 控制接口及实时参考要求，比较 Q1 备选并确定第一平台。10/30 交付路线、资源缺口、最小验证证据与本人 pipeline 解释，见[方案](README_MODEL_ROUTE_20261009.md)。下面既有教学 / 平台步骤保留为各自入口，不要求先完成倒立摆奖励对照才能选型。

教学侧：用户已完成按命令复跑，已讲解观测/动作/奖励及倾斜惩罚力度，理解仍待反馈。下一步由用户按[已准备命令](README_LESSON02_REWARD_20261002.md)运行 `lesson02_tilt_weight3`，只把倾斜惩罚权重从 1 改成 3，然后核验结果并做阶段回顾；不直接比较两种奖励规则的总回报，不自动恢复 TD/GAE 推导。

工程侧：官方简单任务的训练闭环已完成，入口为 `script/teaching/lesson02/run_lesson.py`，实验 ID `lesson02_cartpole_20261002_E002`。无需重复从头做基础环境盘点；转向 X2 时仍需处理旧资产路径、配置覆盖与未提交改动，不直接接手旧实验。简单任务不等于 X2 自主站立/跟踪。

训练环境候选：`/home/yu/miniconda3/envs/x2-sonic-isaaclab`；框架：`/home/yu/projects/IsaacLab`（`/home/yu/IsaacLab` 是同一位置）。X2 训练候选的准确路径、提交与缺口见核验报告。新代码/配置与产物遵守 YUANQI 四目录约定。

10/2 教学阶段核验时，Agent 基线与用户复跑的训练/重载均已结束，GPU 无计算进程；随后只准备对照入口与 CPU 奖励数值检查。10/4 研究检查未重新核验训练/GPU 状态，也未启动仿真或训练。最近已核验完成的训练为 `lesson02_my_first_run`，模型/评测与终端记录分别在其 `data/training/`、`logs/` 目录；基线 E002 保留，备份状态均为 `LOCAL_ONLY`。下一次执行前核实 GPU、实际新产物与 Git 当前状态。

## 恢复时的最小检查

```bash
ssh hp3090
cd /media/yu/FAFF-E9771/YUANQI
git status --short --branch
git log -3 --oneline
```

先确认当前文件版本，再读本文件及相关阶段 README。只复核会变化的状态和仍未解决的问题，不重复已完成的静态盘点。
