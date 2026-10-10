# A3 官方模型全链路执行

日期：2026-10-10，北京时间。独立工作根为 hp3090 的 `/media/yu/FAFF-E9771/YUANQI_A3`，main 分支。状态：当前可执行工作已结束，24任务18通过（限定范围）/2失败/4搁置，65检查通过；最终矩阵、工件和用户待办见文末。官方基线20/20无跌倒，200次短微调5/20跌倒，效果退化；缓冲与导出链路通过。旧全链路记录和 R01/R02/R03 保留暂停。下列阶段条目保留实际推进过程，最终状态以文末和新清单为准。

## 已完成与范围

官方035发布包5文件已验证，模型 SHA `9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d`，CPU重载及网络/优化器有限性通过。固定HF revision和配置身份见 `data/manifests/a3_official_import_20261010.json`。本轮没有重新下载模型。

官方10策略步冒烟实际退出0，完整10步、数值有限、未跌倒；全29关节RMSE 0.050375 rad。只覆盖0.2秒，不作为动作稳定性验收。证据 `data/experiments/a3_fullchain_20261010/official_smoke/explicit_summary.json`，日志 `logs/a3_fullchain_20261010/official_smoke/`。CPUQuota100%执行。

新台账 `data/experiments/a3_fullchain_20261010/tasks.json` 独立初始化。按旧Git工件清单逐文件核验大小/SHA后复用人体参考、A3参考、运动学和MotionLib，见 `reused_history.json`。未复制旧训练账本。seed42整动作16训练/4留出与原固定分配一致。24个PKL格式/数值与26个CSV时长合同重新验证通过；120Hz官方CSV stride4，30Hz新参考stride1，转换后30Hz。动捕TEST_ONLY标识保持，IK/脚穿地质量仍失败，微调使用官方样例。

## 官方完整基线

10:25完成selected20全长度MuJoCo回放，20/20未触发跌倒，24304策略步完整覆盖。按步数加权的全29关节RMSE为0.059060 rad，根位置误差均值0.084169 m，根姿态误差均值2.067994°。未发生跌倒，因此跌倒前与全程统计一致。逐动作指标、时序SHA、完整覆盖校验见 `official_baseline/explicit_summary.json` 与 `official_baseline_aggregate.json`。此结论限于固定资产、离线参考和仿真，不代表真实衣服、高保真验收或真机安全。

16/4划分只隔离本次R04微调，不能声称官方预训练阶段从未使用这4个动作。后续微调保持相同MuJoCo资产、动作、frame0初始化与延迟设置，单种子短试训不证明普遍改善。

## 微调前后对照

11:31全部20动作完整覆盖同条件评测结束。官方0/20跌倒，R05 step200为5/20跌倒。16个训练动作从0/16跌倒到4/16；4个留出动作从0/4到1/4，失败留出为右单腿平衡。留出全程关节RMSE从0.066318rad增至0.215507rad，跌倒前RMSE从0.066318增至0.186666rad；留出根位置平均误差0.111122m→1.477976m，跌倒前0.111122m→1.380673m。原始逐动作首次跌倒/完成、时序SHA与窗口统计见 `paired_comparison.json`。

| 固定动作分组 | 官方无跌倒完成 | step200无跌倒完成 | 全程关节RMSE官方→微调(rad) |
| --- | ---: | ---: | --- |
| 16训练动作 | 16/16 | 12/16 | 0.056607→0.271324 |
| 4留出动作 | 4/4 | 3/4 | 0.066318→0.215507 |

对照计算链路通过，策略效果验收失败，不能把训练200更新或退出0视为改善。没有自动追加训练/调参。小批量短试训稳定性、重启回合重置、学习步长与KL均是后续诊断候选，尚未归因。13项摘要有限不代表训练分布/优化质量良好，最后20更新KL均值0.239462、clipfrac0.484063仅作证据。

官方与step200 ONNX各100真实输入均通过容差，误差分别2.8610e-6、3.8147e-6；后者ONNX SHA `f1c982f68a7dd519fb53a2f97c78a4c0b8774756ff263abe9a834eaf3c664111`。导出正确不能证明策略性能通过。下一步仍用官方模型执行四段动捕和完整缓冲诊断；本轮不部署微调模型。

## 动捕策略诊断

11:35四段转换后的30Hz参考stride1完整MuJoCo回放结束。走路、单腿与9/23录制触发跌倒，9/24录制未触发，合计3/4；仅诊断完整覆盖通过，人体姿态/轨迹/接触质量仍失败，原数据TEST_ONLY/training_allowed=false不变。逐动作首次跌倒、回放完成、跌倒前与全程跟踪误差见 `official_mocap/explicit_summary.json`，对应源SHA与配置见 `reused_history.json`。不能与20个官方样例混为同一数据集验收。

## 独立卡点

Isaac指标导入再次失败：`ModuleNotFoundError: No module named 'smpl_sim'`。评测需兼容 `smpl_sim.smpllib.smpl_eval` 和独立validate，暂时搁置，退出码不能代替指标。RKNN模块仍未安装，转换/包和板端推理搁置。此二项不阻止MuJoCo和短微调。动捕根轨迹、接触、个人标定与姿态质量待用户处理，结构测试不能证明高保真。

## 版本与复现

vendor固定 `fe6868ba37034f89b912f0fb851bce19f120266d`，Python为新目录的 `data/environments/a3-sonic/bin/python`，读取复用原环境。本轮boot `5723f6c9-951e-4866-b4dd-cf4663b8084d`；启动前旧服务inactive、GPU无计算作业。重启仅登记和恢复。

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
data/environments/a3-sonic/bin/python -m script.a3.fullchain_support download
data/environments/a3-sonic/bin/python -m script.a3.fullchain_support split
data/environments/a3-sonic/bin/python -m script.a3.prepare_official_stage
data/environments/a3-sonic/bin/python -m script.a3.reference_contract motionlib
data/environments/a3-sonic/bin/python -m script.a3.reference_contract validate
```

以上download入口实际只核验5个已有文件。新工件留服务器LOCAL_ONLY，未进行网盘往返校验。

## 短微调执行

11:26–28 R05已实际完成200更新，16环境、save10、2次冒烟后完整恢复，10次独立attempt；重启恢复未重置预算或提高阈值。最终路径 `data/training/a3_finetune_20261010/R05_010_s120/model_step_000200.pt`，375295123字节，SHA `bba46271ef3c9962dcb01d525811a5b4f19b4d3cc2fe027afcf384e0f40e7a34`，独立CPU重载、网络/优化器有限性通过，优化器计数4000=200×5epoch×4minibatch，16环境检查通过。证据 `finetune_verification.json`，最后训练13项摘要标量有限 `final_training_metrics.json`；这些不证明策略改善。

R05 ledger complete，训练服务正常退出并disable，旧R04及R01/02/03仍不恢复。微调模型完整selected20同条件评测已启动，输出 `finetuned_evaluation/`，服务 `a3-evaluation.service`，完成200后不会自动加训。随后四段动捕诊断、因果缓冲完整对照和微调ONNX；最终模型LOCAL_ONLY。

R04第一次启动在加载训练前因vendor软链接路径导致本地 `gear_sonic/trl` 遮蔽已安装Hugging Face `trl`，退出1、实际更新0；已保留blocked账本，禁用其新服务。明确修正为REPO_DIR使用vendor物理路径，不安装或修改共享环境/第三方代码。

独立R05使用相同16环境/固定16训练动作/官方035优化配置，旧账本不重置。10:30完成2次更新冒烟，保存 `data/training/a3_finetune_20261010/R05_001_s0/model_step_000002.pt`，375291345字节、SHA `324e573377f5c4b66ea698b59bf0cae985ffb0883696189d3f38352db4d2569e`；保存事务已CPU重载与网络/优化器有限性检查。初始化回执 `logs/a3_fullchain_20261010/finetune/R05_001_s0/warm_start_loaded.json` 检查官方源200000、加载计数0、两网络逐张量相等、优化器空状态。完整恢复回执和最终200还需实际核验，不把2次冒烟当200完成。

R05服务 `a3-finetune-r05.service`，CPUQuota100%、linger=yes，重启后按300秒冷却自动恢复，无启动次数/有效期限制，原温度/异常/两次无保存进度保护不变。job与state位于 `data/experiments/a3_fullchain_20261010/finetune_R05/`；普通异常exit2不自动重启。达到200退出，不追加训练。最近SSH超时暂记UNKNOWN，恢复后重新核验boot/sidecar，而非依据旧PID。

10:31连接恢复实际boot `692bd7da-ce27-435a-8193-2babaa54f941`；R05服务activating/exit75且账本为recovery_cooldown，已识别保存step2，冷却后自行恢复。第二attempt源为独立step2并full_state_resume=true；本轮只登记重启，不定位根因。

后续独立恢复到step10，回执恢复优化器计数200；第三attempt已保存step30。10:38后boot变为 `a6105f54-bd00-499d-b6ea-6783957f500e`，继续300秒保护冷却。实际初始化两个优化器参数组LR均为2e-5，step2恢复优化器计数40，网络逐张量相等；这是已运行官方035实现的实际值，而非仅抄配置字段。

`script.a3.evaluation_job --stage finetuned_evaluation` 与 `a3-evaluation.service` 已登记，等待R05完成或保护停止，不并行使用GPU；只有冷却结束且训练退出后启动。其后每个评测阶段单独选择stage，按完整动作SHA恢复；普通失败exit1/2不自动解除，重启exit75等待。阶段可选微调selected20、四段动捕、官方selected20/动捕因果缓冲；训练未完成时入口实际返回75，未启动评测。

实际缓冲100策略步冒烟退出0且无跌倒，证据 `buffer_smoke/`，完整selected20与4段动捕策略缓冲对照待微调结束后串行执行。

已补缓冲EOF分支：只有源末帧与结束事件已到达后，允许按官方hold_last填末尾窗口，确保最后200ms不被恒定回退截掉。EOF边界与未来数据不变性共4项检查通过；实际输入轨迹增加 `end_of_stream_seen` 字段。评测监督锁覆盖所有stage，避免多个入口同时启动。最新有效微调step40，未完成200。

10:16服务器连接中断后恢复，boot变为 `3f33490f-5dea-483f-8f83-a08e53a04b01`。磁盘持久完整动作16个，剩余4个按manifest恢复，不依赖中断前终端打印。GPU无计算进程；按300秒冷却等待后继续GPU任务，不排查重启根因。微调尚未启动。固定16动作输入preflight已通过。

恢复独立SHA检查进一步确认：上述16个指标文件仅13个有效，3个SHA不符，另1个运行中动作中断。无效输出和原attempt保存在 `official_baseline/incomplete_artifacts/<boot>/`；13个有效动作跳过，其余完整重跑。包装器已补metrics/timeseries/input/buffer文件fsync及目录fsync后才登记passed，防止manifest先落盘。恢复入口 `python -m script.a3.recover_evaluation <output>`，只能在评测进程停止后执行。不能把中断前屏幕进度当持久完成数。

新增因果策略回放钩子，保留官方vendor不修改。编码器速度采用前向差分，最后0–180ms姿态窗口还需要200ms处的位置，因此名义180ms缓冲实际先预填充200ms，并按30Hz到达量化记录实际年龄。逐策略步核验插值及速度所需的最后原始样本已到达；指标针对延迟后的目标帧，启动等待与模拟到达另记。新增3项未来数据扰动/速度边界/指标目标检查及已有17项检查通过，共20项。

官方CPU ONNX已实际导出，100个MuJoCo真实策略输入对比通过：输入[1,1570]、输出[1,29]，最大动作误差2.8610e-6，满足atol=1e-4/rtol=1e-3。模型57673235字节，SHA `5fbb77310cbe63f1468f9f1cbbec8c056256b4806cf4c3a1b15ccba239f66122`。证据 `onnx_official/parity.json`、真实输入 `official_baseline/001_walk_front_slow.inputs.npz`。使用固定MuJoCo策略适配器及opset13 atanh等价表达，未声称原Isaac导出入口或板端通过。

开始官方完整selected20，逐动作可恢复且完整覆盖核验；然后独立R04从官方权重初始化，计数与优化器重置，16环境，2次冒烟后完整恢复至200，每10保存。无累计启动次数/到期限制，CPU90°C/GPU85°C连续2次、5秒采样、300秒恢复冷却、异常与连续2次无保存进度保护保留。评测和训练串行，不操作真机。

每阶段更新本README、索引/current state，审计后scoped提交推送main并核验远端。发布版本以Git历史为准；后续更新实际结果后再报告完成。

## 最终交付与测试矩阵

11:43本轮当前可执行分支全部结束。新台账24项为18通过、2失败、4搁置，待做/运行中为0；通过项包含结构/计算/训练链路，不能合并解释成策略验收通过。65项项目检查通过，日志 `logs/a3_fullchain_20261010/final_tests.log`；2失败为动捕参考质量与微调策略效果，4搁置为Isaac指标依赖、Isaac基线、RKNN转换/包、动捕微调。

| 分支 | 状态 | 实测与边界 |
| --- | --- | --- |
| 官方5文件身份、CPU重载 | 通过 | 固定发布SHA、网络/优化器有限；LOCAL_ONLY |
| 4段SMPL结构/时序与A3重定向 | 通过（诊断） | frame_index/旋转/单位/29关节/合成检查；不代表高保真 |
| 动捕参考质量 | 失败 | IK残差、脚穿地；原TEST_ONLY不变 |
| 官方20动作MuJoCo | 通过（固定仿真） | 全24304步、0/20跌倒；RMSE0.05906rad |
| 官方4动捕MuJoCo | 通过（诊断链路） | 全13390步、3/4跌倒；跟踪质量未验收 |
| 独立R05 2→200 | 通过（训练链路） | 权重初始化/优化器计数重置，最终CPU重载、有限性及optimizer4000 |
| 同条件20动作对照 | 通过（计算） | 16/4固定划分、完整回放及跌倒前统计；策略效果失败 |
| 微调效果 | 失败 | 0/20→5/20跌倒，4留出0/4→1/4；留出RMSE0.06632→0.21551 |
| 因果策略缓冲完整对照 | 通过（模拟回放） | 官方20动作0/20→0/20；4动捕3/4→2/4，无未来源帧读取 |
| PT/ONNX | 通过（导出数值） | 官方/微调各100真实输入、1570→29、指定容差；无板端结论 |
| RKNN输入契约 | 通过（格式） | 实际官方ONNX与固定转换器schema契约；无RKNN二进制/包 |
| Isaac / RKNN / 衣服微调 | 搁置 | 依赖与质量卡点，后续处理见下表 |

完整缓冲对照累计37694实际策略窗口均未读尚未到达源帧；名义180ms、速度前向差分需要200ms预填充，30Hz到达/50Hz策略实际目标年龄200–240ms。官方selected20缓冲关节RMSE0.059128rad，离线0.059060rad；动捕缓冲2/4跌倒仅诊断。结束事件必须已到达才允许hold_last，完整动作末尾已覆盖。证据 `policy_buffer_comparison.json` 与两目录 `*.buffer.npz`，不代表现场时钟/网络/端到端延迟。

已有时序重算了躯干/手腕/腿部世界和anchor相对坐标误差，未额外运行仿真。4留出全程手腕anchor相对误差0.008556m→0.056926m，腿部0.022805m→0.051218m；跌倒前手腕0.008556m→0.039560m，腿部0.022805m→0.047691m，同样支持退化结论。原始与全部分组值见 `paired_comparison.json`。

## 工件与恢复位置

均以 `/media/yu/FAFF-E9771/YUANQI_A3/` 为根；大型模型、NPZ、PKL与完整轨迹仍LOCAL_ONLY，未确认百度网盘往返备份，不删除原件。

| 工件 | 相对位置 |
| --- | --- |
| 官方完整模型包 | `data/models/a3_official_035/checkpoints/035_step200000/` |
| 固定划分/人体与A3参考/来源SHA | `data/experiments/a3_fullchain_20261010/{split.json,mocap,reused_history.json}` |
| 官方基线/微调对照 | 同实验根 `official_baseline/`、`finetuned_evaluation/`、`paired_comparison.json` |
| 动捕与缓冲逐动作证据 | 同实验根 `official_mocap/`、`official_buffer_selected20/`、`official_buffer_mocap/` |
| 最终微调权重与配置 | `data/training/a3_finetune_20261010/R05_010_s120/` |
| 训练独立校验/台账 | 同实验根 `finetune_verification.json`、`finetune_R05/{job.json,state.json}` |
| 两个ONNX与100输入数值证据 | 同实验根 `onnx_official/`、`onnx_finetuned/` |
| 执行日志/保护采样/检查 | `logs/a3_fullchain_20261010/` |
| Git小型清单/测试矩阵与全部文件SHA | `data/manifests/a3_official_stage_20261010.json` |

训练保护134条采样CPU最高74°C、GPU40°C，仅说明这些采样，不能推断重启原因。R05共10次attempt完成，未修改R01/02/03、旧全链路台账或Q1副本。当前训练与评测都结束且服务disable，GPU计算任务为空；巡检yuanqi-a3仍PAUSED，每日研究yuanqi仍ACTIVE。最终提交由Git历史识别，审计后scoped推送main，核对HEAD与远端；失败时保留本地证据，不强推。

## 用户后续事项

| 卡点 | 证据/影响 | 后续处理 |
| --- | --- | --- |
| Isaac指标依赖 | `dependencies.json`，缺 `smpl_sim.smpllib.smpl_eval`，Isaac效果UNKNOWN | 提供与官方评测兼容的指标实现/依赖；随后独立validate后才能验收 |
| RKNN工具链 | `rknn`导入缺失；无转换包与板端证据 | 提供兼容RKNN转换环境及目标平台信息；板端推理需另行测试，本轮不操作真机 |
| 动捕质量/现场接口 | `retarget_summary.json`、`input_audit.json`；衣服微调搁置 | 核对个人标定、重建精度、世界根轨迹、地面/脚接触、现场协议与真实时间戳；质量验收前不修改TEST_ONLY |
| 200次微调退化 | `paired_comparison.json`，5跌倒且留出误差增加 | 保留官方模型为当前仿真基线；后续独立设计稳定性/学习步长/数据分布诊断，不能直接部署该微调版本 |

本轮已推进到剩余事项依赖上述资源/质量处理，停止追加训练和评测。恢复先读本README/current state及清单，已验收项跳过；不要重新恢复旧长训。
