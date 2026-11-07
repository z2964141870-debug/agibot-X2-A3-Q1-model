# CPU加速接入策略：64次保存动作等价检查

2026-11-08，北京时间。CPU输入等价已完成，此阶段核查动作与完整软件处理；原共享训练保持人工暂停，采纳null。

## 目标与协议

只更换CPU状态转换，保留原selected500的f0/history策略、归一化、人体接收时钟、历史、参考锚点与所有动作计算。复用已保存A022前走、A023起走，从frame0各32帧、总64次新CUDA前向上限，与原保存29槽动作逐帧比较`atol=rtol=2e-5`、缺失槽精确零。每条连续历史从原frame0开始，不挑选稳定片段或重新运行936调用。

新独立`native_state_action_worker.py`继承已审计SameStateSession；只在自身单次请求内将状态转换绑定到独立KinematicState，并在退出/异常后恢复函数引用。不编辑旧worker、原199执行/200冻结源码或CPU转换候选。物理积分全程禁止，动作期间另禁`mj_forward/inverse`。保存CPU全字段等价数组的原始状态必须与实际失败trace前缀逐项一致；二者SHA先核。

记录首调用、全部32次及去前两调用的分段CPU状态、观测/历史、因果接收与目标特征、传输/前向、完整worker及本机pipe RPC。新增转换分派包裹耗时另记；原worker内部计时范围保持，host与CUDA流区间嵌套、不可相加。没有本轮旧worker重算；旧耗时只作历史背景，不声称配对的完整链路加速。此检查不是硬件端到端延迟，不减少未来帧，也不修复f0闭环失败。

零新物理/媒体/Adam。H4/H10仍是含未来信息的参考；下一H4时间mask适配或因果闭环实验需要独立方案，不能由此直接宣称自然遥操合格。

## 来源与资源

CPU_STATE01_R2须为234状态/source_integrity PASSED且原结果SHA不变；897原输入及两显式来源复用。GPU账本是唯一允许后续改变的原CPU阶段输入：先核原2433秒/无RESERVED，GPU前持原锁reserve，全部自有worker回收后settle，不重置既有成本。其余输入、模型和旧FAILED根严格不改。

新控制器`native_state_action.py`、独立worker、`native_state_action_math.py`及五合同，另登记既有`native_state_cpu.py`。五本地合同已通过：连续前缀/末状态形状、错位与截断、动作容差及空槽、绑定模型/规格、异常恢复。真实策略运行时合同、部署/准入及64调用尚待；不以本地合同声称CUDA动作等价。

总300秒、单阶段300秒、输出64MiB；原22GiB准入/24GiB退出、GPU18GiB/80度、磁盘余100GiB/private小于500GB、原非阻塞GPU锁和自有子树回收保留。不清缓存、不重启训练。唯一新输出为服务器`data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_CPU_ACTION01_20261108`。派发前核output/dispatch/log/精确进程防重复；中途失联先查同任务，不重派。

先发布独立五文件清单、核40位远端和Mac源码，再当前45014原生密码会话部署、原策略运行时五合同/来源资源准入后有界运行。固定第一跳仍`source /Users/yu/.zshrc`再`ssh -tt hp_han`，不要求用户代测。

## 当前状态

实现与五本地合同通过，真实新CUDA/物理/训练0；服务器部署和实测以随后收据为准。CPU阶段证据已发布`f6da2817f5b6fd3e43f904dffd774edad4c648c0`，四小文件服务器镜像与899保护来源已核同，旧大包/全索引仍待同步。下一先此动作门，再回到未来帧依赖与闭环质量主线。

## 真实部署与准入

源码`6401814cb410bf886fe593ef2e051750f1edf479`已scoped audit、显式push/独立40位远端核同，Mac11文件820019B同。8205B五成员代码包SHA `fbafe01e21ee81fd40631d39c1fd79bb0d719fdf7716286b1e1c0ad4e3255a25`经当前45014逐片精确ACK部署，原899输入前后同。五原策略运行时合同本体.002秒/进程.224522秒全过；新准入12.729931秒通过、906保护来源同，原GPU锁及计算空/原训练暂停、总8053194752B/限34292629504B/OOM0/GPU1MiB25度/余盘183749693440B。三份原始回执5250B已Mac逐字节大小SHA匹配，尚无GPU任务派发或新的reserve；发布准入后只派一次。

回执：[部署](../logs/session_records/q1_cpu_action_deployed_20261108.json)、[原运行时合同](../logs/session_records/q1_cpu_action_contracts_20261108.json)、[准入](../logs/session_records/q1_cpu_action_admitted_20261108.json)。

## 实际完成与结论

准入`5fb3897dd114ffe2b2492e721a76d60fa1b22010`已显式push/独立40位远端、Mac6份540946B核同。03:06:34只派一次370268/start39131332，真实控制器370269/start39131332；两阶段各32请求且全部输入ready，实际64次CUDA前向、0物理/渲染/Adam。两guard为WORKER_COMPLETED/return0/worker_reaped，结果CPU_STATE_ACTION_PARITY_PASSED/source_integrity PASSED/906来源同；最大动作差前走2.741814e-6、起走3.337860e-6，低于原容差，无缺失槽污染。没有重跑旧worker/936调用/失败七例。

| 去每条前两次后，ms | 前走A022，30次 | 起走A023，30次 |
| --- | ---: | ---: |
| 新CPU转换中位 / p95 | .832 / 2.944 | .979 / 1.366 |
| 完整worker含转换分派，中位 / p95 | 9.882 / 34.723 | 11.052 / 15.050 |
| 本机pipe RPC，中位 / p95 | 11.434 / 36.851 | 12.086 / 16.258 |
| worker / RPC超过20ms | 4 / 6 | 0 / 0 |

全部32次中的worker最大310.618/299.242ms、RPC最大318.298/300.410ms保留，去前两次前走worker仍最大38.632/RPC41.332ms。CPU输入加速已保持策略动作，但完整软件长尾尚未解决。新转换在实际worker内的耗时与先前CPU配对微基准分开报告；本轮没有同时重算原worker，不能把单项约17倍改善外推为完整链路同倍改善，也不能将host与CUDA流区间相加。新增转换分派的上下文管理成本已纳入完整worker，不隐去。

03:09:05后核原身份及五训练PID暂停不变、370268/370269退、原GPU锁释放/计算空；总8036917248B/限34292629504B、OOM0/GPU1MiB25度/磁盘余183749152768B/private266931003392B。本阶段59秒已SETTLED，累计2492/14400秒，无RESERVED；原CPU阶段账本2433是历史快照，仅该动态账本在新GPU阶段依法预留/结算，其余保护来源不改。

45129B/15成员原始数值包SHA `ba677682617b9574a987cb0bc2e2a44200989acd612157817a15e7e063d0bb25`已经同45014十个ACK分片传回Mac，15原件逐件大小SHA核同。新软件链六宫PNG/PDF/全部与去前两次CSV已本地生成并目视；本次绘图首次选择的通用依赖运行时缺matplotlib，在建输出前退出，随后复用已有q1-teleop-mps环境生成，没有安装依赖或增加GPU调用。

新证据包`data/deliveries/Q1_CPU_ACTION_CHAIN_EVIDENCE_20261108.zip`为327401B/SHA `9e81bfe84d232c688a112510f045e6cfbebc8b6fb2a53292309caca579d8d0ae`，21成员CRC/逐件大小SHA/7链接通过。没有新MP4，原CPU输入两图805214B证据包完整保留；[用户简报](README_Q1_CPU_ACTION_RESULT_20261108.md)、[真实派发](../logs/session_records/q1_cpu_action_dispatch_20261108.json)、[完成与资源](../logs/session_records/q1_cpu_action_completed_20261108.json)、[交付](../logs/session_records/q1_cpu_action_delivery_20261108.json)。

下一实质工作回到未来帧与闭环质量：核已归档SONIC/TWIST/GigaBrain源码及H4完整事件失败证据，独立登记H4时间mask或带闭环目标的因果方案，保持validation选模/held-out事件及自然性诊断。不再为性能重复234状态或64/936调用，也不继续旧f0同数据拟合；任何新GPU实验先源码、来源、资源与剩余11908秒预算发布审计。原共享保持暂停，旧f0七失败与final完成门和采纳null保持。
