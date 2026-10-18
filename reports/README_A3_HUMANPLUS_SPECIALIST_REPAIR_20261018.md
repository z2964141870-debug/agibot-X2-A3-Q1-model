# A3 humanplus 专项入口修复与恢复

记录日期：2026-10-18。RC01/D05/HP21/R27。19:50已完成独立修复、标准库控制和闭包冻结，尚未登记或运行新科学阶段。用户明确要求“修，修好了继续优化”；HP20始终保留STOPPED，本文件不把旧失败改成通过。

## 目标与范围

修正HP20参考身体重复索引及潜在关节顺序假设，先完成真实Isaac整段035基线，再继续原zzx专项专家路线；专家可靠优于035后才讨论学生访问状态上的纠正与统一蒸馏。仅在humanplus:/home/humanplus/YUANQI_A3计算，Mac只代码/记录；不动共享包、vendor、驱动或其他任务。

## 修复与依据

新humanplus_specialist_repaired_entry/trial/verification及独立audit绑定HP21/R27；复用未变的专家trainer/protocol。原HP20五注册模块不改。reference body已按cfg.body_names筛选，直接取该顺序；机器人身体独立按名字对应。参考关节在原生has_dof_mismatch时用body_joint_indices恢复显式源名字，否则用完整robot.joint_names，再分别映射到action_term的29个动作关节。CPU检查names/shape/原生action ids/body ids一致后才GPU取值。评测增加同步定位、bridge_metadata和每步同步后的replay_progress，禁止把未知步数补成零。科学配方/预算/效果门保持，CUDA_LAUNCH_BLOCKING仅在两个Isaac评测阶段开启，不增加模型forward或交互。

## RC01历史故障的显式恢复处理

本次用户修复并继续的指示是恢复授权。NVIDIA Xid目录原文将43描述为应用软件故障后必须退出，不单凭此认定驱动/硬件坏；已只读核对https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html。结合HP20越界断言、同PID Xid43、原组已退出，采用新应用进程恢复方式，不reset GPU/重启/改权限；唯一因果仍未同步定位证明。

RC01 metadata-only三次5秒间隔采样实际完成，原guard每次仍报告CRITICAL_KERNEL_EVENT，精确等于保存的cursor/message和pid2970870那一条；无其他内核事件/新增OOM/旧组残留，驱动gpu_recovery_action每次None，原资源启动门其余全部通过。RC01结果SHA3fb53aec80768d36bbb5422d885ec3df8000d965196bbc9db1c5a50ab70286cb；记录gpu_execution_recovery_verified=false，不能把元数据当真实执行恢复。

新recovered_health/supervisor只用于独立HP21，原health/supervisor行为不改。原HP20故障前baseline文件逐值保留，原cgroup内存/限额/计数连续核验；同一全boot journal同时验证故障前和故障后前缀SHA/boot/cursor/单调时间。每采样完整保存历史fatal，并明确标为已处理的历史事件；只允许RC01固定的一条cursor/message，新增、缺失、修改、OOM或boot变化一律拒绝。该处理是明确记录的历史事件恢复授权，不能称“全boot没有FATAL”、悄悄清零或为后续事件自动豁免。原8GiB/4GiB、三次5秒启动、CPU90/GPU85两次5秒及900秒期限均保持。

## 版本、控制与入口

前提交46fc550。新增6项标准库控制Mac5通过/真实vendor缺失1跳过，humanplus6全部通过：历史事件精确匹配、新事件/日志变化/新OOM/boot/授权拒绝，参考31/29关节和反向顺序/被动关节，独立绑定与原生vendor AST匹配。没有运行旧控制、旧审计或孤立GPU诊断。D05冻结341源码、112现存输入、11环境源码；输入含RC01及其固定证据链，不继承已删权重。

准备清单data/manifests/a3_humanplus_specialist_repaired_design_20261018_D05.json与a3_humanplus_recovery_20261018_RC01.json，原始元数据logs/a3_humanplus_recovery_20261018_RC01/metadata_samples.json。控制命令`/usr/bin/python3 -B -m unittest script.a3.test_humanplus_recovery script.a3.test_humanplus_specialist_repaired -v`。闭包固定后代码不可改写，下一单次`/usr/bin/python3 -B -m script.a3.humanplus_specialist_repaired_trial register`，阶段为isaac-baseline/train/verify-training/isaac-final/mujoco/effect，每阶段通过独立humanplus_specialist_repaired_audit才继续。

## 固定预算与验收

沿用专项256更新/16env×24步=98304交互/1024原生Adam提交/8192外层Actor+decoder调用；只保存0/256，只最终256评测。fresh035/空Adam、全原decoder可训、codec/std冻结、原奖励/物理/reset/已开启失败采样/实际cached token/配对路由/KL LR保持。任何科学STOPPED不续跑/扩预算/评早期。Isaac匹配前后各499步，MuJoCo最终499步与现存HP11比较、不重放HP11。两模拟器整段不倒且关节RMSE各改善至少1%、MuJoCo根腕<=+5%才达专家中间门；不算统一策略验收、收敛或部署批准。

## 保存与下一步

当前0新模型/仿真/optimizer，RC01仅明确可开始受监督的新进程，不等于修复真实验收。先保存README/索引/current state/小清单与scoped提交，再独立登记HP21，按三次启动门完成499步基线和正常Kit关闭。基线通过后才训练；首新增内核/温度/OOM异常停自有组、暂停自动任务并保存入口。zhy-a3前台修复期间保持PAUSED，修复实际通过后恢复原30分钟。GitHub已知写鉴权失败不重复push，LOCAL_COMMITTED_UNPUSHED。大数据日志LOCAL_ONLY；035及必要输入保护，已删权重不恢复，继续按精确清单管理淘汰权重。

## HP21/R27登记完成

实施提交f94e9e9之后独立register已单次完成，固定D05/341源码/112输入/11环境及原预算。登记元数据接受RC01精确历史事件、没有新异常；0本合同模型/仿真/optimizer。下一依实际三次5秒启动门单次isaac-baseline499步，完成后独立审计；登记快照不是启动或GPU恢复证明。旧HP20与全部旧科学/审计不复跑。

## 19:56 修复后的真实499步Isaac基线通过

登记提交7d1d7c6后，HP21 isaac-baseline单次完成55.213977秒，native PID2984506/原生退出0、Kit关闭验证和独立审计verified=true。实际499步每步同步/进度收据齐全，身体与29动作关节桥接真实元数据一致。035整段不倒、匹配Isaac关节RMSE0.10478640564907678rad，trace SHA39b1f73fa9327c0430695a34a652cdfbc6303daccf983e67c0a978a48a8a76f8；不能与MuJoCo数值等同或称模型提升。

12健康同原boot，CPU最高87/GPU48°C，新增kernel/OOM均0、审计after_issues空、自有组残留空；每样本仍保存RC01处理的原历史Xid43，未称全boot无FATAL。当前应用执行与原生退出修复已在完整阶段通过，非硬件终身健康保证。下一按已登记256更新预算单次train，完成后独立审计与保存验证，通过才最终双模拟器效果；旧HP20/旧审计不复跑，原保护不放松。

## 20:05 训练完成，保存验证入口依赖故障

HP21 train实际256更新/98304交互/1024Adam，6144 rollout Actor+1024原训练Actor+1024 cached decoder=8192外层调用完成，325.196653秒，native PID2986694退出0/Kit正常关闭/独立train审计verified=true。初始实际token概率max_abs_ratio_minus_one1.52587890625e-5、action maxabs2.384185791015625e-6，通过原门；冻结参数/Actor buffer检查保持。保存0/256存在，仅训练成功，不代表效果。

随后的verify-training在导入cloth_constrained_trainer→训练基类→gear_sonic时ModuleNotFoundError，发生在torch.load之前；未开始任何checkpoint反序列化或验证、0最终回放。该模块不应作为纯保存概率检查依赖，已有humanplus_saved_probability纯函数可替代。HP21整个合同保持STOPPED、单次verify审计verified=false，不重跑或改其注册源码/结果。累计82健康CPU最高87/GPU54°C，after_issues空/无自有组残留，0新增kernel/OOM；不是新机器故障。

下一独立完成合同只接纳HP21已成功完整训练阶段及0/256精确大小SHA，采用纯概率核验器，不新增训练/交互、不重放HP21基线、不加载旧失败或淘汰模型。独立登记后才首次实际保存验证及最终Isaac/MuJoCo效果，原256训练预算和双模拟器门保持。不得为这个导入错误重训256更新或把HP21 STOPPED改成通过。用户本轮修复后继续优化授权有效。

## HP22独立完成合同实施

保存故障提交cfd8f8c后，新增completion/entry/verification及freeze/audit，完成阶段仅verify-training/isaac-final/mujoco/effect；不含train或isaac-baseline，直接拒绝原生训练或基线入口。源合同精确要求HP21基线和训练各COMPLETE且审计true、verify在PT加载前gear_sonic导入失败且审计false，原终态不改。新verifier仅用humanplus_saved_probability，未引入训练或Isaac依赖；三项标准库控制Mac/新机各通过，不把静态控制当模型通过。

D06冻结347源码/133现存输入/11环境，精确纳入源0/256权重及sidecar/所有权、初始token概率、原配方、实际optimizer提交账、成功基线/训练及失败入口证据。没有递归采用旧STOPPED全产物链、已删PT或新训练；新合同只核验已完成训练，首次最终Isaac/MuJoCo各499步。下一scoped提交后独立登记HP22，新增训练更新/交互/optimizer均0，原HP21预算不扩，原专家及统一效果门不变。

HP22已在实施提交10f9f51后单次登记，固定上述D06和阶段/源证据。当前0本合同张量验证或最终回放，下一单次verify-training及独立audit，通过后才isaac-final。新增训练预算0，不把源合同STOPPED改写为成功。

## HP22保存验证配置归属错误停止

登记c4ec0d7后单次verify执行了0/256网络、冻结范围、Adam矩/计数和初始token概率检查，随后validate_config错误地拿completion输出HP22/motionlib比对源训练配置中HP21/motionlib而停止；无最终成功收据，不能补称完整验证通过。原因是新完成合同的源/目标目录引用错误，不是模型科学失败或机器异常。单次审计verified=false，4健康最高CPU83/GPU43、0新增kernel/OOM/残留；未启动Isaac最终或MuJoCo，不重复此旧核验。

独立修正须将源配置/配方检查前置到权重加载之前，精确绑定source HP21/motionlib；源PT及sidecar不改，不重复训练或基线。新控制须固定源目录表达式防止再次误用目标OUT。后续新合同完整模型核验即使部分检查重复，也明确计入新核验成本，不伪称此前没有张量加载；不扩大科学交互/optimizer/评测预算。

HP23/D07独立final完成入口已实施，3控制Mac/新机各通过，新增控制固定源motionlib表达式且validate_config必须在torch.load之前；真实源config YAML/原rewards/events/terminations/actions四节无torch导入预检通过。D07固定353源码/133输入/11环境，仍0新增训练更新/交互，源0/256权重不改。原HP22停止提交c911796保留；下一独立登记HP23后完整保存核验，全部通过才最终双模拟器。未把部分通过或静态预检当成完整模型验收。

HP23已在实施90d1201后单次登记，D07及源权重/配方/RC01闭包固定；0新增训练，下一单次verify-training和独立audit，成功才最终Isaac/MuJoCo。HP21/22终态保留。

## HP23完整保存验证通过

登记396d81e后，verify-training及独立audit单次完成/verified=true；step0为134048343字节/SHAa886811d797eba407bcc6bc343b6f6986c877af941ed329ed57d58cfa15a7228，step256为307574417字节/SHA47dd293511428d373c95c316ec8674e2b0f57bd581a09a632252dab1c09c599f。初始Actor/Critic匹配035、最终14个原decoder参数变化且冻结部分保持、28个optimizer参数所有权/ID/矩/1024计数、实际token及初始概率、原奖励/reset/物理四配方节全部通过。4健康最高CPU81/GPU43，0新增kernel/OOM/残留；仍保存RC01历史事件。只有保存验证成功，无新训练或效果证据。下一首次最终Isaac499步及独立审计，通过后MuJoCo499步与原HP11比较。
