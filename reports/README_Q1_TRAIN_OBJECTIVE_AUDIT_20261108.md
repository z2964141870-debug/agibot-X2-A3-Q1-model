# Q1训练目标只读诊断

日期：2026-11-08，北京时间。实验：F0_TRAIN_OBJECTIVE_AUDIT01。状态：真实GPU只读诊断通过，243前向批次/22929行/84次autograd，0optimizer/物理/渲染，30秒已结算。该阶段完成不代表全身自然遥操目标完成。

## 目标与依据

上一轮[旧能力保持序列配对](README_Q1_RETAINED_SEQUENCE_RESULT_20261108.md)两组末轮综合验证误差下降约9%，旧三项验证仍退4.8%至7.2%，全部非零选点被拒。两组选中策略精确同起点，原0/7闭环证据直接复用。继续延长同配方没有充分依据。

本轮先区分旧训练源上的误差是否也退步，以及同一训练批次上旧目标、新状态目标、输出保持与教师动作增量的局部梯度是否冲突。梯度相关性和一阶近似不能证明失败因果，也不能证明f0不可能。

## 固定方案

1. 只解码三份旧train数组共6725行和纯学生train406行。严格沿原来源、组、split与行顺序映射，复用6581旧窗及387纯学生窗；不解码validation/test数组，不用未来状态作为模型输入。保护文件仍做SHA完整性读取，不能误称所有验证文件字节都未读。
2. 检查实际已训练causal_history3000起点，以及上一轮retained_point、retained_delta各新增3000后的last。所有权重、teacher/WM、人encoder、decoder与归一化冻结，不更新或另选模型。
3. 三个权重点都测完整train的组等权误差，另外保留旧origin0/1片等权指标。validation数值仅在分析时复用已保存记录，不计算validation梯度。
4. 原seed2026110806、16旧窗＋16纯窗，精确重建原训练最前四批次；求七目标对四个adapter张量的梯度，记录Gram、cosine、范数和两个实际位移的点积。总目标梯度沿原0.5/0.5、20倍保持和0.5倍增量组合；零梯度或缺失域的cosine为null。

上限1000个学生前向批次、30000行输入、84次autograd、300秒和64MiB新输出；0optimizer、0新物理、0渲染。仅f0＋历史输入。软件计算不是硬件端到端延迟，不判自然遥操合格。

## 来源、资源与执行

五新增文件在`script/supermimic_teleop`：`student_objective_controller.py`、`student_objective_worker.py`、`student_objective_audit.py`、`student_objective_plan_20261108.json`、`test_student_objective_audit.py`。旧六训练执行文件和既有实验根不修改。

原累计4790／14400 GPU秒，r1/r2/r3为1614/1805/1371，r2剩2695；本轮记原r2台账，reserve→worker回收→settle，不重置。保持22GiB准入/24GiB退出、GPU18GiB/80度、100GiB磁盘和private500GB、原非阻塞锁及自有子树清理。原共享大训练保持用户暂停，不清cache。

唯一服务器输出根：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_F0_TRAIN_OBJECTIVE_AUDIT01_20261108`。派发前检查输出、日志、派发收据与精确进程皆无；观察过期不能重复派发。现有Linux45014优先复用，获批hp_han三值90/60/5保持，必要恢复由Agent自主原样source/ssh两跳密码执行。

本地命令，工作目录为权威Q1工作副本`script/supermimic_teleop`：

```sh
/Users/yu/Documents/ChatGPT/元启/data/environments/q1-teleop-mps-20261106/bin/python -m unittest -v test_student_objective_audit
```

六项全过/.168秒及CLI通过，覆盖原映射/批次逐位一致、split/终止边界、拒绝验证文件和未来对象解码、已知梯度符号/零梯度/位移点积、组等权、真实前向和autograd不更新模型且禁止创建Adam。真实原运行时合同及来源/资源准入待发布部署。

服务器执行须本阶段发布清单与SHA，先controller的`--admit-only`，实际唯一执行同命令移除此参数；外层`timeout --signal=TERM --kill-after=5s 310s`，原runtime guard为300秒。实际参数和收据随后登记，不用未发布SHA占位执行。

## 原运行时与真实准入

源码`bf3842bc83bc5d97df1ce34d0dce793ff2ae0a95`和清单发布`9324dea957514304ef6ecf86ff9d40e6e7df5fc7`均已显式push origin Q1、独立40位远端核同；源码九Mac文件590605B核同。16199B九成员包SHA `baafc2ca0c72698a2faf3ea5c164ef87281a4ed4c2f1e8f23218efb685f84bf2`，经45014原Linux逐包13精确ACK部署，1216旧保护文件前后同。

原Morph运行时六CPU合同/.608秒、外层进程2.922691秒全过。10:13:35北京时间，1229来源/原暂停/设备/预算/锁准入11.264721秒通过，总8270012416B/限34292629504B、GPU1MiB24度、OOM0、磁盘余183398465536B；原4790秒台账无预留。实际新Q1前向/optimizer/物理/媒体仍0。

三原回执2122B包SHA `d9a57b471d5e081eabab29d115587d88809b5bc9456e3bc65e91e75c433fed92` 已Mac二进制核同和精确解包，服务器原文未改。先发布本准入，再唯一防重派发；成功部署/合同不重复。

- [新增部署](../logs/session_records/q1_train_objective_deployed_20261108.json)
- [原运行时合同](../logs/session_records/q1_train_objective_contracts_20261108.json)
- [真实准入](../logs/session_records/q1_train_objective_admitted_20261108.json)

## 真实结果

唯一391155/391156-start41714306及worker391230已退出，guard清洁回收0，1229保护输入SHA前后同。北京时间10:17:56完成`TRAIN_ONLY_OBJECTIVE_AUDIT_PASSED`，没有另选策略或改变三个被检查模型；复用已选原策略的0/7闭环证据，不制作相同新视频。

两组last相对实际起点的MSE变化：

| 指标 | point训练 | delta训练 | point原验证 | delta原验证 |
| --- | ---: | ---: | ---: | ---: |
| 全部旧源 | +5.045% | +5.202% | +6.438% | +6.409% |
| R1固定源 | +4.910% | +4.893% | +7.249% | +7.078% |
| 混合固定源 | +2.915% | +2.831% | +4.854% | +4.776% |
| 同来源旧状态 | +1.446% | +1.384% | +5.280% | +5.086% |
| 纯学生状态 | -72.238% | -72.897% | -12.728% | -12.519% |
| matched/pure各半 | -70.932% | -71.581% | -9.176% | -9.046% |

旧能力退步并非只发生于验证集；纯学生训练收益远大于验证收益也保留。不能从这张表直接证明标签矛盾、不可兼得、算法收敛或f0不可能。

原最前四训练批次，起点旧/pure梯度cosine为-0.20935/-0.07762/+0.05332/-0.03670，3/4负、均值-0.06759，属弱平均冲突。pure梯度范数约旧目标7.4至12.4倍；起点保持项梯度全为0，符合“输出精确等于被保持输出”的MSE定义，不是读取失败。实际总目标与旧目标仅1/4批方向相反，不能说所有更新都牺牲旧能力；两末轮总目标与旧目标在这四批均正相关，亦不据此认定训练已收敛。位移点积仅局部一阶近似，不能替代3000次实际更新的误差变化或因果解释。

峰总10739380224B/GPU518MiB/OOM0，末总8266432512/限34292629504B、GPU1MiB25度、磁盘余183396876288B，原暂停/旧五PID退/锁及计算空。30秒记原r2，累计4820=1614+1835+1371，无RESERVED；总余9580、r2余2665秒，预算未重置。

## 图表与原始证据

27359B/11成员原始数值包SHA `f61fc9a53924e10e38863d20459915ee739fefa2f0c35932c5a1e215a9174a8b` 已Mac分片/整包/逐成员核同。新六宫PNG2560×1600、PDF、CSV与训练/验证/梯度完整证据ZIP332703B、18成员、17相对链接CRC/逐文件大小SHA全过，PNG已目视无重叠。

- [完成及回收原始记录](../logs/session_records/q1_train_objective_completed_20261108.json)
- [图表与ZIP清单](../data/manifests/q1_train_objective_evidence_20261108.json)
- Mac图：`/Users/yu/Documents/ChatGPT/元启/data/analysis/Q1_TRAIN_OBJECTIVE_FIGURES_20261108/objective_six_panel.png`
- Mac证据包：`/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_TRAIN_OBJECTIVE_EVIDENCE_20261108.zip`

绘图入口`script/supermimic_teleop/deliver_train_objective.py`仅读保存工件，不在五执行文件清单内；CLI和实际绘图/数值核验通过。validation只复用上一轮原始trace，没有新validation前向、梯度、选点或未来状态输入。

## 下一独立机制

现有残差仅输入930维机器人历史，人体现时/历史仅经过冻结人encoder进入base latent。下一优先检验“残差直接条件于已可用的人体意图”这一假设；本诊断不能证明它是根因。拟在实际起点四张量机器人分支旁加零初始化人体latent分支，控制组喂零、实验组喂同一冻结人encoder已因果可用的latent，两组参数容量/初始化/数据曝光/目标一致。不是直接移植GigaBrain奖励，也不称已获得其效果。

下一方案尚未实现或派发；须先独立来源/可见性/同初始化输出及梯度/成本审计，保留原三验证不退门和选0、teacher/WM/decoder/归一化/物理冻结，再发布、原运行时与资源准入后有界执行。选回不变策略则CPU核同复用旧0/7；非零候选才独立真实闭环及每训练六宫图。严格全身同目标跟随与“当前上肢加自主踏步”分开，后者不能替代原目标。真实人体留出、硬件端到端延迟、自然性仍待。

## 保存与同步

实现/来源/准入均已发布，结果Git/独立远端/Mac和小记录服务器镜像以后续publication为准；旧大索引服务器镜像仍PENDING。没有新agent或goal工具，采纳null。q1后续恢复入口更新只管理本聊天，其他自动任务不改。
