# Q1训练目标只读诊断

日期：2026-11-08，北京时间。实验：F0_TRAIN_OBJECTIVE_AUDIT01。状态：独立实现、原运行时六合同和1229来源/资源准入通过，真实GPU诊断尚未派发。

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

## 同步与下一步

本次Git发布、Mac镜像和服务器部署以后续收据为准；旧大索引服务器镜像仍PENDING。没有新agent、goal工具或其他自动任务变更，采纳null。先完成只读诊断和图表，再决定是否有不同机制值得真实检验，不把本阶段当全身自然遥操目标完成。
