# MorphMimic 与 PrimebotTech SuperMimic-Q1 接入

2026-11-02，用户在完成 X2/T2 发布策略回放后要求继续尝试 Q1 训练，并提出结合 PrimebotTech SuperMimic-Q1 与 MorphMimic。本阶段落实这一授权，不创建新应用目标，不改变其他自动任务。

## 方案与边界

- 使用 MorphMimic v0.1 的共享世界模型、策略/critic、独立机器人 replay 和同源同帧 latent 对齐；先做 G1+Q1 两本体接入。新的共享网络从零初始化。
- Q1 使用 PrimebotTech SuperMimic 的分组软件 PD、50 Hz 控制、2 ms 物理步长、位置目标延迟及转速限矩契约，资产使用已核的 Q1 V3 力矩 motor 模型。该库示例指向外置 V2，V3 接入须单列版本差异，不能称完整复刻其未随库分发的资产。
- 对照为相同 Q1 资产、数据和控制契约的 Q1 单体训练。共享训练必须记录每个机器人实际转移数、optimizer 次数及共同动作；仅训练 Q1 或先后训练独立模型都不算 unified。
- 数据由同一批 SOMA 分别做真实 G1/Q1 重定向；旧 AP03 与共同 1h 的 559 条仅有 9 条同源交集，不能直接作为完整成对输入。衣服演示仍搁置。
- 仍保留原算法奖励，先核按尺寸缩放的具体位置、避免重复缩放；不把两个代码库的奖励权重直接相加。短期不追加 PPO 残差实验。

## 已查明的接口差异

MorphMimic 原版统一 29 关节、14 跟踪 body、240 维状态、93 维 proprio、670 维 motion goal。Q1 V3 实际只有 22 个 hinge，缺少 `waist_pitch_joint` 和左右各三个腕关节，但固定腕 body 仍可用于跟踪。

原 `Translator`、action 的 `perm` gather/scatter 和 morphology 的 log 执行器特征均假设 29 槽真实存在。直接填 `-1` 会错误引用最后一个真实关节；仅在 motor 输出端丢弃多余动作也会让世界模型利用不存在的自由度。因此需要同步约束参考、动作/探索噪声、last_action、观测/encoder bias、世界模型预测和训练误差。浮点缺失槽用显式选择置零，不能依赖 `NaN * 0`。

拟将缺失关节 mask 明确加入 morphology，形成 154 维新契约（原 125 + 29 mask），并仅使用存在槽拟合 min/max。此架构不兼容发布 ONNX 的 125 维 morphology，不能据发布策略已验证就宣称 Q1 零样本可用。

自训 G1 旧 aa40 版本没有 morphology 且 KP goal 版本为 2；v0.1 为 10。PrimebotTech 没有随本次固定源码提供 Q1 训练 checkpoint，MorphMimic 发布包只有部署策略，不能恢复完整 WM/critic/optimizer 共享训练。

## 固定来源

- MorphMimic v0.1：`41c8382f127a6f5109bfe047445a9180f71f1fc0`。
- PrimebotTech/SuperMimic：`118338b19dcd14f768d1e7de5a48d04b78987688`。
- SOMA retargeter：`cbdb2b494a2af82137bf9b6f15122d60167281c4`。
- 服务器资产：`/opt/data/private/ZHY/data/references/q1/model_20261023_P01/q1_v3/xml/q1_v3_mc.xml`。
- 小源码审阅副本：`data/references/morphmimic_q1_review_20261102/`，仅审阅，不修改原冻结树。

## CPU 接口验证结果

准备提交 `a3be66d419be7d5b5bcbb904dc4adf73bc22131d` 已独立核远端 Q1，9 文件 415585 字节在权威 Git、Mac 主根、25386 逐大小/SHA 相同后执行。2026-11-02 19:33:47 CPU 实模核验完成，结果见 [模型契约](../data/manifests/morphmimic_q1_cpu_contract_20261102.json) 和 [12 项测试](../logs/session_records/morphmimic_q1_cpu_tests_20261102.log)。

- 12 测试全部通过，包含缺失槽 NaN 不进入真实关节、缺失动作梯度为零、腰 joint 顺序重排、signed/offset 一般映射、limits、状态/观测/goal/context mask、真实关节误差分母和 descriptor 缺失槽边界。
- Q1 V3 为 `nq=29/nv=28/nu=22/nbody=27/ngeom=71`；14 个跟踪 body 存在。64 组原生关节姿态的 canonical 往返误差及静态 FK body 位置变化均为 0。
- 22 关节零位正轴与 G1 方向余弦最小约 0.93969，两侧零位前臂方向余弦约 0.99965，支持此模型使用正号、零 elbow offset 的候选映射；不沿用 X2 的 elbow `-pi/2`。这不是两个机器人 FK 相等或任意动作可达的证明。
- 固定 Primebot 源码通过 AST 只加载两组 PD 表和三个 tensor 函数，未实例化其完整 simulator。128 组速度/目标 × 10 子步 × 22 motor 共 28160 控制数值与源公式最大差 0；涵盖按方向转速限矩、延迟目标选择、gear 和 XML ctrl bounds。尚未验证 mjlab 原生 actuator 回调、随机化 encoder bias/PD 因子或求解器轨迹等价。
- 154 维 descriptor 仅验证缺失槽和缩放算法，实模测试中的 geometry 使用人工全 1，不能当作 Q1 的实际形态向量；真实几何和 G1/Q1 训练群体边界待提取、冻结。
- 五个明确列出的只读来源前后 SHA 相同，未建立 CUDA context；新增动力学积分、策略推理、optimizer 均为 0。19:35:49 原 boot/GPU 核验相同，GPU 15 MiB、无计算进程、OOM0。

实现位于 `script/morphmimic_q1/`，没有注入原冻结 Trainer。原始 CPU JSON/log 在服务器与 Mac `logs/` 保留；回传小包 3614 字节，SHA256 `af9930ae25489099eff1e256f0010dfe3b2faed4cfeb77b2a787786e1b398344`。

## 当前进度与下一步

独立缺失关节模块与 CPU 契约验证已完成；尚未注册 Q1 到原 Trainer，尚未新增重定向、物理 rollout 或训练。完整结果写入 `data/manifests/morphmimic_q1_integration_20261102.json`。当前资产 XML 自带 1 ms timestep，示例要求 2 ms，接入时须显式设置并核实 10 个子步；当前 CPU 核验没有修改 XML。

延迟还涉及世界模型状态契约：Q1 源控制会在子步中使用上一时刻位置目标，原 MorphMimic 的 33 维 context 只含摩擦、encoder bias、CoM offset。必须明确延迟值和上一目标在模型中的可见性及一步更新规则，不能只把 PD 换掉就宣称完整接入。奖励/终止按尺寸缩放和 22 关节误差权重也须在仿真与 imagination 两侧一致；CPU bridge 中的真实关节 RMSE 不代表训练损失已经替换。

按顺序完成：缺失槽/梯度与真实模型 FK/控制契约验证；Q1 专用 SOMA 配置和成对轨迹；原生 Trainer 两 rank 同 GPU 的通信、资源及五 optimizer 更新验证；冻结实际训练配置并发布核验，再启动足量训练及固定原速视频评估。环境规模和迭代数由真实共享资源测试确定，当前不预填已通过数值。沿用 24/30 GiB、boot/GPU、OOM、GPU 锁和温度守卫；不套用旧 300 秒退步门或 1200 秒短预算。

用户采纳仍为 `null`；本阶段没有 Q1 步态改善结论。
