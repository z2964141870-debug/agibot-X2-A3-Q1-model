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

## 当前进度与下一步

接口核查完成，开始独立缺失关节适配模块与 CPU 契约验证；尚未注册 Q1 到原 Trainer，尚未新增重定向、物理 rollout 或训练。完整结果写入 `data/manifests/morphmimic_q1_integration_20261102.json`。

按顺序完成：缺失槽/梯度与真实模型 FK/控制契约验证；Q1 专用 SOMA 配置和成对轨迹；原生 Trainer 两 rank 同 GPU 的通信、资源及五 optimizer 更新验证；冻结实际训练配置并发布核验，再启动足量训练及固定原速视频评估。环境规模和迭代数由真实共享资源测试确定，当前不预填已通过数值。沿用 24/30 GiB、boot/GPU、OOM、GPU 锁和温度守卫；不套用旧 300 秒退步门或 1200 秒短预算。

用户采纳仍为 `null`；本阶段没有 Q1 步态改善结论。
