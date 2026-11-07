# CPU状态转换：等价检查通过，约减少94%处理时间

2026-11-08，北京时间。低延迟SuperMimic的一项软件改进；原共享训练保持暂停。

done：在原策略真实运行时上，对前走128、起走106个保存状态交替比较旧Torch运动学和新隔离MuJoCo运动学。269状态字段、93观测和930连续历史全部通过原容差，最大状态差5.72e-6，最大观测/历史差1.91e-6。五个坐标与不可变性合同也全部通过。

| 去每片前两调用的CPU转换 | 前走 | 起走 |
| --- | ---: | ---: |
| 旧→新中位，ms | 6.20→0.37 | 7.16→0.40 |
| 旧→新p95，ms | 13.01→0.76 | 16.29→0.98 |

这项处理约快17倍，但完整worker、RPC和硬件端到端尚未复测。它不减少人体未来帧需求，也不证明f0闭环自然；旧七例失败结论保持。A022首次解析观测/历史89.38ms尖峰仍保留在全量证据，没有用稳态数字掩盖。

本轮没有策略前向、物理回放、渲染或训练，GPU累计预算仍2433/14400秒。首派遗漏既有Git路径而在导入时退出，原FAILED保留；补齐环境后在独立R2根28.25秒完成，未重做已完成状态或旧评估。

todo：先用少量GPU调用核动作等价及完整软件链，再将加速转换用于新H4时间mask/闭环实验。原worker和动力学不改，原训练不恢复，采纳仍为null。

两张六宫图、两PDF和计时CSV已在Mac逐件核验；[完整记录](README_Q1_CPU_STATE_20261108.md)、[前走图](../data/analysis/Q1_CPU_STATE01_FIGURES_20261108/walk_forward_loop_001__A022.png)、[起走图](../data/analysis/Q1_CPU_STATE01_FIGURES_20261108/walk_forward_start_002__A023.png)、[原始计时表](../data/analysis/Q1_CPU_STATE01_FIGURES_20261108/cpu_input_timing.csv)。完整269维数组仍在服务器，两图证据包不含新视频；[证据包](../data/deliveries/Q1_CPU_STATE_EQUIVALENCE_EVIDENCE_20261108.zip)。

证据包805214B，SHA-256为`2f478c01e817456c46bc7dd8fd1fce340ae701071874edd71908d422312c46f2`，20成员CRC、逐件大小/SHA和10个相对链接均通过；[交付核验](../logs/session_records/q1_cpu_state_result_20261108.json)。
