# RF15夜间补偿首门结果

2026-10-29 02:31，北京时间。训练在1203.610393实训秒检查后按physical_regression正常停止，270轮/26542080转移、critic5400/actor5200小批更新、0新WM/想象。全pipeline已02:10:47 COMPLETED；02:12:17核controller137006、训练worker137016和交付worker139983均不存在、GPU15MiB无计算进程，双guard WORKER_COMPLETED/reaped0。11项保存审计执行一次全通过、1163来源未变，24项Mac媒体检查及12联系图/3图目视完成。冻结e3601552863084d5f52758e6c5f07ea722dafbf8，不恢复/重试此训练。

## 定量结论

原300训练/30验证与固定两普通走/抬手，完整330片三重复。上肢/腕/根的表格为整条可用验证轨迹范围；门控另用同重复共同前缀，不把两者混为同一数值。

| 候选 | 训练失败/300 | 验证失败/30 | 上肢RMSE rad | 相对腕RMSE m | 固定抬手峰误差m |
| --- | --- | --- | ---: | ---: | ---: |
| 冻结RF14两小时纯底座 | 9/7/5 | 1/0/0 | 1.0156–1.0192 | .1554–.1558 | .3409–.3411 |
| 未训练静态手臂alpha.5 | 9/9/12 | 0/0/0 | .2960–.2969 | .0835–.0841 | .0707–.0709 |
| RF15首门head | 11/9/11 | 0/0/1 | .2950–.2967 | .0824–.0831 | .0727 |

三版本两普通走/抬手三次都完整，仍不能只凭完成称改善。head使“走后停”滑移.1310–.1886m/s，对静态.0852–.1260；共同前缀三次比值2.2127/1.0389/1.6563，三次中两次超过原1.5倍严重退步阈值。该片目标变化约增45.8%/下降17.7%/增44.9%。另一普通走滑移部分改善，但不能用平均掩盖停走恶化。严重退步2/3、严格改善0/3、本夜继续门0/3，故停止；不是因根XY阈值或跌倒停止。

固定抬手仍完整，漂移.0759–.0764m对静态.0767–.0786，峰高误差略增。上肢主要收益明确来自静态参考混合，head尚未学到可靠的新步态收益。KL限率与冻结底座没有保证防滑；这20分钟结果不证明已隔离RF14退化单因，也不是传统“训练集拟合好而验证差”的证据。

数值推荐initial_static，用户采纳仍null。保留final.pt与失败轨迹SERVER_LOCAL_ONLY，不覆盖静态候选。余下RF15预算不用于自动重试或恢复。

## 本地审阅与交付

Mac根/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF15_GAIT_RESIDUAL_SOMA300_E4096_OVERNIGHT_20261029。review/stage_01/README.md、四CSV和summary.json已由保存JSON生成，108训练/验证行、27固定片、63跟踪共同前缀、72逐走共同前缀行；11小原始工件1659618字节Mac/25386逐大小SHA一致，0新物理/推理/optimizer。

原服务器50工件47030912字节已复制Mac逐大小SHA一致；Mac新增21工件7770893字节已回镜25386逐大小SHA一致，共71工件54801805字节。权重/大NPZ仍SERVER_LOCAL_ONLY。review/final保留108训练/验证、27固定片、63跟踪/72逐走共同前缀行，initial和stage_01版本保留。六格有原值和500iteration均值；仅270轮，均值使用可用的部分窗口，WM/想象两项N/A，另有PPO/KL/奖励分解图。

可直接打开本地视频：

- [纯冻结底座](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF15_GAIT_RESIDUAL_SOMA300_E4096_OVERNIGHT_20261029/evaluation/base.mp4)
- [静态手臂alpha.5](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF15_GAIT_RESIDUAL_SOMA300_E4096_OVERNIGHT_20261029/evaluation/initial.mp4)
- [学习补偿后](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF15_GAIT_RESIDUAL_SOMA300_E4096_OVERNIGHT_20261029/evaluation/final.mp4)
- [数值推荐静态候选](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF15_GAIT_RESIDUAL_SOMA300_E4096_OVERNIGHT_20261029/evaluation/recommended.mp4)

各54秒/1350帧/25fps、1280x480，固定8段6类型原速合集，不是连续54秒rollout。静态/学习/推荐中的舞蹈提前终止保留hold标签；普通走/抬手没有因训练后表现换片。base.mp4通用叠字后缀写有gait compensation，但实际为纯冻结HP轨迹，以本段和JSON标签为准，不改已冻结视频工具。目视可见静态混合改善抬手幅度，head未明显修复走路，低抬脚/姿态偏差仍在；媒体验收通过不等于自然步态合格。

[交付/终态/大小SHA收据](../data/manifests/supermimic_q1_RF15_local_delivery_20261029.json)；[首门原值](../data/manifests/supermimic_q1_RF15_stage01_20261029.json)。既有保存和媒体审计各已执行一次，复用，不重复旧RF14收尾或新优化。

## 夜间后续

本次负面证据与静态手臂候选已交齐。独立[参考相位腿部对照](README_Q1_PHASE_LEG_PROBE_20261029.md)02:13至02:22完成六候选/三重复/33环境、0optimizer；五档干预均未给两普通走带来可重复收益，不据此开full330或新训练。09:00探索窗口仍有效；下一步检查高频指令与控制阻尼假设，尚未启动新试验。07:00停止新增优化/物理探索、09:00交付，之后用户决定采用及下一步；不恢复RF15、不热改冻结来源。
