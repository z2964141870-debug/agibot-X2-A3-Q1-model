# RF14最终定量结果与审阅交付

实验日期：2026-10-28；交付验收更新：2026-10-29 00时。状态：RF14训练、全pipeline、保存审计及本地数据/视频交付完成，基本展示未达标，等待用户审阅决定。完整RF14/RF15目标仍active，RF15未预检/训练。

## 目标与决定边界

本轮按原RF11配方继续底座，在累计约1/2/4小时核普通走的滑移、抬脚、姿态和稳定性。用户要求先看实际数据和仿真视频再决定，数值排序仅推荐，不代替用户采纳底座或授权RF15/延长。goal与原监督保持active，RF15未预检/训练。

## 已完成的最终数据

最终累计14300.406953636484实训秒，含RF11原3500.265989576001秒；新增10800.140964060483秒。12665学习轮、130211840转移、WM126650/actor12165；4096环境、原300训练/30验证，冻结`4d87269d7ac815cd75a0424dcf7be56ae2c2661d`。run为TRAINING_COMPLETED/outcome累计预算完成，训练guard正常reap0，原worker110573已不存在、GPU15MiB无计算进程。23:01:21现场pipeline为TRAINING_COMPLETED_PENDING_DELIVERY，controller110563与交付worker129421仍在，不能提前称全流程结束。

| 指标 | RF11初始 | 两小时 | 最终约四小时 |
| --- | --- | --- | --- |
| 验证提前终止/30（三次） | 0/0/0 | 1/0/0 | 4/6/4 |
| 验证平均帧完成比例 | 100% | 99.45%至100% | 94.72%至95.21% |
| 固定走后停完整重复 | 3/3 | 3/3 | 0/3，97/123/125步，目标320 |
| 固定起步走完整重复 | 3/3 | 3/3 | 0/3，149/151/154步，目标210 |
| 固定抬手完整重复 | 3/3 | 3/3 | 3/3，均271步 |
| 抬手根净漂移 | .0115至.0119m | .0856至.0895m | .1011至.1015m |

最后与各自同重复RF11共同前缀：验证上肢RMSE降低约10.1%至10.6%，但相对腕增加5.3%至7.0%、腿增加30.0%至32.0%、指令变化增加53.7%至61.8%、承重滑速增加30.4%至59.0%；根XY增加9.2%至32.0%只作诊断。固定走后停滑速.429至.601m/s，倾斜RMS .360至.517rad，不能把摔倒过程中的前进距离/脚高当自然步态改善。完整原值/共同前缀分别保留，不按失败前可用时长混称等时长改善。

抬手双腕峰高相对骨盆约左.333至.334m/右.583m，参考.534/.560m；平均峰高误差.112m，较初始.356m减小，但两腕峰都在5.38秒，参考1.42/1.72秒，时序仍明显迟滞，不能仅凭峰高改善称上肢动作正常。右腕接近参考峰不代表双臂/全程跟踪达标。

最终阶段严重停止门未触发（仅一次验证失败>=6），按预算正常结束；不把它改记physical_regression或流程故障。数值排序RF11_initial优先、stage03 rank[20,0,7.1179786]仅如实保留，用户尚未选择。最终普通走展示明显退步，RF15没有因此自动启动。

## 交付路径与验证状态

Mac：`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF14_RF11_CONTINUE_SOMA300_E4096_CUM04H_20261028`。`review/final_quantitative`已用12份保存JSON形成`validation.csv`72行、`fixed_demos.csv`36行、`common_prefix.csv`63行及`summary.json`/`README.md`；这是定量结果快照，媒体尚未就绪。原JSON和失败步数在`evaluation/`，权重/NPZ仍SERVER_LOCAL_ONLY。

自动初末/数值推荐视频及六格PNG/PDF/CSV已完成；stage01/stage02预定repeat0已存轨迹另导两片视频，0新物理/推理/optimizer。五片各54秒、1350帧、25fps、1280×480，原速8段6类型合集，非连续54秒rollout；固定片与失败hold保留。Mac三份media_audit共30/30检查通过，15个联系图与六格已目视，完整性核验不等于正常步态认证。

| 本地视频 | 对比重点 |
| --- | --- |
| [RF11初始](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF14_RF11_CONTINUE_SOMA300_E4096_CUM04H_20261028/evaluation/initial.mp4) | 两固定走完整，上肢幅度不足 |
| [累计约一小时](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF14_RF11_CONTINUE_SOMA300_E4096_CUM04H_20261028/evaluation/stage_01.mp4) | 比初始仅新增100秒；走后停271/320步跌倒 |
| [累计两小时](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF14_RF11_CONTINUE_SOMA300_E4096_CUM04H_20261028/evaluation/stage_02.mp4) | 两走完整/有前进，仍有姿态、手臂及时序问题 |
| [最终约四小时](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF14_RF11_CONTINUE_SOMA300_E4096_CUM04H_20261028/evaluation/final.mp4) | 两条固定走跌倒，抬手迟滞 |
| [数值推荐，仅供审阅](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF14_RF11_CONTINUE_SOMA300_E4096_CUM04H_20261028/evaluation/selected.mp4) | 原RF11候选，不代表用户已采纳 |

最终[对照表](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF14_RF11_CONTINUE_SOMA300_E4096_CUM04H_20261028/review/final/README.md)含72验证原值、36固定片行、63共同前缀行，summary的用户接受字段为null；[六格](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF14_RF11_CONTINUE_SOMA300_E4096_CUM04H_20261028/dashboard/training_dashboard.png)保留原值和500iteration均值。40服务器工件70934180字节Mac逐大小SHA一致；33个Mac审阅/解码/联系图9666866字节回镜服务器并逐大小SHA一致，共73工件80601046字节。权重/大NPZ未拉Mac，仍SERVER_LOCAL_ONLY。[交付清单](../data/manifests/supermimic_q1_RF14_local_delivery_20261029.json)。

2026-10-28 23:15:26全pipeline正常COMPLETED；2026-10-29 00:02:05复核原controller110563、训练worker110573、交付worker129421均不存在，GPU15MiB无计算进程。双guard正常reap0，现有saved_evidence_audit的12/12项通过、1145来源未变、末阶段与最终模型/优化器一致，无新优化。[审阅就绪收据](../data/manifests/supermimic_q1_RF14_review_ready_20261029.json)。上文23:01待交付仅为历史快照。

针对“是否过拟合”，[证据诊断](README_SUPERMIMIC_Q1_RF14_DEGRADATION_20261028.md)确认两小时训练失败7/5/5对最终81/81/83，验证同步变差；更支持后期策略退化，未隔离模型偏差、奖励取舍或更新稳定性的唯一原因。不能因模型训练误差下降/即时奖励上涨就认定实际控制改善。

## 来源与下一步

[最终原值收据](../data/manifests/supermimic_q1_RF14_stage03_20261028.json)保存三重复逐片、共同前缀、真实预算和当前终态边界。RF14当前冻结来源不改，保存审计须等pipeline COMPLETED及双guardreap0再执行。重要阶段按Q1 payload audit/commit/push/独立40位远端及Mac/25386镜像核验发布，当前提交号由Git历史提供，不预先称三端完成。

数据、五版本MP4、六格、保存/媒体验收已齐备，先交用户审阅。原监督已工具更新并读回ACTIVE/每10分钟，等待期间无变化静默，不重复渲染/审计或旧收尾。未获用户明确选择/启动决定前，不启动RF15或GPU预检、延长、重试或agent，不自行结算或暂停完整goal。RF15旧自动提cap实现不能直接启动，后续获明确决定后须先完成审阅等待机制的实现和验证。

