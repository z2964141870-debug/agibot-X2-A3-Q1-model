# YUANQI 当前状态与下一步

2026-10-26恢复入口：[HPR01双向源码核查](README_SUPERMIMICHP_Q1_REVIEW_20261026.md)和[RF07根反馈对照](README_SUPERMIMIC_Q1_RF07_ROOT_FEEDBACK_20261026.md)。确认本项目actor直接根位置/朝向误差缺失，独立启用原可微anchor接口，goal540→549；V02现场6测试/真实CPU评测与算术审计通过，0optimizer。RF07待发布部署后启动，保留RF06腿部控制，软3000秒/最多3600秒，不称效果已改善。旧MG02完成，不重复创建目标。师兄HEAD310244未变；短片无合法H32起点时无限采样已实际函数有界复现，非其当前训练失败证据。每小时跟踪已更新为双向核查/验证后采用，不对外代发问题。

2026-10-26，北京时间。MG02的A至E计算、对照和本地媒体交付完成，结果记录已发布Q1并独立核远端373088b2d75bd50827f0c857b005d9ba6b826631，16文件232478字节在Mac/25252逐大小SHA一致。[最终要求审计](../data/manifests/supermimic_q1_MG02_completion_audit_20261026.json)17项全部通过。RF05/RF06均全pipeline COMPLETED，controller/worker已退出，q1-2048训练心跳已暂停，未启动RF07。

## 最新结果

[RF06腿部限矩对照](README_SUPERMIMIC_Q1_RF06_LEGS50_20261026.md)实际训练3132.859650秒（52分13秒），600轮/2457600转移、WM189/actor139，满足最新最多3600秒预算。12腿关节限矩25%→50%，其他控制/动作尺度/奖励/同参考与seed保持，从零。18:21:52全部10阶段完成，13项保存检查及初末算术身份核验通过；最终365步不倒，但只移动1.49cm/参考1.60m，沿目标方向-1.49cm，双支撑98.904%，仅一帧右单支撑，上肢RMSE0.69212rad，未形成行走或完整抬手。初始策略也365步不倒，不能把站住全部记为训练收益。

[RF05/RF06只读比较](../data/manifests/supermimic_q1_RF05_RF06_comparison_20261026.json)共同365步，0新积分/推理/更新。RF05实际7237.63秒/4608000转移，净移1.87cm、双支撑99.178%、上肢RMSE0.68042rad；RF06未见行走改善。两轮时长和更新数不同，不称等时训练消融。[MG02证据表](README_SUPERMIMIC_Q1_MG02_LEG_GOAL_20261026.md)保留A至E完整范围、最新一小时预算和失败边界。

## 本地交付

RF06在Mac对应data/supermimic_for_q1/BONES_RF06_UP01_E4096_LEGS50_MAX01H_20261026/的PNG/PDF/CSV、初末MP4及六张代表帧共11文件1706855字节，逐大小/SHA通过；RF05原11文件1723401字节再次核验通过。六宫格129条CSV记录独立复算，原值最大差0、500训练iteration均值最大差1.11e-16，已目视检查，无需额外图片加工。

[最终MP4](../data/supermimic_for_q1/BONES_RF06_UP01_E4096_LEGS50_MAX01H_20261026/render_final/comparison.mp4)；[初始MP4](../data/supermimic_for_q1/BONES_RF06_UP01_E4096_LEGS50_MAX01H_20261026/render_initial/comparison.mp4)；[训练图](../data/supermimic_for_q1/BONES_RF06_UP01_E4096_LEGS50_MAX01H_20261026/run_dashboard/training_dashboard.png)。权重及大轨迹只留25252服务器LOCAL_ONLY，没有网站或网盘备份声明。

## 恢复约定

权威Git为/tmp/yuanqi-supermimic-q1-20261023的Q1分支。正式计算只用用户自己的25252容器、唯一根/opt/data/private/ZHY，沿用data/logs/script/reports。SSH为 `ssh -S /tmp/yuanqi-q1-25252.sock -T -p 18869 -o BatchMode=yes -o ConnectTimeout=8 -o 'HostKeyAlias=[59.77.15.15]:25252' root@127.0.0.1`。Python为data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python。保留Windows桥，不读取或写入认证材料，不开子agent。以后每轮真实训练最多1小时，不自动延长。

RF06冻结代码0b1da033148fff0b705fd85cdac5a773b5a7f332；报告提交不是运行源码改变。最新终后观测controller3301/worker3304均不存在、boot d994e71f-bae0-46ba-a5f7-6c2163771c6c、GPU14MiB/26度/OOM0、共享内存4941115392字节/余盘1788185260032字节，RF06分配277934080字节。资源是观测时快照，后续使用前重新核实；不重做迁移/hash/profile或已完成评测。

## 仓库跟踪与下一步

2026-10-26 19:00巡查：SuperMimicHP最新310244b73e45775744a75b789eb25a6c9c5a81c9只发布已有24k演示/文档，src/tools及原结果JSON未变。随后用户新增双向核查与积极采用要求，HPR01/独立RF07为新授权工作，不能沿用旧“仅等待”范围。

[师兄SuperMimicHP仓库](README_SUPERMIMICHP_WATCH_20261026.md)独立每小时心跳supermimichp-x2保持，有实质技术变化才通知。HPR01确认X2实际1024环境/100条SOMA/mjwarp、逐关节PD/full effort限矩；本项目未来目标信息及实际优化次数也有差异。RF07先只加当前根反馈，不把源码差异当唯一根因，不照搬X2控制数值；未来目标/片尾H32/20ms时序/吞吐分别留待后续验证。

q1-2048目前PAUSED，待RF07真正启动后复用为其监督，不恢复旧RF06 prompt；supermimichp-x2和yuanqi仍ACTIVE。此前详细时点记录完整保留在[当前状态历史快照](README_CURRENT_STATE_ARCHIVE_20261026.md)，旧RUNNING、旧服务器和旧两小时RF06预算不能当作当前执行依据。报告发布身份由Git历史提供，避免将新报告提交误作冻结训练版本。
