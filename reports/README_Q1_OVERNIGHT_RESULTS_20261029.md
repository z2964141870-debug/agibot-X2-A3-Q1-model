# Q1夜间探索：早间数据与仿真视频

2026-10-29，北京时间。本夜物理/训练已结束，09:00前完成审阅材料。上肢展示有明确改善，普通走的稳健自然步态仍未解决；不把局部收益包装成整体成功。所有候选尚未被用户采纳。

## 先看这两组视频

| 方案 | 同协议30条验证上肢RMSE | 固定抬手峰高误差 | 固定两走及抬手 | 验证提前终止/30 |
| --- | --- | --- | --- | --- |
| 静态alpha.50 | .2962-.2966rad | .0707-.0710m | 各3/3完整 | 0/0/0 |
| 按参考活动量动态alpha.50-.75 | .1943-.1954rad | 约.0113m | 各3/3完整 | 1/0/0 |
| 静态alpha.75幅度对照 | .1427-.1432rad | .0112-.0113m | 各3/3完整 | 0/0/0 |

动态方案相对静态.50，上肢误差约降34%，抬手峰高误差约降84%；走路时混合较低，静止抬手时渐进提高，规则只读参考速度/heading，不按片名切换。它是零优化控制组合的效果，不是新网络训练收益。静态.75上肢更好但训练集失败19/18/17条，动态10/12/11、静态.50为9/11/8；动态也存在全库稳定性代价。三次是同策略配置重复，不是独立训练。

- [静态.50对照MP4](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_ACTIVITY_BLEND_FULL330_REVIEW_20261029/evaluation/baseline.mp4)
- [动态.50-.75候选MP4](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_ACTIVITY_BLEND_FULL330_REVIEW_20261029/evaluation/activity050075.mp4)
- [静态.75幅度/步态取舍MP4](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_ACTIVITY_BLEND_FULL330_REVIEW_20261029/evaluation/static075.mp4)
- [2970逐片原值、固定片与共同前缀](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_ACTIVITY_BLEND_FULL330_REVIEW_20261029/review_v01/README.md)

动态停走承重滑速.0901-.1150m/s，起走.1187-.1420m/s；相对静态没有两片、三次一致的步态收益。抬手净漂移仍约.077m，普通走仍有低抬脚和滑步，不能称“正常步态已训好”。原30验证参与了搜索，不是独立保留测试。

## 两次4096环境训练

| 实验 | 实训时间 | 实际改动与结果 | 状态 |
| --- | ---: | --- | --- |
| RF15 | 1203.610393秒 | 冻结RF14两小时教师，学习腿腰残差；原生接触/摆脚/平顺/静止奖励及较小PPO更新。停走滑移三次中两次严重恶化，按physical_regression门停 | 正常规则停止、全流程已交付，不恢复 |
| RF16 | 1201.120094秒 | 仅接触相位特征/分段换成原重定向接触意图，其他控制/数据/系数固定。601秒停走滑速降16%-24%，起走一次增23%；1201秒未保持局部收益 | 正常短预算完成、全流程已交付，不扩时 |

合计新增物理实训约2404.730487秒，即40分4.73秒；初始化、评测、保存和媒体不计其中。未为凑时长消耗原RF15剩余预算。实际学习都使用4096真实MuJoCo Warp环境、原300/30数据、RSL-RL PPO，不改旧实验、不新增WM或想象更新。

RF16最终验证0/1/0失败，固定两走/抬手各3/3完整，但整组步态改善各阶段0/3。数值rank推荐601秒stage02，仅用于并排审阅，未采纳，也没有与动态上肢方案叠加后冒充已测结果。

- [RF16未训练静态起点](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/evaluation/initial.mp4)
- [RF16约601秒候选](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/evaluation/recommended.mp4)
- [RF16约1201秒最终](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/evaluation/final.mp4)
- [RF16六格训练曲线](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/dashboard/training_dashboard.png)
- [RF16全部4950逐片及固定片原值](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/review/final/README.md)

各视频都是54秒/1350帧/25fps、固定8段6类型原速合集，不是连续54秒rollout；普通走和抬手不事后换片，舞蹈等失败保留hold标记。左侧是真实物理轨迹，右侧是运动学参考。六格为原值和500训练iteration均值，PPO/奖励另附，WM/想象N/A。

## 进一步定位到的具体问题

精确脚底复核使用实际碰撞mesh/sphere和同一地面，而不是踝高。RF16的601秒候选，停走右脚脚底高度p95只有3.93-4.68mm、参考22.41mm；起走2.23-4.57mm、参考25.49mm。右脚多数时间没有可靠摆脚离地，低滑速可以与这种状态同时出现，因此单独“防滑奖励下降”不能算学会走路。

[脚底六格曲线PNG](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/review/sole_plot_v01/collision_sole_clearance.png)；[72足原值CSV](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/review/sole_v01/fixed_actual_sole.csv)。这些是保存姿态CPU分析，0新增积分/推理/optimizer。图中离地段数不当正确步数，低脚底高度也不能独自解释所有训练退化。

## 其他假设已做过哪些对照

| 假设 | 实际核查 | 边界与结论 |
| --- | --- | --- |
| 参考腿目标前馈能否直接改善 | 6档参考相位/摆脚quick33三重复，0optimizer | 无两普通走可重复均衡收益 |
| 高频目标是否主因 | 轻EMA、强EMA、限速等6档quick33 | 轻滤波单片可能改善；强限速明显损害稳定，不能直接套用 |
| PD刚度/阻尼是否不足 | 6档PD、原模型独立副本、16环境增益预检和quick33 | 较高刚度可降滑移，同时足高更低、倾斜变大 |
| 大幅度上肢是否必须持续干扰步态 | 5档quick33和3候选full330 | 动态参考混合带来可复现上肢收益，步态未稳健解决 |
| BF16是否造成越训越差 | 同冻结权重BF16/FP32×静态/动态四档quick33 | FP32没有两走一致收益，不能归因为BF16 |
| 重定向脚锁是否整体失效 | 原330参考/metadata接触、材料点与碰撞脚底CPU审计 | 声明支撑点速度约.0005-.0067m/s，不支持整体脚锁失效；声明相位不是实际原生承重真值 |

完整入口：[RF15](README_SUPERMIMIC_Q1_RF15_RESULTS_20261029.md)、[参考腿相位](README_Q1_PHASE_LEG_PROBE_20261029.md)、[目标滤波](README_Q1_TARGET_FILTER_PROBE_20261029.md)、[PD](README_Q1_PD_SENSITIVITY_20261029.md)、[动态上肢](README_Q1_ACTIVITY_BLEND_20261029.md)、[精度](README_Q1_PRECISION_SENSITIVITY_20261029.md)、[接触审计](README_Q1_REFERENCE_SUPPORT_20261029.md)、[RF16](README_SUPERMIMIC_Q1_RF16_RESULTS_20261029.md)。保留每个失败，不把短试验无收益解释为算法理论上不可行。

## 为什么还不能只说过拟合

RF14后期300条训练集和30条验证一起退化，两普通走最终也失败，证据不符合“训练越来越好、只对新数据变差”的单纯泛化过拟合叙述。夜间RF15/RF16又表明：冻结教师、减小PPO更新和改接触标签，仍可能在滑移、离地、姿态之间交换代价。当前只能确认优化指标与所需物理效果存在偏离，尚未隔离奖励权重、探索、critic、残差权限或原模型偏差各自的因果；不宣称找到单一根因。

## 算法研究与下一步取舍

已复核HugWBC、HOMIE、ExBody/ExBody2、BeyondMimic，以及ConstrainedMimic、X-WBC/XHugWBC、GR00T WholeBodyControl的一手入口。外部视频描述和无法核验的X转述不当Q1结果。

- 下肢先建立明确的速度/朝向、负重转移和实际脚底离地任务，上肢作为训练中可见的干预，再组合幅度控制，是值得单独验证的方向。现有小残差权限是否不足仍是假设，不能靠再多训几小时确认。
- 输出接触/平衡约束必须允许接触切换。ConstrainedMimic官方示例明确展示双脚固定约束会阻止行走；不能把“双脚锁住”当自然步态修复。
- X-WBC当前官方README仍在准备实现/权重，XHugWBC代码入口coming soon。本轮没有可直接装到Q1的已验证通用权重，不虚报已有模型。
- GR00T文档有新机器人适配入口，但仍需实际关节/体/动作顺序、执行器及重定向匹配；其大规模SONIC训练也不能等同本次单GPU短试验。新增Transformer、扩散或LoRA不替代这些控制契约和真实物理验证。

[一手读取层次和适用边界](../data/manifests/supermimic_q1_overnight_research_recheck_20261029.json)；[研究报告](README_Q1_OVERNIGHT_RESEARCH_20261029.md)。以上只是后续假设，没有新增该方向物理/优化，也没有替用户决定继续训练。

## 交付与等待

RF15、ACTIVITY及RF16共11份本地54秒视频已按各自收据完整验收。RF16于06:32:05全pipeline完成，原进程退出/GPU释放、12保存检查与1532训练来源未变，24媒体检查/12联系图及各图目视完成；102小工件68096950字节Mac/25386大小SHA一致。RF16交付发布5059a10a28b3420e86123f32b6a9250ada32fd6d已独立40位核远端、8记录280338字节三端一致；后续综合报告提交不改变冻结训练来源。

权重/大NPZ继续SERVER_LOCAL_ONLY，没有清数据、重转AP03、建网站、重启旧训练或启用agent。原监督保持ACTIVE每10分钟，材料不变时静默等待；用户看数据/视频后再决定候选采纳与下一轮训练，不自动结算整体目标或追加预算。

综合审阅已发布4f773c316aa016a062c1a581d8eff69a3c2d5c04，独立40位核远端，8记录280300字节权威Git/Mac主根/25386逐大小SHA一致；原监督已工具更新并结构化回读ACTIVE/每10分钟/等待用户审阅，其他自动任务未修改。[实际发布与监督回读收据](../data/manifests/supermimic_q1_overnight_publication_20261029.json)。
