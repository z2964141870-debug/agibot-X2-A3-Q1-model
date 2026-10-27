# RF12物理PPO结果与本地交付

记录日期：2026-10-28，北京时间。实验RF12。状态：**改进目标失败，按预设物理退步规则结束；训练和交付流程正常完成。** 不进入B/C，不追加五小时，不重试。

## 目标与范围

从RF11稳定策略初始化，尝试以真实仿真PPO改善上肢、腕轨迹和根部跟踪，同时保住稳定性。仅用户25386、唯一根`/opt/data/private/ZHY`；4096个MuJoCo Warp环境，原300训练/30验证、obs80×10/goal590、原reward、控制和合法转移采样。

本次只执行A。B的实际关节/腕奖励和C的AP03新增583片均未启用；不能将这两项称为已测试。AMASS/AP03准备结果见[阶段README](README_SUPERMIMIC_Q1_RF12_PPO_20261027.md)，本轮收尾没有继续删除数据或重转动作。

## 完成内容

实际训练3503.474673854944秒（58分23.475秒）、812个PPO迭代、79822848个真实转移；critic16240、actor16040次optimizer step，前10轮仅critic。0新增世界模型或想象更新。std固定0.1、24步、5epoch×4minibatch、每小批24576、lr1e-4/clip0.2/gamma0.99/lambda0.95/grad1，归一化复制并冻结RF11。

初末各330片、各三重复的1980个真实rollout已保存JSON/NPZ；每个RF12重复和对应RF11重复按共同存活前缀比较。阶段及最终六格PNG/PDF/CSV、PPO诊断和初末各54秒MP4完成。43个服务器小工件48549915字节与Mac逐大小/SHA匹配；权重及六个大轨迹仅留服务器。

## 关键决定及依据

| 验证指标 | 重复0 RF11→RF12 | 重复1 RF11→RF12 | 重复2 RF11→RF12 |
|---|---:|---:|---:|
| 根XY RMSE（m，共同前缀） | 0.12056→0.31984 | 0.12415→0.33140 | 0.11764→0.33121 |
| 上肢RMSE（rad，共同前缀） | 0.93262→1.06279 | 0.93662→1.06106 | 0.93703→1.06237 |
| 腿RMSE（rad，共同前缀） | 0.26561→0.33464 | 0.26664→0.34031 | 0.26524→0.33649 |
| 骨盆相对腕RMSE（m，共同前缀） | 0.12168→0.14017 | 0.12226→0.14053 | 0.12221→0.14024 |
| 指令振荡（rad/20ms） | 0.05384→0.08157 | 0.05371→0.08437 | 0.05330→0.08085 |
| 承重滑速（m/s） | 0.07274→0.09911 | 0.07252→0.10273 | 0.07072→0.09417 |
| 验证提前终止 | 0→4/30 | 0→6/30 | 0→3/30 |
| 验证平均完成比例 | 100%→97.3138% | 100%→97.2781% | 100%→97.8768% |
| 训练集提前终止 | 21→84/300 | 19→85/300 | 20→83/300 |

共同前缀平均5.215/5.196/5.225秒。表中RMSE为各片坐标分量RMSE的平均，并非XY欧氏距离的另一个口径；完整段失败/完成率单独统计。接触是最后5ms子步求解后的原生世界Z力>5N及受力加权切向速度，不是20ms累计冲量。

根XY误差升至2.653/2.669/2.816倍，三次均超过根误差增加25%的严重退步阈值；并非要求三次验证均达到6/30失败。改善门0/3、严重退步3/3，`run/stage_decisions.json`判定`stop/physical_regression`，所以没有进入B/C或追加训练。

固定普通走stop片由320步完整变成56/57/56步；start片由210步完整变成190/189/188步，三次均未完整。stop片共同1.10至1.12秒的参考净位移0.675至0.688m，RF12沿该方向投影1.150至1.161m（RF11为0.886至0.912m）；包含末端倒地位姿，属于过冲，不能称移动更好。start片共同3.74至3.78秒参考1.785至1.808m，RF12为1.600至1.676m，随后提前终止。

固定抬手三次均271步，但左/右腕峰仅0.3145至0.3182/0.0696至0.0713m，RF11约0.2523至0.2529/0.1285至0.1287m，参考0.53425/0.55992m。合并峰高误差只降0.6至1.0%，右腕更差，未达到25%。左腕峰时为1.32/5.40/4.20秒，右腕均1.06秒，参考1.42/1.72秒；时标以首个保存s1为0、步距20ms。保存曲线前期回报下降，后期有所回升，仍不能覆盖真实物理退步。

## 版本、命令与证据

冻结训练提交`684548c8e912802960a837924e5af7fd4c03248d`，RF11来源SHA`6c4de3171bd45359691a63ffe1712729c14fdb2080f8066e4c25632f4bbc5146`。阶段02:15负面结果已发布`2841acb3ba553bf31c05729d26ead749452c3235`并独立核远端，5文件168560字节三端匹配；后续报告/审计工具提交不改变冻结训练身份。

全pipeline于02:28:14完成；train/delivery guard均WORKER_COMPLETED、worker_reaped=true、返回0，阶段交付返回0。02:33独立核原controller49990/start_ticks3416720及worker49995/start_ticks3416816均不存在，GPU15MiB/26度且无计算进程，OOM0，盘余675969384448字节。1123冻结来源286568437字节最终重核全部相同。

独立新工具只读取保存权重/JSON/NPZ/CSV，不执行策略、仿真或优化，37项通过：RF11初始策略逐张量相同、6个归一化状态不变、std固定、optimizer实际步数、有限权重、全部812条计数、三重复轨迹及结果算术、500iteration均值；阶段checkpoint到final的策略/critic/optimizer/RNG均相同。首次`--help`和输出不存在核查后仅运行一次，未套用旧HP训练审计：

```sh
PY=/opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python
"$PY" script/supermimic_q1_hp/activate.py script/project_tools/audit_q1_ppo_saved.py --help
```

实际审计参数为`--trial data/supermimic_for_q1/BONES_RF12_PPO_SOMA_AP03_E4096_MAX08H_20261028 --output data/manifests/supermimic_q1_RF12_saved_audit_20261028.json`。已有输出不得覆盖重跑。[37项收据](../data/manifests/supermimic_q1_RF12_saved_audit_20261028.json)/[量化结果和原值](../data/manifests/supermimic_q1_RF12_results_20261028.json)/[完成核对](../data/manifests/supermimic_q1_RF12_completion_audit_20261028.json)。

执行日志在服务器`logs/BONES_RF12_PPO_SOMA_AP03_E4096_MAX08H_20261028_{train,controller,delivery}.log`；阶段日志在实验`run/stage_01_delivery.log`。大轨迹/初末权重路径及大小SHA在run/receipt、stage_decisions和审计收据中。

## 本地交付

Mac根：`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF12_PPO_SOMA_AP03_E4096_MAX08H_20261028`。

- [最终MP4](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF12_PPO_SOMA_AP03_E4096_MAX08H_20261028/evaluation/final.mp4)及[初始MP4](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF12_PPO_SOMA_AP03_E4096_MAX08H_20261028/evaluation/initial.mp4)，各54秒/1350帧/25fps，8段6类原速独立片段，固定两普通走/抬手不换片；倒地后显式hold和失败标记，非连续54秒rollout。
- [六格PNG](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF12_PPO_SOMA_AP03_E4096_MAX08H_20261028/dashboard/training_dashboard.png)、[PDF](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF12_PPO_SOMA_AP03_E4096_MAX08H_20261028/dashboard/training_dashboard.pdf)、[原值和500iteration均值CSV](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF12_PPO_SOMA_AP03_E4096_MAX08H_20261028/dashboard/training_dashboard.csv)及[PPO诊断](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF12_PPO_SOMA_AP03_E4096_MAX08H_20261028/dashboard/ppo_diagnostics.png)。WM第8步/想象存活率明确N/A。
- [阶段媒体审计](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF12_PPO_SOMA_AP03_E4096_MAX08H_20261028/stages/stage_01/media_audit.json)12项通过、每片1350帧全解码；初末各8段contact sheet与六格图已目视。最终root视频逐字节等同已解码stage视频，元数据除路径外相同，复用该解码证据；终后审计未追加渲染或安装ffprobe。

旧媒体工具仅用于媒体完整性，RF12的`_media.json`被包成兼容元数据，原文件/视频未改；其HP命名不代表使用HP训练审计。目视为八段代表帧，不冒称完整观看108秒。[本地清单](../data/manifests/supermimic_q1_RF12_local_delivery_20261028.json)。

## 结果与边界

当前真实PPO A配方未修复长期上肢瓶颈，还破坏了RF11已有稳定性与根轨迹。完整抬手片仍不能达到参考幅度，普通走出现明确失败。不能把流程COMPLETED或37项完整性通过写成训练效果成功。

此结果不能推出PPO普遍不可用；预训练策略迁移后的梯度、探索std、critic预热、奖励等作用尚未独立隔离。新增奖励B和扩库C未执行，也不能据此否定它们。当前记录缺逐小批KL/clip fraction等诊断，不能仅凭现有loss精确定位根因。扩散/LoRA等仍只是研究候选，没有本轮适用性证据。

## 保存与同步状态

阶段结论、原值、权重/轨迹清单、37项审计和媒体清单已保存。最终记录在持久权威Q1副本经显式暂存、payload审计、commit/push与独立40位远端核验，再镜像Mac主根和25386；同次完整提交号由Git历史及后续完成回执提供。检查点和大NPZ均SERVER_LOCAL_ONLY，未宣称云备份。

## 下一步与恢复入口

本次唯一RF12的训练/评测/本地交付已收尾，监督在这次记录发布核验后暂停，保持`supermimichp-x2`及`yuanqi`不变。无训练进程待恢复，不自动重试、继续B/C或新开实验；若后续核因，只读取已有证据，新的训练须后续明确授权。

恢复先读[CURRENT_STATE](README_CURRENT_STATE.md)、本结果、阶段README及completion_audit；阶段文档内的RUNNING/交付待续为对应历史时点。
