# Q1 RF13结果与基本展示验收

日期：2026-10-28，北京时间。**训练及pipeline正常结束，但基本展示目标未稳健达标。** 两段固定普通前走三次均完整且确实向前移动；最终固定抬手一次176/271步跌倒，另两次也出现明显原地漂移、滑移与腿部偏差。精确根轨迹按用户最新要求仅作诊断；本次否定稳健展示的依据包含实际失稳，不是单因旧根阈值。

18:42发布核验：结果提交854e406506afcf7a1d6e96bf8cb58b11109a975f已push origin Q1并独立ls-remote核40位相同；14项记录512622字节在权威Git/Mac主根/25386逐大小SHA匹配，补齐断线期间待镜像记录。原监督已18:42:44工具PAUSED并读回，supermimichp-x2与yuanqi未修改。完成回执及本段随后以同流程提交/镜像，最终收据保存该发布链，不追加训练。

## 目标、配方与终态

用户批准UR03折中与continue后，仅一次4096环境腿腰残差短试验。冻结RF11教师/归一化，8手臂75%参考混合；只训练14腿腰残差head/新critic，RSL-RL PPO，原300训练/30验证。结构、课程、奖励同时变化，不能作单因素归因。完整实现与预检见[RF13阶段](README_SUPERMIMIC_Q1_RF13_BALANCE_20261028.md)/[UR04](README_Q1_BALANCE_ADAPT_UR04_20261028.md)。

冻结训练提交`51d4f00a5c9fb6f8bbdef29e11142a6bf822d41a`；plan SHA`99255cc518b50c4a196ec8b4dd8d9ee1d22bbc7dead44e1c626fcf4dcd8a5cd5`，1137来源。后续报告和终后媒体读取工具不改变这一训练来源。

15:21:12启动controller93989/start_ticks8571724、worker93994/start_ticks8571852。16:32:51 run TRAINING_COMPLETED，`one_hour_budget_completed`；16:43:38 pipeline COMPLETED。**实训3500.385425953704秒/1093轮/107446272转移，critic21860/actor21660，0新WM/想象更新。** 初始化、评测、保存与媒体均不计实训，硬3600秒内结束，无重启/追加。

17:08:42恢复原Windows/25386桥后独立核实boot未变，原controller/worker不存在，train/delivery guard均WORKER_COMPLETED、reaped=true、return0，GPU15MiB/26度且无计算进程、OOM0。16:45断线期间UNKNOWN保留历史；连接中断未触发重启。clock终后留PAUSED_FOR_NONTRAINING、active_started=null，累计3500.385426秒；不能把此字段误判成仍在训练。[现场](../data/manifests/supermimic_q1_RF13_connection_restored_20261028.json)。

## 固定协议三重复结果

固定330片、同三次重复/同4096前向块算术；每次30验证，固定两普通前走和抬手均为原片，不换片。下表提前终止数按各片真实终态；完成率/误差另保留原值，不能用验证0失败掩盖训练展示片失败。

| 条件 | 300训练提前终止 | 30验证提前终止 | 固定抬手步数/271 |
| --- | --- | --- | --- |
| RF11固定块 | 21 / 20 / 18 | 0 / 0 / 0 | 271 / 271 / 271 |
| 未训练static075 | 28 / 28 / 27 | 0 / 0 / 0 | 271 / 271 / 271 |
| stage01，1201.398777秒 | 30 / 30 / 33 | 0 / 0 / 1 | 271 / 271 / 271 |
| stage02，2401.480799秒 | 35 / 33 / 31 | 0 / 0 / 0 | 167 / 271 / 271 |
| stage03最终，3500.385426秒 | 31 / 34 / 29 | 0 / 0 / 0 | 176 / 271 / 271 |
| 最终启发式回退，0额外优化 | 26 / 25 / 26 | 1 / 1 / 1 | 271 / 258 / 271 |

最终训练片等权完成96.76%至97.08%，验证三次100%。两段普通前走在上述正式各点三次都完整320/320、210/210步。原完整物理目标最终0/3，严重退步门0/3，正常预算停止；不因训练reward提高而宣布成功。

## 跟踪与学习收益边界

下表是最终30验证配对共同前缀等权误差/比值，三次平均共同前缀均5.304秒；抬手峰高单列其自身轨迹，不冒称为全程配对。完整数值含RF11、static075、三学习点、回退和逐片量值见[结果JSON](../data/manifests/supermimic_q1_RF13_results_20261028.json)。

| 指标 | 最终三次范围 | 相对RF11 | 相对未训练static075 |
| --- | ---: | ---: | ---: |
| 根XY RMSE m | 0.17917至0.18302 | 1.352至1.480倍 | 0.944至0.966倍 |
| 上肢RMSE rad | 0.15939至0.15971 | 0.1700至0.1702倍 | 1.0037至1.0049倍 |
| 骨盆heading相对腕RMSE m | 0.05402至0.05434 | 0.441至0.443倍 | 0.995至1.003倍 |
| 腿RMSE rad | 0.27580至0.27699 | 1.040至1.046倍 | 0.987至0.994倍 |
| PD目标RMS变化 rad/20ms | 0.06638至0.06738 | 1.239至1.269倍 | 1.006至1.037倍 |
| 原生承重切向滑速 m/s | 0.07123至0.07607 | 1.026至1.097倍 | 0.948至1.024倍 |

**手臂约83%改善主要在训练前static075已存在。** head训练最终验证根比static075约改善3.3%至5.6%、腿约0.6%至1.35%，上肢却约增0.37%至0.49%，并未解决固定抬手平衡。不能说新网络学会了上肢，亦不能从组合试验推断某一奖励/算法是唯一根因。回退验证上肢约0.373至0.382rad、每次1条失败，牺牲上肢仍未消除抬手失败，不推荐为已解决方案。

## 普通走与抬手展示

只读保存NPZ、0新物理步/0优化后，第一普通走在320个post-step帧的6.38秒跨度内沿参考净方向前进2.230至2.395m，参考2.193m；第二210帧/4.18秒前进1.943至2.017m，参考2.057m。此处不含reset帧，不能写为从reset开始6.40/4.20秒的位移。两片承重滑速分别0.109至0.116/0.130至0.132m/s。原始支撑侧跳变存在接触抖动，不能把计数当自然步数。

本地初末视频序列抽帧可见前走换脚和保持直立，但仍屈膝、足部贴地、姿态偏差；完整解码和有限抽帧不等于自然步态认证。固定抬手初始static075能完成抬起/落下；最终repeat0抬起后躯干倾斜、腿部失稳并跌倒，后续保留红色TERMINATED hold，不能隐藏或换重复。最终双腕峰约0.470/0.532至0.534m，参考0.534/0.560m，峰高平均误差0.0451至0.0460m；峰时间1.88/2.22秒，参考1.42/1.72秒，幅度收益不代表时序与平衡达标。

| 固定抬手 | 腿RMSE rad | 承重滑速 m/s | 目标RMS变化 rad/20ms |
| --- | ---: | ---: | ---: |
| stage01三重复 | 约0.200 | 约0.021至0.024 | 约0.052至0.054 |
| 最终repeat0，提前终止 | 0.55893 | 0.12532 | 0.18076 |
| 最终repeat1，完成 | 0.41718 | 0.13349 | 0.14588 |
| 最终repeat2，完成 | 0.46316 | 0.14141 | 0.15970 |

最终抬手repeat1/2身体累计XY路径2.817/2.980m，参考同完整post-step段0.2866m。不是仅精确根轨迹稍差：原地动作出现显著非期望移动，完成也不能算稳定原地展示。repeat0仅比较其3.50秒存活前缀，不与其余5.40秒混作同长度。

保留未训练static075/initial作为现成展示候选，以及stage01早期检查点证据；它们本次固定片更稳，仍不是新增训练成功或独立部署验证。当前未切换模型或重做候选媒体，最终失败版本完整交付。后续如获另行训练授权，应优先围绕原地上肢干预下的腿腰稳定、残差偏移约束和旧动作保持提出可检验对照，精确根路径不再作为首要展示目标；本轮不追加训练。

## 审计与本地交付

已有`script/supermimic_q1_balance/audit_saved.py training`只运行一次，12项保存检查通过：教师身份/指纹、head确有改变、固定std、有限值、计数、预算、0新WM、1137来源均一致，0新模拟器/优化器。输出`run/saved_evidence_audit.json`。本地原工具声明格式不适合RF13，新增独立终后`script/project_tools/audit_q1_rf13_media.py`读取`initial_media.json/final_media.json`，不改冻结源码；12媒体检查通过、两段均完整解码1350帧/H264/1280×480/25fps/54秒。

服务器42个已有小文件共28862203字节，Mac逐大小/SHA全部匹配；另生成6张分段/前走/抬手序列图和本地媒体审计，已实际目视9张图（含六格/PPO/奖励分解）。媒体8段6类型、原速跨片合集，**不是连续54秒rollout**；两普通走/抬手固定，抬手与舞蹈失败hold保留，不挑成功版本。[交付清单](../data/manifests/supermimic_q1_RF13_local_delivery_20261028.json)。

Mac根：`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF13_LOWER_RESIDUAL_SOMA300_E4096_MAX01H_20261028`。

- [初始54秒MP4](../data/supermimic_for_q1/BONES_RF13_LOWER_RESIDUAL_SOMA300_E4096_MAX01H_20261028/evaluation/initial.mp4)：未训练零head/static075。
- [最终54秒MP4](../data/supermimic_for_q1/BONES_RF13_LOWER_RESIDUAL_SOMA300_E4096_MAX01H_20261028/evaluation/final.mp4)：保留抬手跌倒与舞蹈失败。
- [六格PNG](../data/supermimic_for_q1/BONES_RF13_LOWER_RESIDUAL_SOMA300_E4096_MAX01H_20261028/dashboard/training_dashboard.png)，同目录PDF/CSV为原值+500iteration均值；WM/想象明确N/A，另PPO与残差/奖励诊断。
- `evaluation/*_walk_sequence.png`、`*_raise_sequence.png`及`local_media_audit.json`为Mac终后核验，不含新策略执行。

最终权重`run/final.pt`9855867字节/SHA`d78dd76ce0d1f74303349cd16a2e8a2c534f35980e41103150ac6173dbcf061d`，大NPZ和权重只留25386服务器SERVER_LOCAL_ONLY，未称网盘备份。

复核命令（只读既有产物；各审计输出已存在，不要重跑覆盖）：

```sh
# 服务器：已有Python经activate.py运行audit_saved.py
data/environments/supermimic_q1_hp_20261026/bin/python script/supermimic_q1_hp/activate.py script/supermimic_q1_balance/audit_saved.py training --trial data/supermimic_for_q1/BONES_RF13_LOWER_RESIDUAL_SOMA300_E4096_MAX01H_20261028 --output data/supermimic_for_q1/BONES_RF13_LOWER_RESIDUAL_SOMA300_E4096_MAX01H_20261028/run/saved_evidence_audit.json
# Mac：独立RF13声明读取，全帧解码与序列图
python3 script/project_tools/audit_q1_rf13_media.py --trial /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF13_LOWER_RESIDUAL_SOMA300_E4096_MAX01H_20261028 --output /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF13_LOWER_RESIDUAL_SOMA300_E4096_MAX01H_20261028/evaluation/local_media_audit.json
```

记录按Q1显式暂存/payload audit、push/独立40位远端核验后镜像Mac主根和25386，包括此前断线未镜像的阶段记录。三端/监督最终状态以[完成收据](../data/manifests/supermimic_q1_RF13_completion_audit_20261028.json)为准；尚未完成发布时不凭本段认定三端齐备。完成交付后暂停原q1-rf08-4096，supermimichp-x2和yuanqi保持，不创建替代任务。
