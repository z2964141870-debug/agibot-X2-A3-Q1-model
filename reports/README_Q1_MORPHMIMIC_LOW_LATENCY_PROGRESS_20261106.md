# MorphMimic 复现与低延迟 SuperMimic 遥操阶段记录

整理日期：2026-11-06，北京时间。范围为前一篇日记之后，已完成的 Primebot Q1 基线整理、MorphMimic G1+Q1 共享训练与保存点比较，以及低延迟遥操探索。本文件与证据一起交付；所有实验结果来自已经保存的记录，本次整理没有新增训练或物理回放。

## done

1. 完成了 Primebot-SuperMimic 原生 Q1 的 10000 轮训练和初末评估。固定测试从 0/9 完整提高到 9/9，全动作库从 0/393 提高到 393/393，动作跟随已经跑通；但脚底离地偏低、承重滑移仍然存在，不能把完成率等同于自然步态。
2. 完成了 MorphMimic G1+Q1 共享训练的阶段复现。环境为 G1 1024、Q1 1024，总计 2048。比较了 16000、32000、48000 三个保存点，两种机器人、motion/human 两个接口、前走/起走/停走，共 36/36 例完整。
3. 补齐训练六宫图和保存点物理对照。32k 到 48k 的平均跟随误差没有继续下降，部分 human 接口的滑移和侧倾反而增大。因此按要求暂停原训练，保留已有保存点，不再为了凑 16 万轮继续原配方长跑。
4. 实际完成了 32k Q1 同 human 接口的 27 例低延迟对照。直接缩短未来窗口会明显损害跟随；80/100ms 预览配合保持或阻尼预测，在三个固定片上恢复了较多性能。其中 26 例完整，20ms masked 前走失败，失败记录保留。
5. 探索了线性与小型 MLP 上肢预测器，以及连续预测修订损失。干净数据预测有改善，噪声、冻结输入、停止和连续性仍有取舍，尚未接入机器人策略或部署真机。
6. 梳理了十二项相关工作的可迁移方法，形成短预览适配、当前目标加历史学生、接触预测、异步调度和执行延迟补偿等后续路线。

## todo

1. 完成急停、反向、丢包、迟到突发和恢复的 84 例物理压力测试。当前新适配器首次在动作等价门前报错；修复已保存，尚未完成服务器验证和重派，不能把三段名义回放当成事件测试。
2. 根据事件失败类型，比较短预览训练适配与“当前目标 f0 + 历史”的因果学生。复用教师和动作解码器，避免直接重复原 H10 配方。
3. 在更大动作范围检验 32k 候选，继续检查离地、换脚、滑移、侧倾、上肢相位和动作突变。
4. 与成竹师兄讨论 SuperMimic-tele 真机接口、时间戳和执行端响应；量测采样到机器人动作的真实端到端延迟，再判断是否适合自然遥操。

## 1. 训练和当前状态

| 项目 | 已完成或最后核验状态 | 解释 |
| --- | --- | --- |
| Primebot 原生 Q1 | 10000 轮训练、初末固定与全库评估已完成 | 属于先前基线，不是本次重新训练 |
| MorphMimic G1+Q1 | 11 月 4 日 11:06 启动的同一轮共享训练 | 每型 1024 环境，总 2048；不是每型 2048 |
| 原共享训练暂停 | 11 月 6 日 16:29:44 核验暂停，原五 PID 退出、GPU 锁释放 | 由用户要求暂停；旧 pipeline 的 FAILED 来自人工停止，不是新数值故障 |
| 暂停前最后周期日志 | 每型 56600 轮 | 不是精确中断轮，也不是保存点 |
| 磁盘保存点 | 0、16000、32000、48000；latest 为 48000 | 56k 后续内存没有即时导出，不承诺精确续训 |
| 低延迟机器人策略新训练 | 暂停原训练后为 0 | 已完成的是保存权重推理和物理对照；新的适配训练尚未开始 |
| 上肢预测器本机训练 | MPS 实际累计 3000 次更新 | 与共享机器人策略训练分开；另有线性模型拟合 |
| 低延迟整体目标 | 最后记录为 blocked，尚未完成 | 后续事件测试受原密码桥接阻塞；本次只整理本地证据，不重新连接或启动任务 |

暂停和完成性证据见 [暂停收据](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/05_records/q1_shared_user_pause_20261106.json)、[目标完成性审计](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/05_records/q1_low_latency_goal_bridge_blocked_audit_20261106.json)。其中 RUNNING/active 等旧字段是各自实验当时的快照，不代表整理时的实时状态。

## 2. MorphMimic 是否已有改善，是否接近收敛

同三段 motion 完整关节 RMSE，Primebot 10000 轮为 **0.165292 rad**，共享模型 Q1 分支 16000 轮为 **0.071873 rad**，约降低 **56.5%**。滑移和上肢误差多数改善，但不是所有指标都赢：前走末根 XY 误差曾从 Primebot 的 **0.003600m** 增至 Morph 16k 的 **0.268490m**。两套训练样本预算不同，也不能只按轮数比较训练效率。

Primebot 全库完成率的原始结果也随包保存：[初始 0/393](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/00_primebot_baseline/initial_library_result.json)、[最终 393/393](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/00_primebot_baseline/final_library_result.json)。这些结果证明完整回放，不证明全库自然步态合格。

下表为每组前走、起走、停走三片等权完整关节 RMSE，单位 rad，越低越好：

| 机器人 / 接口 | 16000 | 32000 | 48000 | 32k 到 48k 误差变化 |
| --- | ---: | ---: | ---: | ---: |
| Q1 motion | 0.071873 | 0.066745 | 0.068501 | +2.63% |
| Q1 human / H10 | 0.075750 | 0.069598 | 0.071632 | +2.92% |
| G1 motion | 0.078618 | 0.060437 | 0.061877 | +2.38% |
| G1 human / H10 | 0.104456 | 0.094814 | 0.108175 | +14.09% |

**阶段判断：当前配方在这三片上的主要收益已进入平台期，继续到 160000 轮的必要性不足；32k 值得优先扩大验证。** 这不是统计意义的全局收敛，也不是认定 32k 为全库最佳。36 例是训练集固定片、确定性 seed0 回放，原定最终评估门没有完成或放宽。

训练信号也接近平台：48k 到 53.9k 的最近窗口，Q1/G1 标准化回报仅约 +0.20%/+0.19%；世界模型八步误差仍约下降 1.43%/1.73%。因此优化器仍在学习，但更低 WM loss 没有在这些物理片上转化为稳定收益。训练图快照截至 53900，不能把它标成精确暂停轮 56600。

![Q1 训练六宫图](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/01_training/q1_training.png)

![Q1 human 保存点物理对照六宫图](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/02_checkpoint_review/q1_human.png)

| 其余六宫图 | 对应证据 |
| --- | --- |
| G1 训练 | [PNG](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/01_training/g1_training.png) |
| Primebot 真实已记录指标 | [PNG](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/01_training/primebot_recorded_training.png) |
| Primebot 标准图与历史缺项 | [PNG](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/01_training/primebot_missing_metrics_training.png)，五项当时未记录，未回填 |
| Q1 motion 物理对照 | [PNG](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/02_checkpoint_review/q1_motion.png) |
| G1 motion 物理对照 | [PNG](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/02_checkpoint_review/g1_motion.png) |
| G1 human 物理对照 | [PNG](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/02_checkpoint_review/g1_human.png) |

逐例数据见 [保存点 CSV](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/02_checkpoint_review/per_case.csv)、[汇总](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/02_checkpoint_review/summary.json) 和 [结构化收敛记录](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/05_records/checkpoint_summary.json)。四张物理图各附 PDF。

## 3. 低延迟探索得到什么

使用已有 **32k Q1 权重、相同 human 接口**，比较 H10/H5/H4/H2/H1，即 200/100/80/40/20ms 真实预览预算；H4/H5 再比较保持和阻尼速度外推。共 27 例，其中 26 例完整。H1 masked 前走在第 248 帧失败，仅 247/1031 有效步，失败没有作为全长成功参与误差平均。

这里的“80ms / 100ms”指未来参考的等待需求，**不是已经测得的采样到机器人响应端到端延迟**。H1 仍是未来一帧 f+1，不是当前帧 f0。把 200ms 内的十个点采成五个，不会自动把最远等待减为 100ms。

前走单片结果如下，承重滑移左/右单位 mm/s：

| 方法 | 预览预算 | 上肢 RMSE / rad | 承重滑移左 / 右 |
| --- | ---: | ---: | ---: |
| H10 masked 基线 | 200ms | 0.085931 | 7.49 / 7.68 |
| H5 masked | 100ms | 0.105290 | 20.59 / 29.73 |
| H4 masked | 80ms | 0.100756 | 21.18 / 25.41 |
| H5 hold | 100ms | 0.087695 | 9.61 / 9.66 |
| H4 hold | 80ms | 0.088671 | 9.45 / 9.01 |
| H5 damped | 100ms | 0.087151 | 8.40 / 9.05 |
| H4 damped | 80ms | 0.087862 | 8.06 / 9.12 |

**80/100ms 加保持或阻尼值得继续，直接 masked 短窗则有明显退化。** 但各片取舍不同：起走 H4 damped 右滑移约 54.22mm/s，高于 H10 的 47.73mm/s；H5 damped 约 39.29mm/s。不能只展示前走就说所有动作都改善，也不能把阈值抬脚事件直接当成自然步数。

![前走低延迟六宫图](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/03_low_latency/forward_six_panel.png)

另外两张：[起走六宫图](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/03_low_latency/start_six_panel.png)、[停走六宫图](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/03_low_latency/stop_six_panel.png)，均附 PDF；[27 例 CSV](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/03_low_latency/per_case.csv) 和 `cases/` 保留每例完整率、误差、动作变化与失败信息。

时间方面，保存 telemetry 的 13473 次目标调用中有 122 次超过 20ms；去掉每片首帧后仍有 95 次，最大约 519.54ms。预热不能直接当作实时期限已经解决。测得的软件计算耗时与网络、感知、执行器响应分开报告。原首次事件入口失败及计时证据见 [事件首次尝试记录](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/05_records/q1_paused32k_event_attempt01_20261106.json)。

## 4. 上肢预测器有没有成效

这部分使用 393 段已保存 reference 的肩肘角度，按来源冻结为 train167 / val98 / test128；不是在机器人执行轨迹上训练全身控制器，也不是完整实时 SMPL 人体预测器。

| 实验 | 真实工作量 | 结果与限制 |
| --- | --- | --- |
| 因果 ridge | 三个线性模型拟合，先验证选型再读 test | 干净均值 0.039339rad，优于阻尼 0.059140rad；加入 0.01rad 噪声后 0.068267rad，反而差于阻尼 0.063950rad |
| 小型 MLP | Mac MPS 1000 次更新，验证选择第 800 次 | 部分噪声场景改善，冻结输入与固定停走仍有退步 |
| 连续修订损失对照 | 控制/修订两模型各 1000 次更新 | 三固定噪声片同目标修订 p95 降约 25%–29%，参考加速度 p95 降约 35%–41%；干净误差增约 3.7%，验证风险没有赢 |

因此“预测更准”和“在线连续预测更稳定”需要同时衡量。当前已有方法上的进展，但没有足够证据称自然遥操合格。测试集已在多轮探索中复用，结果只作为探索性比较；预测器尚未接入机器人策略。

![离线上肢预测探索六宫图](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/04_upper_forecast/offline_forecast_six_panel.png)

图表仅读取已经保存的三次实验收据和训练 trace 生成，没有重新训练或推理。可核查 [ridge 结果](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/04_upper_forecast/ridge_result.json)、[MLP trace](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/04_upper_forecast/mlp_training_trace.json)、[修订对照结果](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/04_upper_forecast/paired_result.json)、[绘图来源收据](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/04_upper_forecast/plot_source_receipt.json)。

## 5. 接下来优先探索的路线

相关工作已经从当前目标学生、时间 mask 训练、运动预测、异步执行、动作队列补偿和约束协同等方向梳理；来源身份与阅读范围见 [十二项工作来源清单](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/05_records/q1_low_latency_research_20261106.json)。文献方法提供设计参考，不代表已经在 Q1 上复现。

| 路线 | 本项目下一项可验证工作 |
| --- | --- |
| 短预览适配 | 固定空间点集和时间轴，只改变时间 mask 课程，比较独立候选 |
| 当前目标 + 历史学生 | 教师可用未来，学生部署只读已到达 f0 和历史；覆盖学生自身偏离状态 |
| 接触与预测 | 分开评价人体意图外推和支撑脚约束，避免错误锁脚或靠不响应获得平滑 |
| 异步与执行补偿 | 记录包龄、推理结束、提交和生效时刻，处理过晚结果和已提交动作前缀 |
| 真机闭环 | 同步时间基准下量测延迟分布，同时验急停、反向、滑移和上肢时序 |

## 6. 视频索引

本包包含 **17 段 MP4**，沿用原速 50fps 文件，没有二次变速或为补媒体重跑物理。Primebot 初始失败后的保持帧沿用原文件；不代表初始策略完成了动作。

| Primebot Q1 / motion | 前走 | 起走 | 停走 |
| --- | --- | --- | --- |
| 初始权重 | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/00_primebot_baseline/videos/initial_forward.mp4) | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/00_primebot_baseline/videos/initial_start.mp4) | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/00_primebot_baseline/videos/initial_stop.mp4) |
| 10000 轮最终权重 | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/00_primebot_baseline/videos/final_forward.mp4) | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/00_primebot_baseline/videos/final_start.mp4) | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/00_primebot_baseline/videos/final_stop.mp4) |

48k human / H10：[Q1 起走](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/02_checkpoint_review/videos/q1_48000_human_start.mp4)、[G1 起走](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/02_checkpoint_review/videos/g1_48000_human_start.mp4)。该轮服务器生成了 12 段 48k 完整视频，本包只有已传到 Mac 的这两段；另外十段未包含。四组采样联系表均在 `evidence/02_checkpoint_review/contacts/`，不能用离散截图代替完整视频。

| 32k Q1 / human | 前走 | 起走 | 停走 |
| --- | --- | --- | --- |
| H10 masked / 200ms 基线 | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/03_low_latency/videos/200ms_baseline_forward.mp4) | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/03_low_latency/videos/200ms_baseline_start.mp4) | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/03_low_latency/videos/200ms_baseline_stop.mp4) |
| H5 damped / 100ms 候选 | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/03_low_latency/videos/100ms_damped_forward.mp4) | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/03_low_latency/videos/100ms_damped_start.mp4) | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/03_low_latency/videos/100ms_damped_stop.mp4) |
| H4 damped / 80ms 候选 | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/03_low_latency/videos/80ms_damped_forward.mp4) | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/03_low_latency/videos/80ms_damped_start.mp4) | [视频](/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_MORPHMIMIC_LOW_LATENCY_EVIDENCE_20261106/evidence/03_low_latency/videos/80ms_damped_stop.mp4) |

低延迟原实验共有 27 段媒体，本包精选上述 9 段用于基线/候选比较，同时保留全部 27 例数值结果。其余 18 段没有包含；缺失媒体不通过重跑实验补造。

## 7. 文件、版本与校验

| 目录 | 内容 |
| --- | --- |
| `evidence/00_primebot_baseline/` | 初末固定与全库结果、6 段视频 |
| `evidence/01_training/` | 4 张训练六宫图、来源记录和窗口统计 |
| `evidence/02_checkpoint_review/` | 4 张保存点物理六宫图及 PDF、4 联系表、2 视频、逐例 CSV/JSON |
| `evidence/03_low_latency/` | 3 张低延迟六宫图及 PDF、3 联系表、9 视频、27 例 JSON 及审计 |
| `evidence/04_upper_forecast/` | 1 张离线预测六宫图及 PDF、训练 trace 和实验结果 |
| `evidence/05_records/` | 暂停、收敛、事件未完成与文献来源记录 |

共 **12 张六宫图 PNG**，另有 3 张低延迟联系表 PNG、4 张保存点联系表 JPG、8 份图表 PDF、17 段 MP4，以及结果 CSV/JSON。模型权重和训练大文件不在本包。

来源版本：原共享训练 `4bef63517164537a9074b7ce58f28e0da35ac8b5`；36 例比较入口 `80cbe2620a4bc63749573ed52c88cfd99e03b3ac`；32k GPU 名义对照修订入口 `184e59b`，完整执行身份见 `campaign_result.json` 与各来源收据。本次图表读取器为 `script/q1_checkpoint_review/plot_upper_evidence.py`。

根目录 `MANIFEST.json` 登记每份文件的源路径、字节数和 SHA-256；`SHA256SUMS.txt` 覆盖 README、证据和 manifest。解压后在本 README 所在目录可执行：

```sh
shasum -a 256 -c SHA256SUMS.txt
```

包内 README 使用相对链接，整目录移动后仍可阅读。此前视频的完整解码审计随包保留，本次整理再次核对源与副本的大小/SHA；新增预测图已目视检查。ZIP 另做压缩完整性和逐成员内容核验。此包是阶段进展证据，不代表原五阶段最终验收、G1/Q1 自然步态或真机遥操已经合格，用户检查点采纳仍为 null。
