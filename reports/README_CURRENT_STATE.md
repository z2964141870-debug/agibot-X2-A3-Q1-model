# YUANQI 当前状态与下一步

2026-10-24：[D05独立根高候选](README_SUPERMIMIC_Q1_SUPPORT_PROJECTION_20261024.md)代码实现，服务器控制与真实候选待执行；只改根部Z/竖直速度，原关节/时间/水平路径保持，默认不接入训练。R03原进程01:26:51实核3900/12000追加轮、累计6900轮，冻结条件通过，GPU40摄氏度/OOM14，无终态收据；继续原预算，两个训练源码目录与输入保持冻结。

2026-10-24 01:14:16实核：R03原controller762428/start3606081278、worker762431/start3606081380同boot非僵尸，3525/12000追加轮、56400新交互、1359组WM/1309次actor，累计6525轮，冻结条件通过，GPU39摄氏度/空闲16671MiB、容器31055925248字节/OOM14，无终态收据。[D04独立参考诊断](README_SUPERMIMIC_Q1_REFERENCE_DYNAMICS_20261024.md)两端控制、3475帧/6950次inverse、276项保存数组重算通过；另三峰值帧inverse与保存数组最大差0，所有证据SHA通过，0积分/推理/optimizer/无Torch。走路较低脚底平均离地19.68mm、实际接触1/1648；出拳7.64秒左腕/左髋约5mm穿透，峰值左肩233.68Nm主要来自自碰撞约束，不宜据此增cap。小证据data/manifests/supermimic_q1_D04_*_20261024.json，NPZ仅服务器LOCAL_ONLY。精确参考质量缺口确认，学习失败主因未定；R03源码/输入/预算不变，等待最终完整评测与V02实际渲染。

2026-10-24 00:28:01实核：R03原controller762428/start3606081278、worker762431/start3606081380同boot非僵尸存活，2150/12000追加轮、34400新交互、809组WM/759次actor，累计5150轮，冻结源码/输入/运行时通过，GPU39摄氏度/空闲16671MiB、容器31050973184字节/OOM14；无终态收据，等待原固定预算及最终评测。[V02新协议渲染准备](README_SUPERMIMIC_Q1_RENDER_PREP_20261024.md)两端各4项控制及R02B三片真实只读预检通过，小收据/日志已取回并逐项大小/SHA一致，0新物理/推理/渲染；EGL、完整视频及画面待执行。两个训练源码目录保持冻结。轮次口径：R01旧接触batch32基线，R02修正接触下32/128从零对照，R03继承B固定追加12000轮，详见[R02对照记录](README_SUPERMIMIC_Q1_BATCH_TRAINING_20261023.md)。以下为早期检查点，不能当作实时进度。

2026-10-23 23:41:35：[R03追加12000轮](README_SUPERMIMIC_Q1_R03_CONTINUATION_20261023.md)实时800轮/12800新交互/269组WM/219次actor，累计3800轮。独立CPU读取第500轮与step0：149组WM/99次actor及Adam12980/1198/23960精确匹配，三网络有限且变化，manifest/20条前缀日志/冻结61项源码/输入/运行时通过。此前真实父final→step0的3网络/3Adam/bin共8项精确一致；原controller762428/start3606081278、worker762431/start3606081380同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，非僵尸/实际argv精确匹配。输出data/supermimic_for_q1/R03_B128_extend12000_20261023，日志logs/supermimic_q1_R03_B128_extend12000_20261023/；GPU40摄氏度/CPU核验后容器31738470400字节/OOM14，小证据data/manifests/supermimic_q1_R03_checkpoint500_20261023.json。训练进行中、尚无最终跟随效果，LOCAL_ONLY，无自动恢复或追加预算。

2026-10-23 23:07:21核验：[Q1 R02批量对照](README_SUPERMIMIC_Q1_BATCH_TRAINING_20261023.md)两臂18阶段及比较重算通过，终态PAIR_COMPLETED；B3000轮/48000交互/101.847分钟，三片完整、关节RMSE0.363888/0.386939/0.311051rad，腿部0.178007/0.251199/0.110641，仍未过0.15门。走路最大位移0.030115m/参考1.126480m，未学会移动跟随。原controller/A/B已退出或僵尸，未当作活跃作业。主机huanyu:/opt/data/private/YuanQi_Q1，核心6b43cc0，GPU39摄氏度/空闲17788MiB、容器26910662656字节/OOM14。

[D03前缀分析](README_SUPERMIMIC_Q1_PREFIX_DIAGNOSIS_20261023.md)A/B四份真实比较完成：同前700步A/B全身0.346602/0.356039rad、腿部0.237951/0.183202、水平根0.561301/0.622730m；B并非所有指标更好。13小证据为data/manifests/supermimic_q1_R02_complete_*_20261023.json，逐项大小/SHA通过，模型/轨迹留服务器LOCAL_ONLY。以B作为已启动R03的父模型，累计目标15000轮/240000交互，保留模型/Adam/采样统计，新seed20261024与fresh replay；真实结果待完整流程。

A batch32已完成95.087分钟训练、保存审计11门及完整配对评测/重算；走路只42.373%后失败。B腿部和稳定性较好而上肢误差较大，选择起点不代表单种子证明batch128普遍更优。出拳仍是复用开发；训练效果、仿真与真机分开报告。

原`R02_batch_pair_20261023`已在140轮因容器30GiB保护保存STOPPED_SIGNAL/0actor；终态和权重保留，不能重用或恢复。缓存建议使容器观察下降1711742976字节，6项控制器测试通过；新运行仍保留30GiB/86摄氏度/OOM保护。A3停止，R01冻结工件保留，大文件LOCAL_ONLY，当前没有新最终策略成绩。Q1分支及本地代码副本`/tmp/yuanqi-supermimic-q1-20261023`维护。

前序证据：[S01/P07接触修正与J01/T01/B01/C01](README_SUPERMIMIC_Q1_IMPROVEMENT_20261023.md)、[D02奖励诊断](README_SUPERMIMIC_Q1_LEARNING_DIAGNOSIS_20261023.md)、[V01动力学视频](README_SUPERMIMIC_Q1_VISUAL_20261023.md)、[R01完整记录](README_SUPERMIMIC_FOR_Q1_20261023.md)。R01三段关节RMSE0.328/0.338/0.290rad，未过0.15门，走路最大位移policy0.041m/参考1.126m；回报增量89%/89%/94%来自限位惩罚减少。R01实际摩擦0.6与新协议分开，不追改原结果。视频页http://127.0.0.1:18863/保留供看旧结果。

可用连接经用户Windows yu@100.65.183.91跳转huanyu；旧100.85.145.78路线超时，未因此停止/重启训练。凭据只用于临时SSH，不写文件。静态PID不证明仍在运行，详细过程在阶段README，通用研究独立待同步项未在本次合并。

[E01/E03后续入口准备](README_SUPERMIMIC_Q1_EXTENSION_PREP_20261023.md)：原11+5控制的证据保留，d5f9476/61项release用于真实R03；登记与启动、实际父状态继承已通过，完整续训保存审计/评测仍待结束，不能用启动检查替代训练效果。

## 用户目标与固定约定

- 长期目标：衣服与头环自然遥操 X2 / A3 / Q1，并形成自主策略训练能力和科研课题。
- 近期目标：独立完成一轮机器人策略训练，并逐步形成可验证的论文工作。10/2 用户明确不必以底层算法创新为目标；教学改为实践先行、数学按需，见[最新安排](README_LEARNING_ROUTE_20261002.md)，仍按[训练工作包](训练工作包.md)从 P0/P1 开始。
- 训练位置：`ssh hp3090`，`/media/yu/FAFF-E9771/YUANQI`。
- 目录：`data/` 产生的数据，`logs/` 纯执行记录，`script/` 代码，`reports/` 重要 README。
- 关键记录与代码：`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`；大模型和关键大型数据：百度网盘。
- Q1 专用工作副本：hp3090 的 `/media/yu/FAFF-E977/YuanQi_Q1`，当前使用并跟踪 `origin/Q1`；用户指定 Q1 重要文件上传到 `Q1` 分支，见[版本管理记录](README_Q1_VERSIONING_20261008.md)。
- 每完成重要任务立即写阶段 README，更新此文件和[报告索引](README.md)，提交并同步 GitHub。规范见 [AGENTS.md](../AGENTS.md)。
- 教学需定期阶段总结；默认每 2–3 个关键概念或遇到理解卡点回顾一次，按反馈调整，区分已讲解与已理解。

## 已完成，可从证据继续

10/10 Q1输入位置复核：hp3090 `data/experiments/q1_cloth_fgp_20261008_E003/E004/body_only.npz`实际读取，分别1766/1767帧、根位移非零；同目录trace含速度与脚接触概率。Mac两份旧JSONL及身体提取小包存在，服务器旧包位移全零与用户确认无移动一致。主要缺口是世界根轨迹/接触的独立校验、现场标定/形体/时序和既有姿态重建精度；所有数据仍TEST_ONLY，未重跑推理/训练/真机，完整路径与证据见[输入清单](README_Q1_SMPL_INVENTORY_20261010.md)。

UMR 成果提交 `7ebf80f63d55f92b8542d5555141fb1e62924e7a` 已推送并核验 `origin/Q1`；校准/GMR 提交也在其连续历史中，回执见 `logs/session_records/q1_retarget_publication_20261009.json`。既有两份 PredActor 暂存 blob 核验未变。

Q1 UMR E008/E009完成：256点/50epochs、同32帧人体膝范围84.43°；默认Q1 75.20°/11.99rad/s，速度约束后41.71°/1.999981rad/s，视觉网格仍穿地2.89cm。25项回归通过；16→10betas截断，非严格排名，无policy/PD/平衡/真机，作业结束，LOCAL_ONLY。下一步核对足底网格/碰撞体/采样点与统一时间/速度/形体，见[UMR记录](README_Q1_UMR_20261009.md)。

Q1 GMR E003–E005：官方 `bb1bbe4` 源码 + 实验 Q1 配置完成四段 AMASS 的不限速/2 rad/s 对照，八组 IK、四段合规参考/支撑 PD。前踢不限速膝范围 93.66°但峰速 23.66 rad/s；2 rad/s 后 40.49°、IK P95 12.55 cm、支撑 RMSE 0.1977 rad，整体保真未验收。两段仅 1e-18 rad 边界尾差已独立规范化，原件保留。NMR 公开 G1 推理、Q1 训练代码/配对数据尚未公开；UMR 后续实测见上方最新阶段。无 policy/真机，见[GMR 记录](README_Q1_GMR_20261009.md)。

Q1 校准 E002：14 组对照与 25 项检查完成；根位移现只绕世界 Z 对齐，慢跑异常高度 0.739 m→正确缩放源 0.049 m，五组旧 qpos 精确复现。合成站姿将前踢膝几何范围 32.93°→63.52°，但 IK P95 5.57 cm→11.21 cm，整体保真未通过，仍 TEST_ONLY / LOCAL_ONLY。进程结束，无训练/真机；下一步实际 Q1 GMR 对照及 NMR 等公开可用性核验，见[校准记录](README_Q1_CALIBRATION_20261009.md)。

Q1 10/9：AMASS 原始 SMPL-X 四段、BONES-SEED G1 两段、PHUMA→X2 两段，共 1,747 帧 / 34.78 秒的 IK 与固定骨盆 PD 完成，限位违例 0；仍是自写 Q1 映射的 Mink/quadprog。发现干净前踢膝幅度 86.69°→32.93°，完整首帧旋转对齐将水平位移混进高度（慢跑缩放源 Z 范围 0.049 m→Q1 0.739 m），高保真/移动未通过。23 项既有检查通过，四张图与独立公式诊断完成，进程已结束；TEST_ONLY / LOCAL_ONLY，未训练或控制真机。见[测试](README_Q1_DATASETS_20261009.md)、[新数据研究](README_Q1_DATASET_RESEARCH_20261009.md)。

该阶段结果 `90aed9e` 已推送并核验 `origin/Q1` 一致；发布回执为 `logs/session_records/q1_datasets_publication_20261009.json`，其他暂存修改保留。

Q1 最新输入阶段：走路/单腿的衣服 IMU→FGP→SMPL 及 Q1 运动学/支撑 PD 数值通路完成，限位违例均 0；两张抽帧图已检查，单腿抬腿幅度不足，见[阶段记录](README_Q1_CLOTH_FGP_20261008.md)。人体标签关节误差 P95 37.75/39.35 cm、根轨迹 RMSE 65.79/53.61 cm；对诊断校准后的 Q1 IK 目标误差 P95 1.43/9.54 cm，不能混淆或认定高保真/无支撑平衡。数值恢复与原走路 9 项指标完全一致，所有测试进程已结束。用户换墙壁供电后仍有 21:06:17 新启动，原因/是否人工与稳定性待确认。无需新大文件传输，全部 TEST_ONLY、未训练/操作真机、备份 LOCAL_ONLY。结果提交 40c8f95 已推送并核验 origin/Q1 一致，回执见 logs/session_records/q1_cloth_fgp_publication_20261008.json；其他暂存内容保留。

Q1：目标为 SMPL 全身跟随（腿部和移动），hp3090 专用副本使用 `Q1`。模型、22DOF 参考/PD 和离线骨架重定向已核验，见[仿真](README_Q1_SIM_CONTROL_20261008.md)、[重定向](README_Q1_RETARGET_20261008.md)。用户新增两段动捕服→FGP→SMPL 录制，明确只测试：3892 帧审查、输入 6 项检查、服务器身体包哈希核验及两段约 75 秒完整重定向/支撑回放完成，见[真实输入阶段](README_Q1_MOCAP_TEST_20261008.md)。位置残差 P95 3.72/6.51 cm；第一段首帧显著屈膝而诊断校准映射到中立位，未实现高保真同姿跟随。两段根部位移全零，动态平衡/行走、policy、真机、实时传输及硬件坐标仍未验证。数据留服务器，网盘 `LOCAL_ONLY`；Mac 仅小型恢复文件，其他研究待同步项独立保留。

1. 学习路线、第一课、训练工作包、实验模板与 EgoLocate/头环静态分析已保存；10/2 新完成官方倒立摆 PPO 基础仿真训练/保存/重载，见[第二课实验](README_LESSON02_CARTPOLE_20261002.md)。尚未完成 X2/人形策略训练。
2. hp3090 上四目录项目已建好，确认 RTX 3090 24GB 与可写 ext4 磁盘；初始化前可用空间约 634 GB，后续使用前重新检查。
3. 初始化提交 `c7b64a7515b551a63830d0e040a203ba543346e5` 已推送 `origin/main`，该阶段结束时本地/远端一致且工作区干净。见[初始化记录](README_SETUP_20260929.md)。
4. 17 份参考节选已同步到项目 `data/references/` 并核验 SHA-256；原始附件仍在 Mac，未复制全部 ZIP。见[清单](../data/manifests/reference_excerpts_20260929.json)。
5. 已加入工件登记、Git 暂存检查和只读环境盘点脚本；见 [script README](../script/README.md)。
6. 本轮把重要阶段落盘要求写入项目规则，建立当前状态和阶段模板。见[记录规范阶段 README](README_RECORDING_POLICY_20260929.md)。
7. 每日研究检查最新为 10/5 北京时间 09:10:30 触发；已查来源无实质变化，静默记录，见[本次检查](README_RESEARCH_CHECK_20261005.md)。最新有内容的[简报仍为 10/4](research/2026-10-04/README.md)：DexPolicy 探索幅度、HumanVerse-500/SONIC 人体数据、FlashDexRetarget 手部重定向，均未复现。没有新增假设或训练。
8. 10/2 Agent 基线后，用户已按说明完成 `lesson02_my_first_run`；新目录、训练与重载完成标记及 23 个产物已核验。1024 并行环境、150 次采集/更新循环，平均维持 0.6446→4.9833 秒，达到时限比例 0→100%，重载动作探针误差 0。两次使用相同种子，不算多种子结果；操作已完成，代码理解与自主设计仍待确认。见[第二课跟进](README_LESSON02_CARTPOLE_20261002.md)及[复跑清单](../data/manifests/lesson02_user_run_20261002.json)。
9. 基础 Isaac Lab 仿真及 RSL-RL 训练/保存/重载已通过；人形 X2 的 P0 未完成。原有训练 sandbox 的资产路径/配置与未提交改动仍待处理，见[静态核验](README_P0_INSPECTION_20260930.md)；本轮未修改该工程。
10. 已讲解单步倾斜惩罚，入口新增 `--pole-angle-weight`（默认 1），准备权重 3 的单变量对照；参数与 CPU 数值检查通过，未启动新训练/重载。新版驱动增加权重记录与重载核对，旧实验精确重载继续用保存的驱动快照。见[奖励项与对照准备](README_LESSON02_REWARD_20261002.md)。

PredActor 专题（10/8）：已读 v3、核验公开发行范围及部分源码；当前为 G1 仿真评测包，完整训练/采集/DAgger/部署仍待发布，权重远端未核验。作为 Q1 基线之后的对照候选；没有新训练或已确认创新，见[阅读与适配评估](README_PREDACTOR_20261008.md)。

## 尚未验证 / 尚未完成

- PyTorch/CUDA、Isaac Sim、官方倒立摆资产、PPO 训练与 checkpoint 重载已验证；X2 精确资产/配置、X2 训练、模型导出与 sim-to-sim 尚未完成验证。
- 百度网盘官方 Linux 客户端 8.7.0 已发现，入口 `/opt/baidunetdisk/baidunetdisk`；登录、上传/下载未验证，未上传模型。记录状态仍是 `LOCAL_ONLY`；如现有入口不合适，再与用户讨论替代方案。
- 历史 X2 支撑站立/A3 上肢里程碑不能当作当前完整遥操验收；当前工作没有启动机器人控制。
- EgoLocate 的 Stage-A 推理实现已找到，所引用完整外部训练项目与 `best.pth` 不在已检查的 ZIP 中。

## 下一项具体工作

Q1 工程侧：根重力轴已修复，GMR与UMR小样本已实际完成；现在先核对足底视觉网格/碰撞体/采样点与统一人体形体、时间轴和速度预算，再扩大公平对照。参考质量通过后再做无支撑跟踪与policy训练，见[UMR阶段](README_Q1_UMR_20261009.md)。现有轨迹仍仅测试，没有真机控制。

教学侧：用户已完成按命令复跑，已讲解观测/动作/奖励及倾斜惩罚力度，理解仍待反馈。下一步由用户按[已准备命令](README_LESSON02_REWARD_20261002.md)运行 `lesson02_tilt_weight3`，只把倾斜惩罚权重从 1 改成 3，然后核验结果并做阶段回顾；不直接比较两种奖励规则的总回报，不自动恢复 TD/GAE 推导。

工程侧：官方简单任务的训练闭环已完成，入口为 `script/teaching/lesson02/run_lesson.py`，实验 ID `lesson02_cartpole_20261002_E002`。无需重复从头做基础环境盘点；转向 X2 时仍需处理旧资产路径、配置覆盖与未提交改动，不直接接手旧实验。简单任务不等于 X2 自主站立/跟踪。

训练环境候选：`/home/yu/miniconda3/envs/x2-sonic-isaaclab`；框架：`/home/yu/projects/IsaacLab`（`/home/yu/IsaacLab` 是同一位置）。X2 训练候选的准确路径、提交与缺口见核验报告。新代码/配置与产物遵守 YUANQI 四目录约定。

10/2 教学阶段核验时，Agent 基线与用户复跑的训练/重载均已结束，GPU 无计算进程；随后只准备对照入口与 CPU 奖励数值检查。10/4 研究检查未重新核验训练/GPU 状态，也未启动仿真或训练。最近已核验完成的训练为 `lesson02_my_first_run`，模型/评测与终端记录分别在其 `data/training/`、`logs/` 目录；基线 E002 保留，备份状态均为 `LOCAL_ONLY`。下一次执行前核实 GPU、实际新产物与 Git 当前状态。

## 恢复时的最小检查

```bash
ssh hp3090
cd /media/yu/FAFF-E9771/YUANQI
git status --short --branch
git log -3 --oneline
```

先确认当前文件版本，再读本文件及相关阶段 README。只复核会变化的状态和仍未解决的问题，不重复已完成的静态盘点。
