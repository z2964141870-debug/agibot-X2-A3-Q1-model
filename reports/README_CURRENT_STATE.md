# YUANQI 当前状态与下一步

2026-10-24当前执行：[Q1 P05扩库训练准备](README_SUPERMIMIC_Q1_EXPANDED_RUN_20261024.md)。组库入口/本地4控制通过；模型槽后端、数值/梯度/64环境资源验证及有限训练入口已实现，仅语法通过，尚无实际新后端或训练结果。P04 B011原生退出0，算术累计1341候选/train311/1121.28s（移动34/119.90s），余520训练候选；B012(offset1341/count256)原句柄8737运行中。下一完整P04组库，再验证模型槽后端与资源；拟新R02固定64环境/H32/12000轮，尚未登记或启动，不恢复旧实验。

2026-10-24当前推进：[PHUMA P04扩量](README_SUPERMIMIC_Q1_PHUMA_EXPANSION_20261024.md)。B008/B009及新跨批汇总原生退出0；累计829个不同候选/209个合格NPZ实际时轴重算完成：train196/708.38s（移动16/57.14s）、dev6/21.26s、test7/27.56s。训练参考11.81分钟，仍未达30分钟，尚未组装新库/启动训练。17:47:23无转换/汇总进程；内存27766009856/OOM14/GPU39，原失败保留。新证据P04_progress_B009/P04_B009_execution；下一个offset829/B010/count256单路，余1032训练候选。AMASS缺男女SMPL-X模型、BONES缺完整G1包/metadata，已告知用户。模型每环境约125.6MB，仅做0物理步元数据查询，尚无新后端或扩环境。

2026-10-24当前优先：[AMASS/BONES-SEED接入](README_SUPERMIMIC_Q1_SOURCE_EXPANSION_20261024.md)。用户已扩展新实验数据范围。huanyu的`script/amass_bridge/data/amass_raw/AMASS_smplxg`找到17928个NPZ，头部索引及独立重算完成：17265标准SMPL-X动作、120Hz、16betas、标称56.34h；另490静态/173待处理，不是Q1合格时长。旧桥接缺字段时默认30Hz，禁止直接复用其PKL。四控制通过、实际索引退出0/OOM14，0新重定向/训练。

当前缺SMPLX_MALE.npz、SMPLX_FEMALE.npz及完整BONES-SEED g1.tar.gz（23499973647字节）/metadata。目标均在`/opt/data/private/YuanQi_Q1/data/references/`，精确路径见[缺项清单](../data/manifests/supermimic_q1_source_readiness_20261024.json)。BONES本机24个混合派生PKL不是全库，官方匿名下载401/GatedRepo；旧hp3090连接超时。AMASS无需用户再迁移，补身体模型后从原始字段构造Q1参考；PHUMA继续可用，跨源先查重/固定移动评测，目标30分钟再1小时合格库。新训练尚未启动。

2026-10-24上一阶段：[SuperMimic数据用法复核](README_SUPERMIMIC_Q1_PHUMA_DATA_USE_20261024.md)。论文BONES-SEED主库129514段/264.56h，LAFAN1对照40段/2.45h；R01实际11训练片/41.30s，只有2移动，属于小样本闭环。代码已有1秒50/40/10采样，但首轮只选112候选，19门后留18片。现有源端温和训练候选26246片/标称26.44h，未去重或Q1准入。下一先准备独立P04扩量/筛选协议，以合格30分钟再1小时为工程目标，同时核查小库可学习性与在线交互预算；尚未扩量或启动新训练，不把量少当唯一失败因果。

PHUMA R01于14:36:50完成固定3000轮/48000转移训练及全部配对流水线，97.6787分钟；7阶段原生退出0、11项保存审计及两份18片轨迹重算通过。14:38:44独立核验121冻结文件、三划分算术、日志身份与原进程组无活跃成员，GPU38摄氏度/OOM14。最终16/18完整但0/18过跟随门；test三非移动片完整关节误差0.272425→0.240971rad，开发移动根误差0.262304→0.263104m、位移0.066904/参考0.373315m，移动仍不足。见[PHUMA主阶段](README_SUPERMIMIC_Q1_PHUMA_20261024.md)及[完成核验](../data/manifests/supermimic_q1_phuma_R01_completion_verification_20261024.json)。final205225043字节/SHA7db966ff…，LOCAL_ONLY，训练发布ed421738不变。

[V01图表](README_SUPERMIMIC_Q1_PHUMA_PLOTS_20261024.md)18片/36组误差重算及两图目视/身份通过。[D02C足底诊断](README_SUPERMIMIC_Q1_PHUMA_SUPPORT_MAPPING_20261024.md)实际退出0，8训练片/1536帧、32几何检查和264统计重算通过，121冻结未变/OOM14。四支撑失败片源固定落地后已有17.68–20.45mm波动、映射20.81–27.61mm，IK/目标曲线RMSE0.56–3.04mm；源与映射贡献筛选缺口，但两合格移动参考也学习失败，不能据此认定策略主因。后续模型诊断已延后，先核论文数据用法；当前无新训练，不追改门或原终态，目标仍未达。

2026-10-24 11:26用户要求停止当前整轮并改用PHUMA；[R04V2已停止](README_SUPERMIMIC_Q1_HORIZON_PAIR_20261024.md)。11:26:30总控/worker均STOPPED_SIGNAL，原生退出0；A实际932轮/14912新控制转移、322组WM/272次actor，B未启动，无完整对照/最终评测。11:27:30–32核验原worker缺失、controller Z/RSS0、两个进程组无活跃成员，3份权重大小/SHA及冻结条件通过；GPU38摄氏度/OOM14。停止证据`data/manifests/supermimic_q1_R04V2_user_stop_20261024.json`。

本轮PHUMA-only参考来自全G1 73076片扫描及固定112候选筛选，18合格（train11/development4/test3）；不是全量PHUMA训练。旧A3动作/权重不参与，原始30Hz按原速重采样50Hz、三网络随机初始化、16环境/B128/H32。[D01输入诊断](README_SUPERMIMIC_Q1_PHUMA_DIAGNOSIS_20261024.md)保留85支撑比例/44自碰撞失败的重叠计数，D02C已核查其映射来源；不据输入筛选瓶颈直接认定策略移动失败原因。

R04V2及原R04停止/失败终态、日志、源码和权重保留，不执行原流水线后续阶段、不解锁或重启。停止保存final为205227539字节/SHA `b9e3914e8bddcbcabef7a7383cf7e2d097bd5a184a000358a2937b11859e6fb8`，未做完整张量/最终效果审计。A3保持停止，全部大型原件LOCAL_ONLY。以下D08/R03/V02为历史证据，不是续跑许可。

[D08动作响应](README_SUPERMIMIC_Q1_ACTION_RESPONSE_20261024.md)已完成2176步/192响应独立数组审计，走路根H8为16/16正向/相对误差0.4412，H32为11/16/0.7595。[D07奖励/模型诊断](README_SUPERMIMIC_Q1_REWARD_MODEL_20261024.md)显示走路95.47%回报增量来自身体朝向项、根位置项下降，最终H32走路根预测41.371mm/CV9.665mm。两项支持本次时域试验，不证明短时域必然有效；不重复已完成诊断、不接入未合格D05B。

下方R03/V02为已完成历史证据；当前训练以上方PHUMA R01为准，R04V2保持停止。

2026-10-24 09:28:12实核：[R03原固定追加预算已完成](README_SUPERMIMIC_Q1_R03_CONTINUATION_20261023.md)。06:08:22训练退出0，06:11:04八阶段流水线完成；追加12000轮/192000转移，累计15000轮/240000转移，训练约6小时48分。13项保存审计、两份完整动力学评测/独立重算、61项冻结条件、八个原生退出0事件与summary重算通过。初始三片轨迹大小/SHA及标量与R02B final一致。本次只读复核，没有重复重型张量审计或新训练。

三片全身RMSE走路0.363888→0.188527、浅蹲0.386939→0.201056、开发出拳0.311051→0.262525rad，均完整且不早停，但全未达0.15门。走路腿部0.178007→0.176143rad几乎不变，最大水平离起点距离0.030115→0.029120m/参考1.126480m，全根位置RMSE0.629545→0.645016m；全身改善主要来自上肢。三片腰部误差变大。已有学习收益，仍不是可用移动跟随；出拳为复用开发数据，无独立未见测试/真机验证。

final服务器`data/supermimic_for_q1/R03_B128_extend12000_20261023/run/final.pt`为205225747字节/SHA`10d0838663b5d24215f493ce0738228597452e899b1a378df6d6b453b7e10ea7`，原件LOCAL_ONLY。小证据`data/manifests/supermimic_q1_R03_completion_verification_20261024.json`，读取日志`logs/session_records/supermimic_q1_R03_completion_read_20261024.log`。观察时controller762428为Z、worker不存在，GPU38摄氏度/空闲17788MiB、容器25031303168字节/OOM14。连接经Windows跳板恢复18864转发；断线没有触发重启。

[V02/R03完整回放](README_SUPERMIMIC_Q1_VISUAL_R03_20261024.md)已完成：三视频826/579/339帧，ffprobe/SHA、九帧目视与浏览器三动作播放通过，桌面/手机无溢出。新入口`http://127.0.0.1:18865/`，原18863为R01；视频与评测一致、走路仍近原地。09:52收尾冻结条件通过，GPU38摄氏度/空闲17788MiB、容器25255591936字节/OOM14。小证据`data/manifests/supermimic_q1_V02_R03_verification_20261024.json`，大视频LOCAL_ONLY。下一核查参考、奖励和世界模型动作响应后选择独立改进，不自动追加R03预算。[D04](README_SUPERMIMIC_Q1_REFERENCE_DYNAMICS_20261024.md)的参考支撑/自碰撞缺口保留；[D05B](README_SUPERMIMIC_Q1_SUPPORT_PROJECTION_20261024.md)无训练资格，不能接入。已完成D04/D05/D06不重复，冻结源码/输入保留。

主机huanyu:/opt/data/private/YuanQi_Q1；Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。当前PHUMA训练发布`ed4217384e7abfc8a9cea0b23b95592d4a3e19b0`，Mac代码副本`/tmp/yuanqi-supermimic-q1-20261023`，分支Q1。R01旧接触、R02/R03及R04V2历史保留，A3停止。目标未完成，不把训练完成或审计通过当作可用策略。下方A3/教学条目为其他阶段历史，以各自最新入口为准。

[PHUMA D03共同轨迹模型诊断](README_SUPERMIMIC_Q1_PHUMA_MODEL_20261024.md)已发布入口首次真实执行因training/heldout与三划分标签接口不匹配退出1，模型张量尚未加载、输出未创建。新诊断标签适配已修，本地四项控制通过；D03B延后于数据规模复核，无真实模型数值结果、0新训练，test不用于方向筛选。

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

- 每日研究最近[10/24三项简报](research/2026-10-24/README.md)：PhysibleMDM参考可行性、MPBRS折扣/许可边界、DiffDreamer模型分工；53请求52成功、8摘要/3相关正文/4固定文本/111条去重，无新假设或科学计算。最新Q1背景核至cf61084保存记录，不把02:38快照当实时；主线以Q1阶段为准。通用研究main ce88294与Q1 229abe5研究文档已推送，09:25:57独立ref/各五份固定内容核验通过，见[检查](README_RESEARCH_CHECK_20261024.md)及[回执](../data/manifests/research_watch_sync_20261024.json)。
