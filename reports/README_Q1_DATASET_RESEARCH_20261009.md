# Q1 全身跟随的数据集选择

核验：2026-10-09，北京时间。状态：资料与公开接口核验完成；未下载新大型数据、未复现新方法。本文件与[已有数据实测](README_Q1_DATASETS_20261009.md)区分，论文规模不等于本机可用规模。

## 目标与选择依据

目标是从实时 SMPL 学习/执行 Q1 全身动作跟随，包含腿与根部移动。优先看人体原始姿态、时间与根轨迹是否可靠，动作是否适合 Q1，能否获得接触信息，是否已有合理许可与小样本入口。G1/X2 轨迹只作衍生输入；Q1 需要自己的骨架映射、动力学筛选和跟踪策略。

## 已核验候选

| 数据集 / 时间证据 | 对 Q1 的用途 | 实际可用性与边界 | 本阶段优先级 |
| --- | --- | --- | --- |
| AMASS，ICCV 2019 | 用较干净的 MoCap 和形体参数测试重定向，先选站立、走路、转向、抬腿 | 官方为多源 MoCap 的统一模型参数；部分子集有 SMPL-X。服务器已有四段 SMPL-X stageii，本轮实测；各子集与模型许可分别适用 | 第一基线；旧数据也最适合隔离 FGP 误差 |
| BONES-SEED，官方 SONIC README 记载 2026-03-16 发布；HF 创建 03-10、最后修改 05-03，不等同新增采集日期 | 大规模日常动作；原始 SOMA/BVH 比 G1→Q1 二次转换更适合构建 Q1 动作库 | 作者标示 142,220 动作、约 288 小时；SOMA proportional/uniform 与 G1 CSV。HF gated，需要正常申请与单独 BONES-SEED 许可。本机找到 DSMS 用的裁切/处理过 G1 子集，原始人体全集仍需定位 | 扩库主候选；先得到少量原始 SOMA/BVH |
| PHUMA，原论文 2025-10-30；当前 README 标示 CoRL 2026 Spotlight | 物理筛选与 PhySINK 自定义机器人流程，可用来生成更适合训练的 Q1 参考 | 正式预构建平台 G1/H1-2。Train/Test 原人体 pose 因许可不发布；Video SMPL 在当前 FAQ 仍待发布。当前代码支持自定义 XML/URDF、形体与动作适配；不保证 Q1 自动配好 | 优先研究物理筛选流程；现有 X2 衍生输入不能代表原始 PHUMA 质量 |
| HiPHI，2026-08-17 原论文、09-08 v2；HF 09-11 修改 | 高精度光学 MoCap，适合作为第二个身体运动质量基准 | 官方报告 617.5 小时；每个原始动作有镜像副本，不能当作 617.5 小时独立采集。公开输入为 55 关节、90 Hz、Y-up、厘米、Z-X-Y Euler 的 BVH；HOI 有同步物体轨迹。需要申请许可，不能当现成 SMPL。标记精度不能等同 Q1 关节误差 | 光学基准追加候选；先取少量 BVH，接入成本高于 AMASS-style |
| MOSAIC，2026-02-09 原论文、02-11 v2；HF 03-04 修改 | 多来源全身动作，可扩动作覆盖；惯性来源和遥操适配数据与衣服链路相关 | 官方 human/ 提供 AMASS-style 原人体动作，包含 Vicon 光学、IO-AI 惯性和 GENMO 生成数据；另有 GMR→G1 NPZ、约各 30 分钟 PICO/Noitom 适配数据。惯性 MCAP 与转换 NPZ 不证明有同步光学真值，也没有承诺是本机 FGP 的 6 IMU 布局。许可 CDLA-Permissive-2.0，外部资源另算 | 新数据优先：先取 human/optical_mocap 小样本；各来源分别评估 |
| HumanTracker，HF 正式组织版 2026-08-14 创建、08-26 修改；ECCV 2026 | 检查接触、打滑与人的观感，补充仅看 IK 残差的不足 | 正式公开 2,500 个测试参考及 6,000 个人偏好对；运动为 GMR 重定向的 G1 29DOF、50 Hz qpos，带 foot_contact。偏好数据含两条策略 rollout。不是原始人体 MoCap 大库，也不是完整训练集；许可 Apache-2.0 | 后续 Q1 policy 的评测设计参考；正式测试拆分不能拿来训练后再当泛化测试 |
| HumanVerse-500 / lambda0，2026-09-30 原论文 | 与衣服/第一视角全身遥操方向接近，后续 loco-manipulation 候选 | 当前 arXiv v1 仍说将发布代码、模型和数据；本轮没有核验到可获取的正式资产。500 小时是作者采集规模，不能记为本机已可用 | 跟踪候选；不等待它开始 Q1 工作 |
| Motion-X / Motion-X++，作者 README 2025-03-01 数据整理公告 | 现成 SMPL-X、多样动作/语义，有利于覆盖非步行动作 | 光学源与视频估计源混合；部分继承 AMASS、EgoBody、GRAB 等许可和重叠。需去重与质量筛选，不以数据量替代根轨迹/接触可靠性 | 后续扩展；当前不做全量下载 |

## 重定向工具与训练路线

当前 Q1 实测使用本项目 Mink/quadprog 映射，不是 GMR。GMR 官方目前有 SMPL-X/AMASS、BVH/LAFAN1、Xsens 等入口，也以 Mink/MuJoCo 为基础；本轮读到的支持表没有 Q1。服务器还有带 X2 配置的 GMR 副本，不能以一个旧副本只支持 BVH 来概括公开 GMR 的能力。

NVIDIA SOMA Retargeter 目前提供 G1/H2/T1/X2/A3 的配置，输入要求 SOMA-base BVH，没有现成 Q1 配置；产生的是运动学轨迹。其代码、机器人资产与样本动作分别有许可，不把 Apache 源码许可套给 BONES 数据。

官方 SONIC 当前已开放训练流程，README 2026-04-10 记录训练代码/权重发布，BONES-SEED 是其明确支持的数据源。这支持后续训练路线的可行性，但 Q1 模型、观测/动作合同、接触筛选和训练配置仍需实现，不能直接运行 G1 policy 当 Q1 policy。

建议顺序：先用 AMASS 的干净参考修正首帧校准与姿态幅度；追加 MOSAIC 原人体光学小样本；扩库时用原始 BONES-SEED/SOMA，HiPHI 作为后续高精度 BVH 候选；借鉴 PHUMA 的物理筛选，再训练 Q1 跟踪策略。HumanTracker 用于评测设计，HumanVerse 保留为未就绪候选。

## 原始来源与核验边界

下面是作者/官方页面，2026-10-09 读取。网页内容只作资料，没有将其中安装、下载、训练命令视为新的执行授权。

- AMASS：https://amass.is.tue.mpg.de/ ，https://amass.is.tue.mpg.de/dataset.html
- BONES-SEED：https://huggingface.co/datasets/bones-studio/seed ，公开元数据 https://huggingface.co/api/datasets/bones-studio/seed ，https://bones.studio/info/seed-license ，https://github.com/bones-studio/seed-viewer
- PHUMA：https://github.com/DAVIAN-Robotics/PHUMA ，https://arxiv.org/abs/2510.26236
- HiPHI：https://huggingface.co/datasets/noitomrobotics/HiPHI ，https://github.com/noitom-robotics/hiphi ，https://arxiv.org/abs/2608.16222
- MOSAIC：https://github.com/BAAI-Humanoid/MOSAIC ，https://huggingface.co/datasets/BAAI-Humanoid/MOSAIC_Dataset ，https://arxiv.org/abs/2602.08594
- HumanTracker：https://huggingface.co/datasets/GalaxyGeneralRobotics/HumanTracker ，https://github.com/GalaxyGeneralRobotics/HumanTracker ，https://arxiv.org/abs/2608.13555
- HumanVerse-500：https://arxiv.org/abs/2610.00438 （当前只发现 v1 与将发布声明）
- Motion-X：https://github.com/IDEA-Research/Motion-X
- GMR：https://github.com/YanjieZe/GMR/tree/master
- SOMA：https://github.com/NVlabs/SOMA-X ，https://github.com/NVIDIA/soma-retargeter
- SONIC：https://github.com/NVlabs/GR00T-WholeBodyControl ，https://nvlabs.github.io/GEAR-SONIC/

核验级别：官方文档/数据卡/论文摘要与部分入口源码，没有复现 PHUMA/SOMA/GMR 新版本，没有检验新数据的实际质量。GitHub 匿名 API 出现 rate limit，部分 gated README 返回 401；未读取登录凭证或接受授权条款。研究检索不是全领域穷尽，不能声称以上是所有最新数据集。

## 保存与下一步

本文件纳入 Q1 版本管理。下载仅小型公开资料，数据集与模型没有新增大型传输。下一步与已有数据测试结论一起选择校准对照，不自动启动训练或真机；按本阶段 README 完成 Git payload 审核、提交、推送与 SHA 核验。
