# CLOTHO 实时部署分支核查

记录日期：2026-11-13。范围：通过已认证 GitHub API 读取作者仓库的三个分支、说明及关键源码；本轮没有安装、模型调用、BLE、标定或机器人控制。

## 给用户的结论

此前只核对了 main 和用户提供的旧 ZIP，漏查了作者的 **fgp_realtime** 分支。该分支在 2026-11-09 发布了 CPU ONNX 部署，明确列出两套完整资源目录 onnx_s1 与 onnx_cano_s_ep120_fast。这才是应优先核对的现成 CPU 部署路径。此前“范围内没找到 FGP-S 权重”仅适用于旧提供包；不能由此推广为作者仓库没有相关可部署版本。

当前设备实际仍运行 U100h B＋GlobalPose 的 PyTorch CPU 路径，没有换成此分支的模型/ONNX/物理链。旧 780.54 ms 和本次分段计时都不代表新部署版性能。作者文档的 30 Hz 是该版本的运行目标，不是本项目已在 Orin 实测通过的速度。

## 三个分支

| 分支 | 实核提交 | 内容与适用范围 |
| --- | --- | --- |
| main | bed3424b86caa5849c75b5046252331f73cdce20 | 旧提供 ZIP 对应版本；默认 U100h B 去噪＋GlobalPose，也有修复后的 LIP 与 online_guarded |
| fgp_realtime | 1e719140b43a2ec4dc22f4159014b2eb9de51ed9 | 2026-11-09 12:12:49 UTC，CPU ONNX、CANO→LIP→GP 速度网络→原生物理，独立 Windows 部署 |
| fivepoint-physics_optimization | dcc6df97dad70543989f21f566f3fa9b8381a224 | 2026-11-09 08:29:49 UTC，五点静止概率/c129 物理/FGP-LIP-GRU CUDA Graph，另一条 GPU 路径 |

main 当前提交时间为 2026-10-24；不能因默认分支名把它当作仓库所有实现的最新版本。fgp_realtime 相对 main 实核 ahead 3、behind 0。仓库实核无 GitHub Release；资源由 ZIP 另外传输，不在 Git 树内。

## CPU ONNX 路径

入口为 online_fgp+LIP_realtime_onnx.py 和 online_fgp+LIP_debug_onnx.py，默认模型目录 models/onnx_s1。可切换 models/onnx_cano_s_ep120_fast。源码新增 S1 规格：4 个 canonicalizer block，384 维/4 层预训练 encoder；S 规格仍为 256 维/3 层 encoder。**目录名不替代规格，必须读取资源包内 manifest.json**，本轮未拿到 manifest 实体，不能把两套模型都断言成纯 S、也不能按文件名报参数量。

作者 deployment_assets.json 对两套目录分别列出：

| 工件 | 每套字节数 | 作用 |
| --- | ---: | --- |
| cano.onnx | 95,439,547 | 服装 IMU 规范化；两套文件大小相同但 SHA 不同 |
| lip.onnx | 8,912,335 | 人体姿态；两套清单 SHA 相同 |
| gp.onnx | 23,231,818 | 267→9 的循环速度/静止预测；两套 SHA 不同 |

另需各自初始隐藏状态 NPZ、manifest/verification JSON、SMPL、动力学缓存及原生物理库；不能只取某一个 ONNX。清单记录证明作者声明这些资源，不等于本地已持有或已核验它们。

onnx_inference.py 显式使用 CPUExecutionProvider，启用 ONNX 图优化、顺序执行、可配置 intra-op 线程，默认 CANO/LIP 为 min(12,逻辑核数)，GP 为 1。该部署入口只开放 ONNX backend，显式 CUDA 会被拒绝。导出可支持仅末帧 residual（manifest v2/cano_output=latest）；实际包是否启用以 manifest 为准，不能仅看代码分支就声称已获得相应加速。

CANO 使用 30 帧已到达历史，满窗后交付末帧并作因果 EMA；这是启动预热，不是持续等待 30 帧未来，也与 MorphMimic 的未来目标窗口不同。LIP/GP 隐藏状态与初始状态包需要一起对齐。实时链包含断流冻结、长断流重置及恢复预热。

新 GP 路径不是原 Full_GR_OV_10 姿态输出的直接替换：LIP 负责姿态，GP 循环网络处理 267 维特征，再由原生 double-tracking 物理输出姿态/平移。作者说明两个资源包分别存在训练来源 CANO 或 LIP 哈希不一致的提示，当前代码警告后继续；可运行不证明组合精度已经验证，后续比较须保留此来源差异。

## 设备、衣服与显示

作者现成部署目标为 Windows x64：主程序 Python 3.9，物理库 carticulate.pyd 要求 CPython 3.8 x64，由子进程调用。Linux/macOS 需兼容的重新编译库；不能把 Windows pyd/DLL 复制到 ARM64 Orin 就算部署完成。纯 ONNX 前向可作为后续 ARM64 分项评测，但不能据此冒充完整物理链通过。

作者实时默认输入是串口 BLE-UART 接收器，不是当前用户两件衣服的原生 Bleak 直连。现有 BLE 前端应保留，再按传感器槽位、标定、30 Hz 重采样、时间戳及丢包语义适配到新 pipeline；不能直接照抄 COM6 启动并宣称衣服已连接。Unity 包有四个播放器，但网页和 HumanFrame 不依赖 Unity，可继续使用本项目的网页预览。对接 HumanFrame 时不携带 Unity 的一米显示偏移。

## 资源与下一步

必需资源为 fgp_realtime_weights.zip；可选 fgp_realtime_viewers.zip 与 fgp_realtime_offline_data.zip。它们与现有 checkpoints.zip 是不同的交付物。README 没有提供新 ZIP 的公开下载链接，Git 树及 releases 也没有这些二进制。本轮在 /Users/yu/Human+ 范围文件名检索没有找到对应新包；不据此断言其他设备或位置不存在。

后续应先取得完整资源包并按 deployment_assets.json 核 SHA、包内 manifest/来源配对，再用同一保存 IMU 分别测 CANO/LIP/GP/物理及完整时序，不沿用 0.78 秒，不只看前向均值。先离线与网页，不接电机。Windows 可先验证作者原生完整链；Orin 需另解决物理库 ARM64 构建，或评估 fivepoint 分支的 GPU 路径。当前不改现有设备默认模型/线程，不启动 BLE/标定/硬件或旧 Q1 训练。

## 来源与保存

结构化来源：[x2_clotho_realtime_branch_review_20261113.json](../logs/session_records/x2_clotho_realtime_branch_review_20261113.json)。旧计时更正：[FGP资产核查](README_X2_FGP_S_ASSET_AUDIT_20261113.md#1506-权重未替换与分段耗时复核)。

作者仓库：humanplus-ai/CLOTHO-mocap。通过 GitHub contents API 按上述完整提交读取 README、cano_model_config、onnx_inference、deployment_assets、check_deployment、realtime 入口、GP ONNX/物理，以及 fivepoint 的 README_INFERENCE/README_RECEIVER；没有把私有第三方完整源码或模型提交进本仓库。main 的八份源码/说明与本地逐文件 SHA 相同。此阶段只读核查，不是新版本实际运行验收；阶段记录设备镜像待补。
