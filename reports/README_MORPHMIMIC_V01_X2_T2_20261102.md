# MorphMimic v0.1 X2 Ultra / T2 权重测试

用户要求优先测试另外两个机型，并明确同意直接测试 v0.1 发布权重。本阶段不训练；分别验证真实 SOMA 到 X2 Ultra/T2 的重定向、发布 ONNX 的动力学跟踪和本地原速视频。作者训练包含这两个机型，不能将本测试称为未见本体零样本迁移，也不能当作我们已完成共享训练复现。

## 固定来源

- 发布页：`https://github.com/lzyang2000/morphmimic/releases/tag/v0.1`。
- tag 对象 `70511b3da574403edbc2c89cc84d793a6182898b`，peeled commit `41c8382f127a6f5109bfe047445a9180f71f1fc0`。此前研究所记后续 main 不作为该 tag 的源码。
- `morphmimic_policy.zip` 63372873B，SHA256 `be874424beeda23cbccda08ed65b3fa146e926d2f338b8f898d28baa21d1b99f`。
- 实际 `policy.onnx` 69994291B，SHA256 `cf95194793e4732447f66432fc2381d82dd7c3d82c94da15ad074f2cfe295512`；与包内 `SHA256SUMS` 一致。`MANIFEST.txt` 中嵌入的文件哈希陈旧，与实际及 `SHA256SUMS` 不同，原文件保留，不据其文字代替校验。
- 包内自述 unified/share all、iteration60000、goalv2；实际只发布部署 policy/机器人 JSON，没有可独立审计的 WM/critic 或训练 `.pt`。不将说明文字视为这些缺失网络的内容审计。
- SOMA retargeter 固定 `cbdb2b494a2af82137bf9b6f15122d60167281c4`，使用 v0.1 原 `assemble_robots.py` 的软限位和 distal effector 修正、原 batch 配方及 converter。

全部模型、源码归档、大 NPZ 和 BVH 均留用户25386的 `data/references/morphmimic_v01_x2_t2_20261102` 或独立评测根，标记 SERVER_LOCAL_ONLY，未做网盘备份。

## 已发现并纠正的接口试跑问题

先前试跑将 G1 NPZ 关节改名/补零给 X2/T2，同时保留 G1 身体位置。这不是真实机器人重定向，身体几何和 T2 关节轴不一致。原 `MORPHMIMIC_V01_X2_T2_EVAL_20261102/synthetic_npz`、两模型导出和 X2 视频/JSON全部保留为接口烟测；X2 的1032帧及其误差不能用于评判策略质量，T2当时只完成资产导出，未进行动力学回放。

该试跑未挂项目资源守卫，且 `--bake` 改写了新隔离 v0.1 副本中的两个 JSON；这是执行偏离。原 aa40/G1 训练冻结源码未改。后续仿真须挂原共享 GPU 锁、24GiB准入/30GiB运行内存、boot/GPU/OOM/温度/磁盘保护和进程回收，运行配置另存，不继续改发布包原件。

原 SOMA 首次 assemble 因 LFS URDF 指针解析失败，取回精确 X2 URDF 后原 assemble 已完成；首次 converter 因 `newton` 缺失退出，尚未生成真实 X2/T2 CSV。错误和目录保留。retargeter要求 Python>=3.12/Newton1.3，而原训练环境Python3.11，下一步另建隔离环境，不升级原训练环境。X2必要 LFS mesh 仍须按指针 SHA/大小逐项补齐；T2资产来自 v0.1 vendored 文件。

固定首片是既有真实 SOMA `220713/walk_forward_loop_001__A022.bvh`，原 G1 转换为1032帧/50Hz；不能将此数值直接当作新重定向已完成。保留559条源 BVH，不重复提取或按结果换片。

## 原 G1 分支

G1原训练已跑满160000轮，12编号checkpoint内容审计通过，最终motion三片完整，详见 [G1记录](README_MORPHMIMIC_G1_EVALUATION_20261102.md)。为优先满足用户X2/T2要求，剩余keypoint评测经SIGTERM停止，guard为STOPPED_SIGNAL；controller因人工停止记通用FAILED，不是训练失败。6份最终视频全组尚未验收，保留未完成状态，不重启训练。

## 当前状态与下一步

原生 hp_han Windows 两段式 SSH 会话81774已再次返回唯一根；上下文前快照 GPU15MiB/无计算/OOM0。运行前须重新核实时变资源。先保存本阶段记录，再补独立retargeter环境和真实参考；通过输入/模型一致性核验后，X2与T2串行运行发布策略，保存真实失败、指标和原速视频，最后取回Mac并核大小/SHA、解码与抽帧目视。当前没有新的有效X2/T2动力学质量结论。

不创建/结算goal，不启agent，不改其他自动任务；衣服Q1演示继续搁置，用户采纳仍为null。发布状态以独立远端和三端旁收据为准。
