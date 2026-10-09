# Q1 数据集盘点与离线重定向测试

记录日期：2026-10-09。实验：E001（八个已有样本对照）。状态：离线数值测试与诊断完成；高保真全身跟随未通过。用户要求：说明当前重定向，测试服务器已有 AMASS、BONES-SEED、PHUMA，并核验更新且合适的数据集。

## 目标与范围

Q1 最终目标为 SMPL 驱动全身跟随，包括腿部和移动。本阶段只做小样本离线输入、运动学 IK 与固定骨盆支撑 PD 对照；没有训练或真机控制。大文件保留 hp3090，Mac 只保存代码、README、清单和小型预览。

## 已完成盘点

- 当前实际调用 `script/q1/q1_retarget.py`，Mink 1.2.0 / quadprog、14 个身体目标、22 个 Q1 关节、首帧相对诊断校准；没有调用 GMR、Q1 SDK 或 SONIC policy。配置见 `retarget_config.json` / `sim_config.json`。
- hp3090 的 Q1 根为 `/media/yu/FAFF-E977/YuanQi_Q1`，分支 `Q1`，测试基线提交 `ad15839b0600d40d72e36cb82dcba4abb065d70c`。保留既有 PredActor 两份暂存修改。
- AMASS 候选：`/media/yu/FAFF-E9771/data/project_snapshots/HUMAN_PLUS_A3/2026-08-09/datasets/amass_sources`。四份 stageii NPZ 为 SMPL-X、165 维 poses、16 维 betas、120 Hz，动作是站立、左转走路、前踢、站立摆臂。另在 `data/humanplus-phuma/assets/human_motion/ACCAD` 找到原始人体样本；此目录不能因项目名而认定为 PHUMA 筛选结果。
- SMPL-X 模型与 Python 库已存在服务器；模型位于 `/media/yu/FAFF-E9771/data/humanplus-phuma/assets/body_models/smplx`，无需下载或移到 Mac。
- BONES-SEED 当前找到的 `.../shooting-for-contact/trajectories/g1/bones_seed` 是已经重定向的 G1 CSV，不是人体 SMPL。使用同项目 G1 MJCF 做 FK，再将具名身体目标送入 Q1 IK；不会直接截取 G1 关节角当 Q1 关节角。原始 SOMA/BVH 全集是否在服务器仍在查找。
- PHUMA 找到既有 X2 转换库与迁移清单；原清单记载 G1 数据 73,076 个 NPY / 2,205,637,464 字节。本轮没有定位到清单所列完整 G1 原库，测试的是 `phuma_x2_hybrid1200_clean291_50fps` 中两段 X2 衍生产物，不能代表原始 PHUMA。精确匹配的模型是 `data/x2-sonic-sim/assets/agibot_x2/x2_ultra.xml`（31 关节）；27 关节 fist 模型被明确排除。

## 关键决定与边界

优先用较干净的人体数据绕过 FGP，判断昨日幅度不足是输入误差还是重定向限制。新适配器区分 SMPL-X、G1 和可能的 X2 产物；必须保留 betas、时间与根轨迹，核验身体局部坐标与世界坐标，不直接复用衣服录制的 Y-up 假设。

保持当前求解器和速度限位，以便可比；首帧相对校准会把首帧映射为 Q1 中立位，因此残差低不等于同姿保真。固定骨盆 PD 不验证移动、地面接触或动态平衡。

## 实测结果

下表的误差是 Q1 FK 对**首帧相对校准后目标**的 14 身体位置残差 P95，不是人体重建误差，也不是机器人对原人体同姿误差。

| 样本 | 50 Hz 帧数 / 时长 | IK 位置 P95 | 支撑 PD 最大单关节 RMSE | 限位违例 |
| --- | --- | --- | --- | --- |
| AMASS 站立 | 150 / 2.98 s | 0.73 cm | 0.0071 rad | 0 |
| AMASS 走路左转 | 396 / 7.90 s | 4.24 cm | 0.0458 rad | 0 |
| AMASS 前踢 | 115 / 2.28 s | 5.57 cm | 0.0575 rad | 0 |
| AMASS 站立摆臂 | 287 / 5.72 s | 1.97 cm | 0.0107 rad | 0 |
| BONES-SEED G1 慢跑（已有裁切） | 178 / 3.54 s | 6.90 cm | 0.0729 rad | 0 |
| BONES-SEED G1 爬行（已有处理） | 366 / 7.30 s | 11.77 cm | 0.0483 rad | 0 |
| PHUMA→X2 走路片段 | 56 / 1.10 s | 2.95 cm | 0.0241 rad | 0 |
| PHUMA→X2 弯腰 | 199 / 3.96 s | 17.01 cm | 0.0422 rad | 0 |

合计 1,747 个参考帧、34.78 秒。八段身体输入、IK、固定骨盆 PD 均完成，有限数值和关节限位检查通过；没有将八段记为高保真跟随通过。已有单元检查 23 项通过；SMPL-X 每个样本三帧的快速关节 FK 对完整官方库 mesh forward 的误差均小于 1e-5 m。首次完整 forward 检查遇到 expression 的 batch 维不匹配，已通过显式扩展零 expression 修正，失败与重试日志均保留。

## 关键失败结论

### 1. 干净输入仍被压缩，不只取决于 FGP

AMASS 前踢右膝几何屈伸范围约 86.69°，Q1 输出约 32.93°；慢跑源左右膝范围约 86.99°/92.29°，Q1 只有约 27.89°/16.50°。这些角度由具名 hip-knee-ankle 点计算，是跨骨架的运动幅度诊断，不能当严格关节角一一对应。

前踢第一帧人体膝已经弯曲 43.36°/36.40°，但当前算法把首帧映射到 Q1 中立位（几何约 9.85°/9.85°）。六个测试触及配置的 2 rad/s 速度上限。证据支持校准/映射/速度约束是当前瓶颈之一，不能据此断言 Q1 硬件抬腿能力不足，也不能认为更换大数据集就能解决。

### 2. 首帧完整旋转对齐把水平位移混进竖直方向

当前 `q1_retarget.py` 使用 `alignment = robot_root_rotation @ human_cal_rotation.T`，同一完整 SO(3) 对齐既用于姿态又用于世界位移。如果首帧骨盆倾斜，它会旋转世界重力轴。

| 样本 | 按当前体型比例缩放的原始世界 Z 范围 | 当前 Q1 根部 Z 范围 | 水平位移贡献到 Q1 Z 的范围 |
| --- | --- | --- | --- |
| AMASS 走路左转 | 0.036 m | 0.231 m | 0.195 m |
| BONES-SEED 慢跑 | 0.049 m | 0.739 m | 0.732 m |
| BONES-SEED 爬行 | 0.102 m | 0.846 m | 0.854 m |

独立脚本用保存的 source / qpos 精确重建该公式，误差小于 1e-5 m。这是重定向坐标校准问题的明确证据；爬行还有大姿态变化，不能用当前首帧中立校准评价原动作可执行性。本阶段保留问题版本与证据，尚未更改重力轴/校准行为；这些 TEST_ONLY 轨迹不可提升为训练数据。

### 3. 支撑 PD 不跟踪上述根轨迹

PD 使用 pelvis weld；根部运动仅存在于 IK qpos，支撑回放实际固定骨盆。根位移非零不是自主行走通过，低 PD RMSE 也不验证落脚、接触或动态平衡。四张真实 Q1 mesh 图已检查；源人体与 Q1 的展示世界航向/尺寸不同，不把图上距离直接当误差。旧抽帧规则按手腕世界位移选峰，走路/慢跑会有重复末帧，不能用这些抽帧推断完整相位。

## 实现、版本与复现

- 新增 `dataset_input.py`：SMPL-X 保留 gender/betas/全部 pose 字段，使用 smplx FK；原 AMASS Z-up 世界坐标保持，身体局部轴单独转换。G1 CSV 用匹配 MJCF FK；PHUMA→X2 用具名 31 关节、xyzw 根姿态与匹配 ultra 模型 FK，再统一到 Q1 的 14 身体合同。
- `validate_egolocate.py` 新增 `--input-format dataset-reference`，默认衣服输入入口保留；输出几何幅度与根轨迹统计。使用原有 Q1 模型和参数，未切换 GMR/PhySINK 或安装环境。
- `run_dataset_tests.py` 与 `dataset_test_cases_20261009.json` 保存精确输入、命令、日志、超时与恢复检查；`summarize_dataset_tests.py` 验证工件哈希及根部重力轴问题。
- 测试基线：`ad15839b0600d40d72e36cb82dcba4abb065d70c`；本轮代码内容 SHA 与包版本在每段 summary / 清单中，发布提交由 Git 历史提供。

```bash
ssh hp3090
cd /media/yu/FAFF-E977/YuanQi_Q1
PY=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" script/q1/run_dataset_tests.py \
  --output-dir data/experiments/q1_datasets_20261009 \
  --log-dir logs/q1_datasets_20261009
```

已完成目录允许核验后恢复；保留不完整/版本不同工件，不覆盖。重新运行对照时改用新的输出与日志目录。根部诊断的复现入口：

```bash
"$PY" script/q1/summarize_dataset_tests.py \
  --suite-dir data/experiments/q1_datasets_20261009 \
  --manifest data/manifests/q1_datasets_20261009_recheck.json
```

## 产物与日志

- 完整轨迹：hp3090 `data/experiments/q1_datasets_20261009/<case>/`；包含身体参考、IK qpos/残差、支撑关节/力矩记录、summary 与部分抽帧。
- 总结果与校准诊断清单：`data/manifests/q1_datasets_20261009.json`，记录原始输入、模型、代码、产物大小/SHA-256；形体参数数值只留原文件和服务器转换 sidecar，Git 清单仅保存数目与指纹。派生结果 `suite_summary.json` 在实验根目录。
- 幅度/根高度对照图：`data/experiments/q1_datasets_20261009/retarget_diagnostics.png`，身份见 `data/manifests/q1_dataset_plot_20261009.json`。它展示运动学诊断，不是落脚或平衡评测。
- 终端：`logs/q1_datasets_20261009_suite.log`、`logs/q1_datasets_20261009_diagnosis.log`、`logs/q1_datasets_20261009/`；站立单独首跑/重试日志也保留。
- Mac 小摘要/四张图：`data/sync/q1_datasets_20261009/`，没有下载原始动作全集、SMPL-X 模型或大型轨迹。当前不需要用户迁移新大文件。
- 新数据与训练资料结论见[研究 README](README_Q1_DATASET_RESEARCH_20261009.md)：优先 AMASS 基线、MOSAIC 原人体光学小样本，再考虑原始 BONES/SOMA、HiPHI 与 PHUMA 物理筛选；HumanTracker 为后续评测，HumanVerse 未核验到发布资产。

## 保存与同步

本阶段报告、脚本、清单已保存，测试进程已结束。所有大型产物状态为 `LOCAL_ONLY`。发布流程为审核 diff / 暂存区、Git payload audit、仅本轮路径提交、显式推送 `origin Q1` 并核验远端 SHA；发布证据另存 `logs/session_records/q1_datasets_publication_20261009.json`，未推送前不记为已同步。保留既有 PredActor 暂存修改，未提交到本轮。

发布核验：结果提交 `90aed9e0cce60511e44734f8aa019c2d358605d7` 已推送 `origin/Q1`，2026-10-09 08:33:32 北京时间核对本地 HEAD 与远端 `refs/heads/Q1` 一致；15 个本轮文件已发布，既有两份 PredActor 暂存修改保留。上述同步回执由后续独立提交保存。

## 下一步

第一步先分离世界航向对齐与身体姿态校准，保留重力轴；再使用可复现的中立/配对姿态校准对照，检查同姿与幅度。随后在同一 AMASS 样本上建立 Q1 的 GMR 或 PhySINK 对照，处理地面/接触，最后进入无支撑跟踪策略训练。不能将本次支撑测试当作训练、平衡或真机验收。
