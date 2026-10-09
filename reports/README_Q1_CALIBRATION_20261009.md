# Q1 重力轴修复与合成站姿校准对照

记录日期：2026-10-09。实验 E002。状态：根轨迹坐标修复通过；合成站姿实验完成，整体姿态跟随未验收。

## 目标与范围

按用户要求继续 Q1 全身 SMPL 跟随，先隔离坐标/校准错误，再做 GMR 与新方法对照。只进行离线 IK 与固定骨盆 PD，没有训练或真机操作；旧 E001 产物完整保留。大型产物在 hp3090，Mac 仅小型代码、清单与 README。

## 完成内容与决定

- `q1_retarget.py` 将世界根位移对齐与身体姿态校准分开。根位移默认只绕世界 Z 轴对齐航向，保持重力方向；身体姿态仍按原配对旋转处理。旧完整 SO(3) 位移方式保留为 `full_rotation_legacy_diagnostic`，只用于复现实验。
- `dataset_input.py` 新增同一 gender/betas 模型合成的直立、双臂向下姿态；航向用源首帧的世界 Z 轴航向，平移原点锚定源首帧骨盆。快速 FK 对官方 SMPL-X forward 核验。它是模型合成配对，不是动捕服与真机实测配对。
- `validate_egolocate.py` 增加显式配置和校准模式，并记录实际使用的配置 SHA；默认仍为首帧相对诊断校准，不把合成站姿实验宣称为已解决全身映射。
- `run_calibration_comparison.py` 使用同一输入与限速，比较 legacy / gravity_fixed / standing_pair 三组。四段原始 AMASS 每段三组，BONES-SEED G1 慢跑只比较前两组，共 14 组。没有直接套用人体模型校准 G1 产物。

## 实测结果

25 项既有及新增单元检查通过，包括倾斜校准下竖直位移不串轴、竖直前向轴的航向处理。14/14 离线评测完成、关节限位违例均 0。五组 legacy 的完整 qpos 对 E001 保存轨迹最大绝对差均为 0，证明基线精确复现。

| 样本 | 原方式根 Z 范围 | 修复后根 Z 范围 | 缩放源 Z 范围 |
| --- | --- | --- | --- |
| AMASS 站立 | 0.0024 m | 0.0013 m | 0.0013 m |
| AMASS 走路转向 | 0.2307 m | 0.0361 m | 0.0361 m |
| AMASS 前踢 | 0.0473 m | 0.0500 m | 0.0500 m |
| AMASS 摆臂 | 0.0013 m | 0.0011 m | 0.0011 m |
| BONES-SEED G1 慢跑 | 0.7389 m | 0.0493 m | 0.0493 m |

全部九组非 legacy 结果的根高度增量与缩放源高度增量最大差约 5.6e-17 m。仅修复世界位移时，关节运动/身体相对残差不变，这是本次单变量对照的预期结果；不把它记为抬腿问题已解决。

| AMASS 样本 | 首帧相对 IK P95 | 合成站姿 IK P95 |
| --- | --- | --- |
| 站立 | 0.73 cm | 4.54 cm |
| 走路转向 | 4.24 cm | 5.60 cm |
| 前踢 | 5.57 cm | 11.21 cm |
| 摆臂 | 1.97 cm | 4.85 cm |

前踢右膝几何屈伸范围由 32.93° 增到 63.52°，源为 86.69°。幅度改善伴随身体目标残差增大，不能宣称整体保真通过。几何角度不是不同骨架关节的一一对应误差；两组目标校准不同，P95 也不能当对同一原人体姿态的准确率排名。

合成站姿的前踢首帧 Q1 膝几何角仍约 9.86°；评测从 Q1 中立位开始、受 2 rad/s 约束，没有预热到源初始屈膝姿态。启动过渡、体段映射、目标权重和真实骨架差异仍待隔离；不能把剩余问题直接归因于 Q1 太小。

## 版本与复现

- 主机：`hp3090`，根 `/media/yu/FAFF-E977/YuanQi_Q1`，分支 `Q1`。
- 实验基线：`1e2932df9ebd784cfd584d2bd0caa349c6567842`；本次运行代码/配置的 SHA 在每段 summary 和比较清单中。最终发布提交由 Git 历史提供。
- 原输入：`data/experiments/q1_datasets_20261009/<case>/body_reference.npz`，原人体形体值保留服务器，不进入公开清单。

```bash
ssh hp3090
cd /media/yu/FAFF-E977/YuanQi_Q1
PY=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
"$PY" -m unittest discover -s script/q1 -p 'test_*.py'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" \
  script/q1/run_calibration_comparison.py \
  --output-dir data/experiments/q1_calibration_20261009_E002 \
  --log-dir logs/q1_calibration_20261009_E002
```

恢复已有目录时核验代码、配置及工件 SHA；不同版本或不完整目录保留，新的独立对照使用新目录。

## 产物与日志

- 轨迹、配置与总结果：`data/experiments/q1_calibration_20261009_E002/`，每段含身体/关节参考、IK trace、支撑回放与 summary。
- 可发布比较清单：`data/manifests/q1_calibration_20261009.json`，为已核验 `comparison.json` 的精确副本，不含形体参数值。
- 完整子进程 stdout/stderr：`logs/q1_calibration_20261009_E002/<case>_<mode>.log`。
- 小型检查记录：`logs/session_records/q1_calibration_checks_20261009.log`；Mac 小副本 `data/sync/q1_calibration_20261009_E002/`。

## 边界与下一步

根轨迹修复是运动学坐标核验，固定骨盆 PD 没有跟踪根平移，也没有验证落脚、打滑或动态平衡。全部产物 `TEST_ONLY`，训练/真机验收未完成，备份 `LOCAL_ONLY`。

下一项先以相同 AMASS 样本建立 Q1 的实际 GMR 对照，独立登记骨架配置与限速差异；NMR 等方法先核验公开代码、权重、许可与目标骨架，不能把 G1 的学习模型直接当 Q1 模型。GMR 和新方法的研究/运行结果另写阶段 README。

本阶段报告、代码、清单与入口已保存。按授权审核 diff、暂存区与 Git payload 后仅提交本轮路径，推送 `origin Q1` 并核验远端。同步结果由后续发布回执记录，未核验前不记为已同步；既有 PredActor 暂存修改保留。
