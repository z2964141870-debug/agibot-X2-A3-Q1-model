# Q1 衣服与 FGP 导出 SMPL 的位置和信息缺口

记录日期：2026-10-10。阶段：Q1-SMPL-INVENTORY。状态：已有文件位置、字段和适用范围复核完成。

## 目标与范围

回答已有衣服经 FGP 导出的 SMPL 是否存在、位于 Mac 还是 hp3090，以及用于全身跟随时主要还缺什么。只读取已有文件和源码，不重新推理、重定向、训练或操作真机；不迁移大型文件。

## 已完成：服务器重新生成的两段

项目根：hp3090 `/media/yu/FAFF-E977/YuanQi_Q1`。以下四个 NPZ 在本轮实际打开并读取字段，不只依据旧日志判断存在。

| 动作 | 主输出完整路径 | 帧数 | 字节数 |
| --- | --- | --- | --- |
| 走路 E003 | `/media/yu/FAFF-E977/YuanQi_Q1/data/experiments/q1_cloth_fgp_20261008_E003/body_only.npz` | 1766 | 2876717 |
| 单腿站立 E004 | `/media/yu/FAFF-E977/YuanQi_Q1/data/experiments/q1_cloth_fgp_20261008_E004/body_only.npz` | 1767 | 2878460 |

两份 `body_only.npz` 均有 `pose_matrix [T,24,9]`、`axis_angle [T,72]`、`trans [T,3]`、`joints [T,24,3]`、`parents [24]`、`source_time_s`、`frame_index`、来源 SHA 和测试用途字段。这是 24 关节 SMPL 身体参考，不是 Q1 的关节控制动作。

相同目录的诊断文件：

- `/media/yu/FAFF-E977/YuanQi_Q1/data/experiments/q1_cloth_fgp_20261008_E003/fgp_trace.npz`：3266448 字节、1796 帧。
- `/media/yu/FAFF-E977/YuanQi_Q1/data/experiments/q1_cloth_fgp_20261008_E004/fgp_trace.npz`：3266513 字节、1797 帧。

`fgp_trace.npz` 保存规范化 IMU、局部姿态、根位移、修正前后根速度、关节线速度和角速度、`foot_contact_probability [T,2]`、`velocity_valid`、`frame_index`。中间文件从源帧 29 开始，身体输出从源帧 59 开始；配对时按 `frame_index` 对齐，不能直接按数组下标配对。

身体输出的 `trans` 都不是全零：走路 XYZ 范围为 `[2.68649,0.21216,2.03748]` 米；单腿为 `[0.63139,0.36240,1.13386]` 米。位移由既有学习式速度头与脚接触修正估计，不是独立测得的世界轨迹。导出固定 30 Hz 时间轴，约 58.83/58.87 秒。

输入是 `train_released.zip` 中已标定、对齐的 IMU PT 样本：`G_0113_zj_train/183926_zj_walk`、`G_0109_yy_train/193310_yy_oneLegStand`。不能把本次成功推理等同于已验证现场原始 `sensor_data.txt` 字节流的全套转换。源包 Mac 在 `/Users/yu/Human+/data/train_released.zip`，服务器在 `/media/yu/FAFF-E9771/data/fgp/CLOTHO_train_data/train_released.zip`；10/8 已核验同大小、同 SHA，本轮不重复传输或解压。

FGP 在 `/home/yu/projects/x2-teleop/FGP-main`，速度扩展在 `/home/yu/projects/fgp/scripts/clotho_velocity`。完整推理、权重身份和误差诊断见[衣服转换阶段](README_Q1_CLOTH_FGP_20261008.md)。Mac 现有同步目录检索未发现 E003/E004 的这两份重新推理身体包，只有摘要和预览；后续优先使用服务器原件。

## 已完成：原两段录制的身体提取包

Mac 原 JSONL 本轮 `stat` 确认存在：

- `/Users/yu/Documents/ChatGPT/YAMAHA/packaging/original-cloth/data/hand_follow_20260923_191240/egolocate.jsonl`：199376995 字节。
- `/Users/yu/Documents/ChatGPT/YAMAHA/packaging/original-cloth/data/hand_follow_20260924_150132/egolocate.jsonl`：187213915 字节。

Mac 已提取的身体小包本轮也确认存在：

- `/Users/yu/Documents/ChatGPT/元启/data/sync/q1_mocap_test_20261008/hand_follow_20260923_191240/body_only.npz`：1945546 字节。
- `/Users/yu/Documents/ChatGPT/元启/data/sync/q1_mocap_test_20261008/hand_follow_20260924_150132/body_only.npz`：1913458 字节。

服务器对应小包本轮实际读取：

- `/media/yu/FAFF-E977/YuanQi_Q1/data/test_inputs/q1_mocap_20261008/hand_follow_20260923_191240/body_only.npz`：1969 帧、1945546 字节。
- `/media/yu/FAFF-E977/YuanQi_Q1/data/test_inputs/q1_mocap_20261008/hand_follow_20260924_150132/body_only.npz`：1923 帧、1913458 字节。

这两份是从已有 FGP→SMPL JSONL 提取的身体字段，并非本项目重新从 IMU 推理的 E003/E004。二者 `trans` 实际全零；用户已确认当时没有移动，不能认定这是推理故障，也不能用其验证移动跟随。详见[真实录制测试](README_Q1_MOCAP_TEST_20261008.md)。

## 关键判断：主要缺口

1. **根轨迹的独立校验。** E003/E004 有根位移、速度，缺的是与其同步且可信的世界轨迹或外部约束证据。原 JSONL 有相机位置/朝向、融合与标定等字段，字段存在不等于这些量已有效或已同步；身体小包也没有保留完整相机链。
2. **接触与地面的可靠信息。** trace 有左右脚接触概率，但不是实测接触标签或足底力。地面高度、法向、足底位置和滑动仍需联合验证，不能把概率直接当作已验证的支撑条件。
3. **现场标定、个人形体和时序证据。** 发布包有 `device2bone.pt`，不能说毫无标定；现场传感器映射、标准站姿、重力轴和朝向仍未独立验证。身体 NPZ 没有显式个人 `betas`，模板骨架与实际身高/骨长关系需确认。固定 30 Hz 来自发布 PT 合同，不是当前设备实测时钟；实时使用还要确认帧率、延迟、掉帧及异常检测。`betas` 缺失不意味着所有重定向方法都不能运行。
4. **已有姿态的准确性。** 10/8 包内标签对照：走路/单腿根相对关节位置误差 P95 为 37.75/39.35 cm，根轨迹 RMSE 为 65.79/53.61 cm。诊断显示模板骨长不是该关节误差的主因，根朝向与预测姿态都有明显偏差。这是包内标签对比，既有模型可能见过这些样本，不是独立泛化或动捕真值评测；本轮没有重跑该评测。

因此，主要问题是世界运动、接触和标定缺少验收，以及现有姿态的重建质量不足。数据已有姿态/位移字段，不应描述为“完全没有 SMPL”或“只有上半身”。

## 版本、复核命令与证据

开始时 hp3090 分支 `Q1`，HEAD `612ea12443b6bcd347fa17367ae3ac17a22f6d20`。已有两份 PredActor 暂存修改独立保留。本轮只增加此说明及入口索引。

以下命令只检查文件；Python 使用已安装的 NumPy，不加载神经网络或设备：

```bash
ssh hp3090
cd /media/yu/FAFF-E977/YuanQi_Q1
git status --short --branch
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python - <<'PY'
from pathlib import Path
import numpy as np
for run in ('q1_cloth_fgp_20261008_E003', 'q1_cloth_fgp_20261008_E004'):
    for name in ('body_only.npz', 'fgp_trace.npz'):
        path = Path('data/experiments') / run / name
        with np.load(path, allow_pickle=False) as arrays:
            print(path, path.stat().st_size,
                  {key: arrays[key].shape for key in arrays.files})
            if 'trans' in arrays:
                print('trans_all_zero', bool(np.all(arrays['trans'] == 0)))
PY
```

既有源证据：两段实验各自 `summary.json`、`logs/q1_cloth_fgp_20261008_E003.log`、`logs/q1_cloth_fgp_20261008_E004.log`，以及 `data/manifests/q1_cloth_diagnostics_20261008_D001_summary.json`。本轮六份服务器 NPZ 解码完成，Mac 两份原录制和两份小包大小已核验；没有启动新模型作业。

## 保存、边界与下一步

报告保存在 Mac 与 hp3090，入口只合并本阶段摘要。重要说明按约定提交到 `Q1`；本次提交身份由 Git 历史提供，发布完成以本地 HEAD 与远端 `refs/heads/Q1` 的实际比对为准。数据继续为 `TEST_ONLY/training_allowed=false`、备份 `LOCAL_ONLY`，不因路径复核而改为训练数据。

后续若要补齐移动参考，优先做带明确世界轴、站姿和地面标定的同步移动录制，并比较根轨迹、朝向及脚接触。训练仍可在合格人体参考重定向后进入仿真策略阶段，是否适合训练取决于参考质量及训练流程，不能仅因缺机器人动作标签就判定无法训练。本轮只完成信息核验。
