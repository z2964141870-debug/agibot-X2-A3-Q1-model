# Q1 自主起身和站稳后的遥操交还

交付入口为 `Q1AutonomousRecovery.step`。正常时由遥操输出关节目标。摔倒后模块独占控制，先用已有起身专家翻身、撑起，再由私有的冻结 V2b 策略跟踪固定站姿。私有站立策略有自己的动作历史、足锚和参考，操作员断开也能运行。外部控制权只取 0 或 1。站稳连续 0.50 秒且收到新鲜有效的遥操参考时，在同一拍重置遥操控制器、重建外部足锚和参考，然后交还。仍在晃动或缺少参考就继续自主控制。此次复用原权重，无新增训练。

[运行时入口](../script/q1_h003_recovery_expert/q1_autonomous_recovery.py) · [接入说明](../script/q1_h003_autonomous_recovery/USAGE.md) · [逐试次结果](evidence/q1_h003_autonomous_recovery_20261120/results.json) · [工件清单](../data/manifests/q1_h003_autonomous_recovery_20261120.json)

## 闭环验收

在 yu-3090 的 Q1 MuJoCo 平地仿真中使用 50 Hz 控制和 200 Hz 物理。开发种子为 251201，两个留出种子为 251233、251257。每个阶段均从初态连续运行到结束。

| 留出场景 | 结果 |
|---|---|
| 从倒姿自主起身，随后接通遥操 | 48/48 自主站稳、完成交还，交还后至少 1 秒保持直立。俯卧、仰卧、侧卧各 16/16。首次确认站稳的中位时间 5.78 秒，最慢 12.66 秒。该组刻意延后遥操输入，交还中位时间 6.82 秒。 |
| 全程断开遥操 | 6/6 在 14 秒观测内起身，末秒持续稳定。全过程遥操调用与交还均为零。 |
| 行走时受推 | 20 次中 4 次直接站住，16 次真实倒地后全部自主恢复、交还，之后至少 1 秒保持直立。倒地样本从施扰到交还中位时间 6.67 秒。 |
| 连续两推 | 10/10 通过。一推化解或一次完整接管、恢复与交还都可通过，末秒仍须直立。共发生 18 次交还。 |
| 无推直线、八字 | 各 701 拍，两个种子下均未触发 recovery，位姿与归档发布版逐帧相同。 |

**交还的独立复核**覆盖 82 次交还。每次交还当拍和之前 25 拍的仿真真值均满足骨盆高度 ≥0.35 m、直立余弦 >0.92、双足合计承重 >10 N、其他部位不触地、骨盆水平速度 <0.20 m/s、机体角速度 <0.80 rad/s。实际控制器使用 IMU、关节和足端运动学，不读取这些仿真真值。逐拍核对得到外部回调与控制归属冲突 0 次，发出时估算 PD 越限 0 项。

开发过程中，单靠起身策略持有目标并按躯干回正、关节低速放行，会在骨盆仍有明显横移时交还。加入自主站立环节和双足相对速度检查后，开发集 24 次倒姿中有 23 次在时限内完成交还，1 次持续由 recovery 持有控制权。留出集的通过数仅描述本次仿真。水平足速依赖地面支撑假设，实机及低摩擦地面尚待验证。

两个单推或倒姿样本的交还发生在原 14 秒观察窗口末尾。验收器现允许在既有首次交还时限内完成交还的样本继续记录 1.5 秒。双推的第二推优先安排在已发生的首次交还后 1.2 秒，超时后备只用于仍未交还的情况。复测涉及 39 条轨迹，改动生效前共 20,656 拍位姿与遥测逐位相同。使用同样的模型和阈值。

## 使用与复现

独立 ZIP 内含起身 ONNX、站立所需的 V2b 两张 ONNX、配置、运动学 XML、运行时代码、接口说明与 SHA-256 校验。Linux 和 Mac 均从临时目录独立导入并运行所有权检查，覆盖过期和超前遥操帧、不稳后重新计时、采样中断、二次摔倒及连续 30 秒固定倒姿。固定倒姿检查是输入回归测试。原速视频展示同一留出种子的俯卧、仰卧、侧卧三条完全断开遥操的真实仿真轨迹。此包提供关节目标接口，真机尚未部署。

服务器原件目录为 `/home/yu/projects/SuperMimicHP/runs/h003_gait_engineering_01/autonomous_recovery`。在相同的父级实验结构下，将本次源码放入新的子目录，复制公共入口 `q1_autonomous_recovery.py`，保留上级原有 bench、relay、run_suite、dependencies、formal_03 与 flat/published。运行以下命令可重做仿真、交付包和视频。接入详情见包内 USAGE.md。

```sh
export Q1_V2B_BUNDLE="$(cd ../formal_03/pinned/q1_v2b_gantry_deploy_bundle && pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=../../../.venv/bin/python
"$py" evaluate.py dev
"$py" evaluate.py test
"$py" summarize.py
"$py" bundle.py
MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json "$py" render.py
```

## 遥操全程在线的配对复核与基线备份

固定两个留出种子 251233、251257，同一初态、同一推力、同一遥操输入时序，各运行旧版与独占 recovery。48 次静态倒姿中，旧版 48 次交还，2 次在交还前连续半秒满足站稳真值；新版 47 次交还，47 次全部满足，另一次继续持有控制权。20 次行走受推中两版各有 16 次真实倒地，交还前连续站稳分别为 9/16 与 16/16。两版均交还的样本里，新版交还时间的配对中位增量分别是 0.82 秒和 0.62 秒。[配对逐试次结果](evidence/q1_h003_autonomous_recovery_20261120/matched_results.json)

基线标签为 `q1-recovery-baseline-20261120`。在 yu-3090 和 Mac 各保留一份实验归档，含独立运行包、权重、原速回放、原始轨迹及两版对照脚本。503 个文件归档时逐件校验，两端核对整个归档的 SHA-256。[归档位置与校验值](evidence/q1_h003_autonomous_recovery_20261120/backup.json)

复现配对实验时，按上文设置环境，在新实验目录内分别运行两组、两个种子及两种控制器，再运行计分脚本。

```sh
for arm in legacy exclusive; do
  for seed in 251233 251257; do
    for group in static push; do
      Q1_STABILITY_ARM="$arm" "$py" compare_online_run.py test "$group" "$seed"
    done
  done
done
"$py" compare_online_score.py
```
