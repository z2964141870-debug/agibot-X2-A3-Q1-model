# PHUMA P14整段常数根高度适配

记录日期：2026-10-25，北京时间。阶段P14。状态：实现、本地三项解析控制及服务器四控制完成；真实八组与独立复核待。

## 目标与范围

用户将BONES获取交给师兄，等待通知，当前继续现有PHUMA/AMASS/SMPL-X和SuperMimic世界模型路线。[P13](README_SUPERMIMIC_Q1_WALKING_REACH_20261025.md)证明固定根与P12C修正脚目标存在保守腿长外球冲突；本阶段只用同四train及2/8rad/s的八个stance_collision组，检查整段常数根高度适配能否改善。原P11/P12/P12B/P12C/P13生产者、helper、控制和输出不改。

不使用P10预留test调参，不修改R02D的340/6/7、516冻结输入、固定7000新轮，不启动第二训练。新IK仅为诊断，不自动合库或准入，不放宽原十九门，8rad/s无原速度门资格。

## 实现与决定

新`probe_walking_root_offset.py`绑定P12C/P13完成收据、guard、独立审计、原数组与源码SHA。对每组P13必要区间求整段交集，固定`delta=min(0,min(upper)-0.005)`，须`delta>=max(lower)+0.005`且`abs(delta)<=0.12m`；不满足就拒绝，不自动改阈值或做逐帧投影。5mm是保守几何候选的预留量，不是动力学或真机安全保证。

同移根Z与所有非足位置目标，保留根XY/方向/竖直曲线形状、原速、非足相对根位置和脚部世界锚点；脚姿态、推断stance标签保持。复用已控P12C原生足地0.5mm约束及腕部避碰，独立求解，保存原/新根与目标及实际delta。遵循原保存坐标：原qpos先减公共origin shift供IK，floor取负shift，求解后加回shift；分析资产仍floor=0。

新`audit_walking_root_offset.py`不调用生产者的区间/变换/统计函数，从真实根变换和显式链重算区间及delta，再核脚锚点、非足位移、根/时间/姿态、限位、速度，fresh FK、足底/滑动和两种差分inverse。原contact auditor要求root不变，因此不复用或修改它来掩盖本阶段有意改变的根高度。

## 版本、命令与产物

权威Git`/tmp/yuanqi-supermimic-q1-20261023`，分支Q1；服务器`huanyu:/opt/data/private/YuanQi_Q1`。本地三项解析控制原生退出0、三文件语法检查退出0；本地没有执行MuJoCo控制。服务器完整四控制通过后才运行：

```bash
ROOT=/opt/data/private/YuanQi_Q1
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -u script/supermimic_q1_sources/probe_walking_root_offset.py \
  --root "$ROOT" \
  --output "$ROOT/data/supermimic_for_q1/PHUMA_P14_walking_root_offset_20261025" \
  --expected-oom-kill 14
```

单CPU/nice10/无CUDA，28.75GiB共享内存/OOM14、100GiB余盘、128MiB实验、900秒无进展保护，guard只回收自身worker。建目录前仅检查内存、创建后由guard完整检查，避免P13首次bootstrap错误。输出`probe/{receipt.json,*.npz,dynamics/*.npz}`与`guard.json`，stdout在同名logs；独立审计输出实验根`independent_audit.json`。

## 已核状态与边界

07:37:54现场原boot/ticks/cmdline/组与三进程非Z匹配，R02D三状态RUNNING、新增4550/7000轮/291200转移、累计9550轮，WM1769/actor1719次，最新数值有限。GPU39摄氏度/1228MiB，共享内存13.81GB/OOM14，盘余250436395008字节。共享盘继续下降，本训练当时约2.054GB，不能把全盘增长归因本项目；100GiB停止线保持，若触发则保留真实终态、不自动恢复、不删未知文件。

P14尚无真实结果；几何必要区间通过不等于实际可达，支撑/碰撞四项子集通过也不等于十九门准入、动力学跟随或论文结果。当前目标尚未达成。

## 保存与下一步

代码、阶段README、索引和当前状态先审计、提交、推送Q1并独立核远端，再逐SHA部署。大型原件均服务器LOCAL_ONLY，Mac仅小证据。下一完成服务器四控制、真实八组及独立复核，按结果决定是否值得进一步完整准入；R02D仍按原预算及既定最终七阶段配对推进。

## 服务器控制与部署

准备发布`0f5a8d5a123f4953718361d9d6f8f73454e5ac2a`已显式推Q1、独立核远端完整号一致；三新文件部署前不存在，部署后逐SHA与发布一致。[控制记录](../logs/session_records/supermimic_q1_P14_root_offset_controls_20261025.log)保存原生四控制输出、命令和SHA。服务器原生退出0，四项全过/0跳过，其中真实Q1的32组限位内姿态确认降低根80mm使双hip同移而关节位置保持。它不证明目标可达或真实动作通过。

07:46:06 R02D4775新增轮/305600转移、三状态仍RUNNING，最新数值有限/OOM14，共享内存13803823104字节，盘余207749152768字节；P14目录当时不存在。原保护保持，真实P14待执行。
