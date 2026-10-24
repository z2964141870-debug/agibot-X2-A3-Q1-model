# PHUMA P12训练步行接触一致性对照

记录日期：2026-10-25，北京时间。状态：独立诊断实现，服务器控制与真实执行待。

## 目标与范围

用户要求继续已有PHUMA、AMASS和SMPL-X资源；BONES由师兄处理。本阶段以[P11](README_SUPERMIMIC_Q1_WALKING_RETARGET_20261025.md)相同四个训练walking来源，比较支撑脚目标修正及Mink原生避碰。仍复现SuperMimic世界模型、想象BPTT、critic；这里是离线参考适配诊断，不增加强化学习方法。P10的15个目录继续预留test，不用于参数选择。唯一R02D训练保持原340输入、7000新增轮和516冻结。

## 方案与边界

新脚本`script/supermimic_q1_sources/{walking_contact_targets,probe_walking_contact,test_walking_contact_targets}.py`均在原冻结集合之外。重新连接原选择/源索引/预留目录并要求与P11逐项相同；原片、时轴、根位置/方向和关节位置限位不变。2与8rad/s分别比较baseline、stance、stance_collision，共预定24组；baseline重新解算且须与P11保存误差小于1e-8。8rad/s仅诊断，仍不满足原2rad/s训练准入。

接触是假设：目标足底相对较低脚15mm内视为支撑，闭合孤立一帧缺口，去除短于4帧的内部片段。只适用于固定walking训练样本，不能当接触真值。支撑期固定踝部位置、以校准脚姿态保留航向并展平；两个支撑段之间用SciPy五阶Hermite位置修正和旋转插值连接。各脚锚点独立，根轨迹保持；所有变体使用由对应baseline整片最低足底确定的同一个常数地面原点。没有逐帧投影根高。

Mink沿用原全身任务权重、位置/速度限位与硬根约束；stance改变脚任务目标，stance_collision另加腕部到躯干/腿部原生距离约束（2mm余隙、50mm激活）。它不是全身碰撞认证，最终仍查全部模型自穿透。安装版本Mink1.2.0的距离上界含多余`/dt`；已读现场函数并核对[上游问题752](https://github.com/kevinzakka/mink/issues/752)与[修正658](https://github.com/kevinzakka/mink/pull/658)。仅本诊断实例使用`gain=.85*dt`，使位移QP上界为`.85*(distance-.002)`；不改环境或冻结代码。两个时间步的真实原生不等式控制通过前不能执行真实对照。

保存全部qpos/qvel、原/新任务目标、误差、支撑标签、全帧脚底及穿透。复用既有MuJoCo脚底点Jacobian、三/五点差分及inverse/RNE分解，对比接触、近地滑动、根残余外力和关节超cap。只做FK/IK/逆动力学，0积分/策略推理/optimizer；完整十九门与真实动力学跟随尚待，所有输出无训练资格。

## 资源、版本与命令

Git权威副本`/tmp/yuanqi-supermimic-q1-20261023`，分支Q1；服务器`huanyu:/opt/data/private/YuanQi_Q1`。代码/报告审计发布并独立核远端后逐SHA部署，不编辑原516冻结文件。单CPU/nice10/无CUDA，28.75GiB/OOM14、100GiB余盘、128MiB输出、900秒无进展停止及自身worker回收；无自动重启。

```bash
ROOT=/opt/data/private/YuanQi_Q1
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -u script/supermimic_q1_sources/probe_walking_contact.py \
  --root "$ROOT" \
  --output "$ROOT/data/supermimic_for_q1/PHUMA_P12_walking_contact_probe_20261025" \
  --expected-oom-kill 14
```

日志为同名`logs/`，guard.json在实验根，selection/receipt及NPZ在probe/，inverse数组在probe/dynamics/。完成后独立重算保存数组统计，不把相同helper重复调用当独立证据。原始模型/数组仍服务器LOCAL_ONLY；Mac只取小收据，代码/重要结果按Q1规范发布。

## 结果与下一步

本地三项来源/标签控制通过、三份新代码语法通过；Mac没有Mink/MuJoCo/SciPy，未执行本地真实几何控制。服务器完整五项控制及实际24组对照待。任何求解失败保留该变体FAILED及原因，诊断完成不等于全部变体成功。先查支撑/滑动是否改善且未增大原任务误差、穿透或动力学残差；不因近地比例改善自动合库。R02D原预算继续，目标未见步行动力学跟随尚未达成。
