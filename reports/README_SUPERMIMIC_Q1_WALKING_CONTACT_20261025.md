# PHUMA P12训练步行接触一致性对照

记录日期：2026-10-25，北京时间。状态：本地三项/服务器五项控制通过，首轮24组IK和逆动力学完成；高度单判支撑的锁脚方案拒绝，独立重算待。

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

## 06:21真实依赖控制通过

准备代码发布`3259124bf9f5b3232565172021834fdfd511c10b`已推送并独立核Q1远端。三份新文件部署前均不存在，且不属于516冻结；部署后SHA分别为helper`448b5042c02d12851ae6d2221bc09da7d9ac77620a280cee3bc7fa1a941af142`、producer`8e4f57014fbae5b593680dfc6cd176efac781699c5680c75e25935929aecd5b3`、test`ec0eb30580ca5ce908d91e5b78a6a62c71189d29c5dbfc809074a6d950b4a3cd`，与发布一致。

服务器`python -m unittest test_walking_contact_targets -v`原生退出0，五项全过/0跳过：来源角色/目录排除、相对高度不变性、非法输入拒绝、支撑锚点/摆动连接/原数组不变、原生Mink两个dt的位移不等式实际数值。日志`logs/PHUMA_P12_contact_controls_20261025.log`，SHA`ca24fa860eba785f96def040c45f2c98739f6554e640e309ed719d8f24d6476b`。这不是四片真实对照或准入结果。

06:19训练实读2475新增轮/158400转移，全部516冻结重新核SHA保持，OOM14/内存约14.15GB、盘余约655.86GB。共享盘继续下降，当前独立诊断维持128MiB上限及100GiB余量，不删除未知文件。下一实际新目录P12，不改R02D原预算/输入。

## 06:22首轮完成但方案拒绝

06:21:32至06:22:00原生退出0，24组均完成（四来源、两速度、三变体）；guard842934/worker842935已回收，0CUDA/积分/策略/optimizer，新参考0。八个baseline逐qpos复现P11，原输入前后SHA保持。实验分配17,805,312字节，结束共享内存13,785,378,816/OOM14、盘余647,097,020,416字节。原样[receipt](../data/manifests/supermimic_q1_P12_contact_receipt_20261025.json)304495字节/SHA`ad5cc6e367f507899d96d1e05f1e713747469e6967a24ee422aaa30b77d90fa8`和[guard](../data/manifests/supermimic_q1_P12_contact_guard_20261025.json)保留。

仅按两脚相对高度推断支撑过宽：每脚81.41%至100%被标记站立，fast02甚至双脚整片同时站立。固定世界锚点与向前根轨迹冲突，脚任务最大改动0.38至1.04m，明显失真；8rad/s下三片原任务P95由17.9至21.1mm升为165至357mm。所有修正变体的支撑/穿地/自碰撞子集均未通过，不能导出为训练参考。

8rad/s较低脚底15mm内比例：Circle06由64.32%降31.66%、fast02由43.02%降11.73%、Circle02由61.97%降36.62%、slow02由18.59%降2.51%。slow02在2rad/s下约11mm穿地。此失败属于本次接触假设/目标构造，不是正在运行的R02D失败。

原生避碰启用后，全部八组stance_collision的模型自穿透最大值为0；无避碰stance六组有6.68至18.20mm穿透。这个小样本结果支持保留避碰方向，但不能证明全部动作可避碰或动力学可行，且它未修复落地。根残余外力中位数仍约一倍体重。下一先独立FK/差分/数组重算，再在同四个train中加入固定脚水平速度条件，避免把低高度摆动当站立；保留首轮原件/生产者，不以P10测试调参。

新增独立`audit_walking_contact.py`不调用原IK/producer统计；重读原数组并使用独立脚底查询、新MuJoCo状态逐帧FK和inverse，核全部速度/三五点加速度、任务误差、足底、穿透及关键结果。原始生产者/helper保持不变；审计发布部署后实际结果待。
