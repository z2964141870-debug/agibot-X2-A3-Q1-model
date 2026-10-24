# PHUMA P11训练来源步行限速归因

记录日期：2026-10-25，北京时间。状态：独立诊断入口已实现，本地/服务器控制及真实结果待。

## 目标与依据

继续现有PHUMA资源与SuperMimic世界模型路线。P10B固定15个新步行候选全部支撑失败；D02C/D04已说明源足底/目标映射和浮脚问题，但没有区分明确walking样本上的限速作用。本阶段只用此前选择的train来源，排除全部已知dev/test目录候选和P10的15个预留目录，不用补充test选择修补参数。

现有重定向配置`reference_velocity_limit_rad_s=2`是本工程离线配置，提供的Q1 URDF各关节velocity字段为32或36rad/s。这些字段不是实测可安全达到的硬件速度，不据此提高真机限制。只在同一仿真模型中诊断2/4/8rad/s三种IK速度限制；原root、全身任务目标、原速30→50Hz、关节位置限位和初始静态拟合保持。

## 实现、边界与预算

新增`script/supermimic_q1_sources/probe_walking_retarget.py`。固定seed2026102511，先以HumanML原事件名选择四个train walking目录，再读取数据/IK，不根据Q1几何或策略结果选片。所有已知旧held-out与P10预留目录参与排除；其他命名空间的隐藏重复仍未全解，不称全源身份认证。

复用原Mink/quadprog、PHUMA解码、源FK/映射和静态初姿态拟合；只更改独立solver实例中的速度限制，不改任何冻结文件。对比源固定落地、任务脚底曲线和解后脚底，保存每组qpos、足底曲线、近限速比例、残差及自穿透。每条曲线仅以一个常数对齐供分析，不写回参考；不做旧D05B逐帧根高投影，不调用导出或准入，不放宽原十九门，不混入当前训练。

单CPU worker/nice10/无CUDA，28.75GiB共享内存/OOM基线14、100GiB盘余、64MiB输出；900秒无进展停止，回收自己的worker组。0积分/策略/optimizer、新训练参考始终0。

## 版本、命令与产物

权威工作副本`/tmp/yuanqi-supermimic-q1-20261023`，分支Q1。发布并独立核远端后，逐SHA部署新producer/test，跑来源排除控制与真实导入，再执行：

```bash
ROOT=/opt/data/private/YuanQi_Q1
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -u script/supermimic_q1_sources/probe_walking_retarget.py \
  --root "$ROOT" \
  --output "$ROOT/data/supermimic_for_q1/PHUMA_P11_walking_velocity_probe_20261025" \
  --expected-oom-kill 14
```

输出为`probe/{selection,receipt}.json`、四份NPZ、`guard.json`；stdout在同名logs/。结果和小收据保存后再选择接触/映射方向。当前没有已运行结果或新的合格参考；R02D是唯一训练，原7000新轮及340/6/7输入保持，真实移动目标未达。

## 下一步

先实际完成有限对照并重算保存曲线/差分。若提高限速没有解决目标脚底，不能以继续提高上限代替接触约束；若有收益，仍须独立新协议和实际动力学验证，不能由URDF元数据或几何改善直接宣称有效策略。大型原件LOCAL_ONLY，阶段小记录按Q1规范发布。
