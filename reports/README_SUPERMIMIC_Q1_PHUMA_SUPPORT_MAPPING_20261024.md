# PHUMA足底高度映射诊断

记录日期：2026-10-24，北京时间。阶段：SM-Q1-PHUMA-D02。状态：离线诊断入口已实现，Mac四项刚体几何控制通过；服务器控制与真实八片分析待执行。PHUMA R01原训练继续，源码/输入/3000轮预算不改。

## 目标与范围

[D01](README_SUPERMIMIC_Q1_PHUMA_DIAGNOSIS_20261024.md)确认112候选中85片支撑比例失败、44片自碰撞失败（重叠），但尚未区分PHUMA源轨迹本身、机器人目标映射和IK残差各自对支撑不足的影响。本阶段只读八个固定训练集样本，比较同一50Hz时间轴上的源G1碰撞足底、Q1目标脚刚体姿态和原始Q1 IK参考足底。development/test不用于本次诊断选型。

新入口位于 `script/supermimic_q1_analysis/analyze_phuma_support.py`，不在当前R01冻结集合内。直接复用固定PHUMA解码/重采样和独立足底查询函数；不执行IK、积分、策略推理、训练或修改任何参考。源ZIP不重复全包顺序读取，仅按已固定成员读取并逐份核验原始大小/SHA；固定源46文件和R01的121冻结输入重新比对。

## 方法与控制

三种足底曲线各保留原始世界高度。另用每条整片曲线一个固定平移比较时间变化；源曲线乘既有腿长比例表达为Q1米。该诊断平移不写回轨迹，不是逐帧根高投影，也不证明动力学可行。Q1目标脚是任务给定的刚体姿态，不保证双脚/全部关节能同时到达。

新刚体几何查询支持sphere/box/mesh，四项控制覆盖带局部平移的旋转球、旋转盒垂直投影、网格顶点/多几何最低点和未知/空几何拒绝。Mac已原生退出0、4项通过。真实分析将逐帧把刚体查询与MuJoCo实际几何比较，同时复现原PHUMA源FK和既有支撑统计；这些运行尚未发生，不预写通过。CPU进程禁用CUDA，至少1GiB初始余量，距原30GiB训练保护线不足256MiB时停止诊断，OOM增量亦拒绝。

## 固定样本与产物

固定训练样本为两片已合格移动、两片仅支撑比例失败的移动、两片仅自碰撞失败的移动、一片仅支撑失败的蹲和一片仅支撑失败的上肢。这是针对性诊断样本，不代表整个PHUMA或独立测试集统计：

- `train_locomotion_1bc87468bec5a472`。
- `train_locomotion_26832240824d6963`。
- `train_locomotion_fe9abef3042c3c39`。
- `train_locomotion_1f75cd47fbf61bb7`。
- `train_locomotion_6cafc84ec90e5186`。
- `train_locomotion_007047b45054fd5a`。
- `train_squat_04ebe3a321c47d2f`。
- `train_upper_a5678394f936531f`。

服务器预定输出 `data/supermimic_for_q1/PHUMA_D02_support_mapping_20261024/receipt.json` 和逐片诊断NPZ，原件留服务器LOCAL_ONLY；Mac只取回小型JSON/日志。真实执行stdout/stderr放 `logs/PHUMA_D02_support_mapping_20261024.log`；控制输出单独保存。

## 版本与复现

当前训练版本仍 `ed4217384e7abfc8a9cea0b23b95592d4a3e19b0`。本诊断代码提交由本次Q1 Git历史提供，发布并独立核验后才部署运行；运行收据另记实际诊断发布号。Python沿用服务器已有环境，不安装依赖。复现入口：

```bash
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" \
  script/supermimic_q1_analysis/analyze_phuma_support.py \
  --root /opt/data/private/YuanQi_Q1 \
  --motions /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_P03_references_20261024 \
  --training-spec /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_R01_B128_3000_20261024/spec.json \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_D02_support_mapping_20261024 \
  --code-commit "$DIAGNOSTIC_COMMIT" \
  --clips train_locomotion_1bc87468bec5a472 train_locomotion_26832240824d6963 train_locomotion_fe9abef3042c3c39 train_locomotion_1f75cd47fbf61bb7 train_locomotion_6cafc84ec90e5186 train_locomotion_007047b45054fd5a train_squat_04ebe3a321c47d2f train_upper_a5678394f936531f
```

输出排他创建，不能重跑覆盖同一目录。先服务器四项控制、资源门和当前进程身份复核，再执行一次真实八片分析；保存结果/失败后更新本README和两个入口，提交推送并独立核验Q1。随后继续等待原训练与既定匹配评测，不把诊断准备记为跟随提升。
