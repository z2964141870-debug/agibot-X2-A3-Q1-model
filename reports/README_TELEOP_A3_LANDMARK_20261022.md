# A3肩部任务点几何诊断

记录日期：2026-10-22（北京时间）。实验：T12。状态：已保存T10轨迹的目标重建、独立FK和解析几何核验完成；未重解IK、未做动力学或训练。

## 目标与完成内容

解释标签姿态的A3重定向自身IK P95为何超过5cm，避免把几何目标冲突交给策略训练补偿。输入固定为T10原30帧`[30,60)`的walk标签/ONNX两臂，直接复用保存的qpos、目标和T09人体源。T10及历史终态均未修改。

新增`script/teleop_research/a3_landmark_geometry_audit.py`。重新核验T10机器人资产、T09配对源身份，按原继承链重建目标；两臂目标最大差均0，使用独立MuJoCo `mj_kinematics`从保存qpos计算的身体位置与原轨迹最大差均0。仅运动学计算，无`mj_step`。

## 实测发现

标签臂自身位置P95仍为6.3993cm。分部位P95：左/右肩7.5158/8.7584cm，左/右膝3.9420/3.8728cm，左/右腕1.4639/2.1462cm。肩是此窗口越过5cm全局门的主要部位，膝有残差但本轮尚未证明其唯一原因。

源码将人体SMPL肩点映射到`left/right_shoulder_yaw_Link`原点，而Chain目标把该点固定在躯干上。实际机器人该点在肩roll枢轴下游，局部偏移为`[0, ±0.01, -0.1475]`m；它随肩pitch/roll运动，并非刚性躯干肩点。模型中的`shoulder_roll_Link`原点相对躯干固定，已在全部保存qpos上核验，误差小于6e-16m。

该错位还导致目标肘点的几何冲突：给定目标躯干姿态，实际roll枢轴至肘的可达距离仅为左`[0.2800,0.280713]`m、右`[0.2805,0.281212]`m；标签目标却要求左`[0.20884,0.27210]`m、右`[0.21783,0.27408]`m。对应距离越界P95为6.974/6.186cm。解析关系已在真实保存FK上验证，数值误差小于7e-16m。

若再要求原目标肩朝向精确匹配，肩位置的必要残差P95达到左18.496/右17.511cm。此项明确是**给定躯干及肩朝向的条件下界**，不能说所有位置目标在任意姿态下都绝对不可达。旧IK允许朝向误差并移动躯干，因此最终以约7–9cm肩误差折中；本轮证明的是目标任务点/朝向之间的几何不一致，而非策略无能力。

## 决定与下一实验

值得先修正肩部任务点定义。独立候选只改变肩点合同：位置使用相对躯干固定的roll枢轴，完整肩朝向仍由原yaw身体承载，分别建立位置/朝向任务。其他腿部、接触、实际足几何、限位、求解预算和原5cm质量门保持。

候选必须单独登记并通过实际IK与保存轨迹的独立FK复算；重新报正确解剖点、未变的腕/踝、地板、接触、速度和限位。任务点定义改变后的误差不能伪装成同一旧点的数值改善；须并列保留旧点指标。尚未验证候选有效，不宣称完整重定向修好或进入035动力学。

## 版本、命令与证据

执行主机huanyu，使用现有`/opt/data/private/hechengzhu/ZHY/data/env/bin/python`，CPU单线程且CUDA屏蔽，无环境变更。T11已同步提交`f91e54a7f2ba6991627499b9ed73c13c4f23d7fe`，远端main独立查询同号。

```bash
cd /opt/data/private/yuanqi_ops
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /opt/data/private/hechengzhu/ZHY/data/env/bin/python \
  script/teleop_research/a3_landmark_geometry_audit.py \
  --root /opt/data/private/hechengzhu/ZHY \
  --t10 data/experiments/teleop_20261022_T10_paired_a3_kinematic \
  --data data/fgp/train_released.zip \
  --output data/experiments/teleop_20261022_T12_geometry_audit.json
```

该输出首次创建，不覆盖重跑。实际退出0，日志`logs/teleop_20261022_T12_geometry_audit.log`。小型证据在`data/manifests/teleop_A3_T12_geometry_20261022.json`；包含驱动、继承源码、机器人资产和两臂保存轨迹的大小/SHA。原NPZ保留服务器，备份仍为`LOCAL_ONLY`。

## 同步与恢复

本报告、索引、当前状态、审计驱动和小清单保存到指定仓库main；同步以实际提交/远端HEAD比对为准。恢复先读T11和本报告，下一是独立肩任务点候选及配对几何资格检查，继续前核对保存输入/源码身份。035仍是固定基线，没有新增策略效果。
