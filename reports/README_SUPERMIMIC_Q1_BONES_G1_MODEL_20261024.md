# BONES源G1候选模型与官方运动学核对

记录日期：2026-10-24，北京时间。阶段：SM-Q1-BONES-P02-M01。状态：现有候选与宇树官方G1运动学/足碰撞结构核对完成；BONES原生成模型及实际CSV仍未验收。

## 目标与范围

为[BONES主线](README_SUPERMIMIC_Q1_BONES_GOAL_20261024.md)的G1到Q1转换确认可用源FK模型。现有GEAR SONIC的G1模型已能载入，不能仅凭29关节名称相同认定几何一致；本次与宇树官方固定模型做实际编译数组和数值FK对照。未重定向真实BONES动作、未训练。

## 来源、许可与实际结果

官方来源unitreerobotics/unitree_ros固定提交adb48c7910ef8404ca34a07df95fc3acbb09f318，相关文件最后提交日期2024-10-22；文件robots/g1_description/g1_29dof_rev_1_0.xml及根LICENSE。许可BSD-3-Clause，原XML25820字节/SHA165fa7a5275745e45fa25e1bc78010c1055d81bb81c013b3c7bc8b9347a3b9c3。来源/许可原文留服务器被忽略的data/references/bones_seed/model_candidates/unitree_ros_adb48c7/，Git仅保存原创比较工具与收据，不上传原模型/网格。

21:38:15原生执行退出0：候选与官方均nq36/nv35/nu29、31body/30joint；身体/关节顺序、父子关系、qpos/dof地址、关节类型、零位、位置/旋转、轴向和限位逐数组最大差均0；左右足各4个碰撞球参数、局部位置/旋转及碰撞mask完全一致。固定seed20261024的64组随机合法关节/基座姿态，所有body的FK位置差和符号无关四元数差均0。

官方XML编译只把meshdir指向既有本地网格，以避免重复下载；未独立下载/比对官方网格字节。该方法能支持所比较的骨架运动学及球形足碰撞一致，不能证明视觉网格、全部物理参数、BONES生成用FBX或生成模型身份相同。BONES官方查看器使用FBX；仍须用实际源数据检查零位、坐标、足底和关节范围。

## 版本、命令与产物

代码script/supermimic_q1_sources/compare_g1_kinematics.py，SHAf307fbaf08ceca8f1edbc327c3651722868c4520fb5d7b4db33ba2d8286128a9，两端一致。阶段前发布fc4177704ed70160bc642eae264e9cc940c4b1e1。候选路径、XML完整SHA、所有数组/足参数及数值FK在[模型收据](../data/manifests/supermimic_q1_bones_g1_model_20261024.json)。服务器原始产物data/supermimic_for_q1/BONES_P02_model_20261024/official_comparison.json，日志logs/BONES_P02_model_20261024.log。

```bash
/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python script/supermimic_q1_sources/compare_g1_kinematics.py \
  --candidate /opt/data/private/yuanqi_ops/data/g1_leg/run/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml \
  --official data/references/bones_seed/model_candidates/unitree_ros_adb48c7/g1_29dof_rev_1_0.xml \
  --mesh-dir /opt/data/private/yuanqi_ops/data/g1_leg/run/gear_sonic/data/assets/robot_description/meshes/g1 \
  --output data/supermimic_for_q1/BONES_P02_model_20261024/official_comparison.json
```

上述输出已经存在，重核必须使用独立新输出路径。没有物理积分或优化器更新；随机FK是模型一致性证据，不是训练数据或跟随效果。

## 下载状态与下一步

21:35至21:36左右再次实际读取Windows D:\tmp，最大partial为6894708545字节，比21:16的2720542770字节增长4174165775字节；说明文件写入已推进，仍未见完整.tar.gz。没有据此猜下载百分比、包种类或完成时间，未读取partial内容。

接下来以该已核运动学候选准备BONES专用原速重定向，输出在真实来源/几何/独立质检完成前不得用于训练。保留P01来源隔离及已有准入门，不伪装成PHUMA来源、不通过改动作速度补量。小记录和工具同步Q1与两端，原模型/网格留服务器LOCAL_ONLY。
