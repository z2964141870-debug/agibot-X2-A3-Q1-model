# A3 足地板约束与连续接触区间 E17

日期：2026-10-12。实验：A3_ZHY_SMPL_FLOOR_20261012_E17。状态：VERIFIED_QUALITY_REJECTION，定位每帧直接设置root绕过碰撞约束，未训练。

## 目标与配置

E16支撑期实际脚底高度P95通过，但7/7有摆动足穿地、4/7root修正速度失败。下一独立对照保留原始七段及所有质量预算，加Mink已有CollisionAvoidanceLimit约束实际碰撞脚几何与官方MJCF floor；最小距离1mm、激活距离15cm、gain0.85、无bound relaxation。非支撑足目标按现有真实足顶点投影给至少3mm地板间距，源代码/vendor/物理模型不改。模型仅CPU运动学，不运行动力学或机器人。

root修正的逐帧变化限为0.5m/s（比质量1m/s上限更严格），保持支撑脚world锚点，其他身体随受限root移动；仍量化全部root修正和实际误差，不能靠限幅boolean认定姿态目标实现。

滑动使用真实前向时间区间，必须两端原始人体contact均成立。修正的是之前帧中心差分混入非接触端点，不改变0.25m/s阈值；接触覆盖要求至少5个有效区间。合成测试保留真实支撑滑动并排除非接触gap运动。E14/E15/E16历史数值与记录不改。

## 运行与保存

源码 `script/a3/smpl_floor_retarget.py`、`script/a3/smpl_floor_probe.py`；注册锁定父E16代码/工件与本轮新增SHA。vendor仍 `fe6868ba37034f89b912f0fb851bce19f120266d`，ZHY CPU单线程，0模型前向/optimizer/策略回放/无CUDA。复用90/85°C、4GiB保留、5秒采样、900秒上限、首故障停止，不自动重试或恢复。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.test_smpl_floor_retarget
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_floor_probe register
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_floor_probe run
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_floor_probe verify
```

工件 `data/experiments/a3_zhy_smpl_floor_20261012_E17/`，日志 `logs/a3_zhy_smpl_floor_20261012_E17/`，清单 `data/manifests/a3_zhy_smpl_floor_20261012_E17.json`。原始/数值数据仅服务器LOCAL_ONLY，Mac无大型工件。失败也保存，已有实验不覆盖。

## 验收

4CPU契约通过（1.860秒）。新机21:26:25–21:26:45完成2100帧，21:26:55全部独立目标/root/contact重构、qpos FK/真实foot geometry、CSV复核通过。7/7位置P95/接触滑动/贴地/root修正速度通过，6/7位置最大通过；root修正P99≤0.5m/s，连续接触区间滑动P95约0.0005–0.0194m/s。squat最大位置误差0.20746m未过0.15m门槛。

但7/7脚几何仍穿地，最低-1.74至-6.90cm，故合格pair0/真实optimizer0。添加6个实际碰撞足/floor约束不是无穿地证据，必须核对保存的真实几何。父失败记录保持。

只读定位保存fjw walk最差第42帧（两脚source contact均false）：右足实际碰撞geom72最低Z/`mj_geomDistance`均-0.0307409m，floor位姿确实世界Z0；geom72已在Mink约束pair列表。不是视觉mesh/碰撞geom遗漏或地板轴向误判。E16/E17继承的solve每帧先将configuration root qpos直接改为目标位置，再调用IK，可能在碰撞约束生效前就把脚放进地板。碰撞约束限制后续逼近，不能保证修复每帧外部直接改qpos造成的穿透。下一独立变量让floating base和关节一起求解，以root任务代替每帧root直接设置与硬约束；实际root偏差仍计入原IK/速度门槛，不放宽足几何检查。

7健康样本21:26:22–21:26:55，同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`，CPU最高48°C/GPU27°C、cgroup可用最低8,864,530,432字节，OOM0，无关键内核/遥测错误。最大间隔9.44秒含worker结束到核验。0模型前向/策略回放/训练，计算结束；回执：[E17清单](../data/manifests/a3_zhy_smpl_floor_20261012_E17.json)。

恢复读本文件及[E16失败结果](README_A3_SMPL_CONTACT_20261012.md)。最终泛化仍需未参与这些诊断的新会话，不能把损失下降或接口通过当策略提升。
