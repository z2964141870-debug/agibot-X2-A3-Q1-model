# A3 接触约束根轨迹与足目标 E16

日期：2026-10-12。实验：A3_ZHY_SMPL_CONTACT_20261012_E16。状态：VERIFIED_QUALITY_REJECTION；接触期贴地通过，但摆动脚穿地/部分root修正速度失败，未训练。

## 目标与边界

E15修正形态映射后，七段位置P95已降至2.2–4.5cm，但六段穿地、五段支撑滑动，squat最低脚点悬空4cm。新阶段检验人体根位移与按机器人骨长生成的足目标如何满足支撑接触；不是换数据、放宽门槛或训练学生来吸收错误。

人体contact来自原始toe高度和速度（沿用5cm高度/0.25m/s阈值），没有实测接触传感器。每段接触开始固定支撑踝XY，在双脚支撑时由两只脚共同确定root平移修正；所有root修正逐帧保存。支撑足保持heading、roll/pitch按世界竖直，利用既有机器人真实脚mesh/box顶点把目标脚底放在1mm。非支撑足仍使用E15形态目标。不把此工程约束描述为人体姿态真值或真实动力学平衡。

IK仍Mink/quadprog、URDF限位和速度保护。支撑脚position/orientation cost固定400/40，其他20/1；源帧期间8×1/240s仍总计1/30s，不提高关节速度预算。首帧先用120×1/120s受限IK定位（等效1秒，非策略回放）处理任意录制片段的初始姿态，随后300真实源帧按正常速度检查。

保留E15全部质量门槛，新增接触期实际脚几何高度P95≤2cm、root修正最大≤0.4m/修正速度P99≤1m/s、参考root速度P99≤4m/s。上限是本阶段预登记工程预算，不是部署安全证明。某段通过只允许限定离线监督，不得把七段诊断数据作为未见最终测试。

## 代码、记录与执行

父阶段E15已审计推送核验：`eed802c96e35f30a38047a67f34c47960c84038c`。vendor固定 `fe6868ba37034f89b912f0fb851bce19f120266d`；原始与E14/E15失败工件不改。

源码 `script/a3/smpl_contact_retarget.py`、`script/a3/smpl_contact_probe.py`；实际代码/输入SHA在registration.json锁定。只在ZHY CPU单线程计算，0模型前向/optimizer/策略回放/无CUDA，原机仅同步小代码/记录。90/85°C、4GiB保留、5秒采样、900秒上限、首故障停止、无自动恢复，已有实验不重复或重置。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.test_smpl_contact_retarget
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_contact_probe register
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_contact_probe run
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_contact_probe verify
```

工件 `data/experiments/a3_zhy_smpl_contact_20261012_E16/`，日志 `logs/a3_zhy_smpl_contact_20261012_E16/`，清单 `data/manifests/a3_zhy_smpl_contact_20261012_E16.json`。标签/CSV/数值trace留服务器LOCAL_ONLY；Mac仅小代码/README/清单，无新增原始数据或模型。

## 验收与下一步

3CPU目标契约通过（2.692秒）。新机21:10:58–21:11:19实际完成2100源帧，21:11:28独立复核contact mask/root修正/目标、全部qpos FK/真实脚几何、CSV与质量。7/7位置P95通过（0.02783–0.04980m），6/7位置最大值通过，7/7限位速度/接触覆盖/接触期脚底贴地通过；典型行走接触期脚底高度P95约0.9–1.2mm，warmUp/squat最高约1.68cm。

但7/7整体脚几何仍穿地，最低-3.36到-8.12cm；支撑脚约束未处理非支撑脚，不能宣称“贴地通过所以不穿地”。5/7原滑动指标通过，4/7root修正速度失败（上限1m/s，实测0.421–1.567m/s）；root修正最大0.092–0.240m均在登记0.4m预算内，参考root速度预算通过。合格pair0/三个可训练分区为空，真实optimizer0/策略回放0/无CUDA。

原滑动指标用帧中心差分并以本帧contact过滤，contact边界仍含前后非接触帧位移。其数值如实保留；后续应按前后两帧均接触的实际时间区间统计滑动，并新增合成边界测试，阈值仍0.25m/s。这项指标修正不能抹去真实摆动脚穿地；下一独立对照同时约束实际foot/floor几何，保存所有修正，不人工改宽门槛放行。

7健康样本21:10:55–21:11:28，同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`，CPU最高40°C/GPU27°C、cgroup可用最低8,870,793,216字节、OOM0，无关键内核/遥测错误。最大间隔9.34秒含worker结束到独立核验；计算已退出。0模型前向/训练，原始和E14/E15工件SHA不变。回执：[E16清单](../data/manifests/a3_zhy_smpl_contact_20261012_E16.json)。

恢复读本文件与[E15结果](README_A3_SMPL_CHAIN_20261012.md)；重要结果立即更新README/索引/current state、审计scoped提交推送并核验。来源已由用户确认，不重新请求授权。
