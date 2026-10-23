# Q1独立根高支撑参考候选

记录日期：2026-10-24，北京时间。阶段：SM-Q1-D05。状态：本地4项及服务器7项控制通过；首批真实候选在出拳自碰撞平移不变性检查处失败退出1，两段部分产物保留，未接入训练。

## 目标与范围

针对[D04](README_SUPERMIMIC_Q1_REFERENCE_DYNAMICS_20261024.md)确认的参考足底悬空，准备单一变量的独立候选。R03仍按原固定预算训练，不改变其输入、冻结的两个训练源码目录、控制设置、预算或保护，不启动额外GPU作业。

## 完成内容与关键决定

新增`script/supermimic_q1_analysis/build_support_projection.py`、`audit_support_projection.py`及控制测试。候选只调整每帧根部Z及相应竖直速度；保持原50Hz时间轴、水平路径、根姿态、全部关节位置/速度、来源和三片覆盖。重新计算身体FK/速度，不直接修改原参考。

使用SciPy稀疏有界最小二乘，将较低足底限制在地面下0.05至0.5mm，目标0.25mm；最小化根高修正相对目标的误差及二阶差分，二阶差分权重10、求解单位1mm，绝对根高修正上限3cm。这是固定的几何候选，不是优化接触力或已验证支撑平衡；微小穿透只为避免精确零高度受数值误差变成无接触，不证明所产生支撑力正确。

每片独立记录绝对原目标的身体位置误差、全片自碰撞和足底接触，复用D04的三点/五点差分对新候选做逆动力学检查。原D04不重跑，候选不修改关节，因此预期自碰撞不会消失。出拳仍为复用开发数据，不提升为独立测试。所有候选明确`qualified_for_training_pilot=false`，须先完成独立导出审计及后续匹配动力学评测再决定是否接入。

## 版本、命令与证据

工作副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`；服务器`huanyu:/opt/data/private/YuanQi_Q1`，Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。服务器已确认SciPy1.15.3/NumPy1.26.0；未安装或修改共享依赖。代码身份由本次Git提交和部署大小/SHA记录。

本地AST三文件通过。新增工件控制后，本地7项中4项通过、3项因缺SciPy明确跳过；真实导出阳性及错误关节、速度、身体字段阴性控制通过。源码`82bcbb37180cc9fa80eb0bb7892b47a915f95c10`已推送并独立核对Q1远端，三文件部署大小/SHA一致。服务器同命令原生退出0，7项全通过、0跳过，控制耗时0.431秒；收据`logs/supermimic_q1_D05_controls_20261024.json`952字节/SHA`a1a174c75ff603990d140ccad4da7449f19b67c8d8b2eabbed06bc083cbbfa79`，对应同前缀.log。启动要求30GiB保护线下至少896MiB余量，运行每128帧保留128MiB并核对OOM；本分析失败不会向R03发送信号。

```bash
$PY -m unittest discover -s script/supermimic_q1_analysis -p test_build_support_projection.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY -u script/supermimic_q1_analysis/build_support_projection.py \
  --root "$PWD" \
  --admission data/supermimic_for_q1/R03_B128_extend12000_20261023/run/admission.json \
  --diagnostic data/supermimic_for_q1/D04_reference_dynamics_20261024/receipt.json \
  --audit data/supermimic_for_q1/D04_reference_dynamics_20261024/summary_audit.json \
  --output data/supermimic_for_q1/D05_support_projection_20261024
```

上述候选生成命令已实际执行一次，输出不能重复覆盖。原生进程769443/start3606934714于01:42:21启动，实际退出1，无最终receipt；启动及退出记录位于`logs/supermimic_q1_D05_{native,execution}_20261024.json`。日志`logs/supermimic_q1_D05_projection_20261024.log`4603字节/SHA`4d9221e693d16cc8f5578253b24b60519408917a0562ae1982d7ec9086cdba46`。

走路1652帧、浅蹲1157帧的部分候选已生成，实际足地接触覆盖均100%；绝对原目标身体位置误差P95分别由2.097cm/2.113cm变成4.345cm/2.681cm。二者并不构成完整通过，第三片出拳在原/候选自碰撞逐帧计数或最大深度的严格不变性断言处失败。应先定位具体差异，不能直接放宽阈值。部分NPZ及新候选逆动力学数组保留，未生成伪完成收据，后续若修复使用独立新目录；原D04不重跑。

## 首批失败定位与D05B准备

只对出拳678帧执行CPU定位、0积分/inverse/策略/optimizer。逐帧自碰撞计数完全一致；6帧最大深度差超过原1e-10m检查，最大4.4866912e-7m，低于模型既有`ccd_tolerance=1e-6m`。逐个核对70个机器人几何体，平移误差2.220446e-16m、朝向差0；第一份诊断的all-geom误差包含不应随机器人移动的固定floor，后续明确排除唯一world geom0，不把它误判为机器人变形。

定位收据`logs/supermimic_q1_D05_punch_invariance_probe_20261024.json`8381字节/SHA`cd65e7c3a6405472a55c7e7a4af00aa56b70b9cd603ffeba984d731daaa92625`，补充`logs/supermimic_q1_D05_robot_geometry_probe_20261024.json`1287字节/SHA`18594b7a983a317c43207b683dedd8e4d2ebfe0802fdbb21b501035bf9eb064c`。出拳原/候选均有70/678帧（10.3245%）自碰撞，最深12.4684mm、7.68秒、左髋/左腕；根高候选没有解决自碰撞。

两份实际定位命令保存在`logs/session_records/supermimic_q1_D05_probe_commands_20261024.json`，不包含认证。原控制/退出/两份定位小收据已取回`data/manifests/supermimic_q1_D05_*_20261024.json`，两份执行日志大小/SHA与原生收据一致；未下载NPZ。

独立D05B将数值深度不变性检查绑定模型原有CCD容差，自碰撞计数仍要求精确一致；新增控制验证0.45微米差在1微米内通过、超过容差或计数改变必须拒绝。几何候选设置不变，原D05失败目录不重用。D05B控制通过后使用新输出`data/supermimic_for_q1/D05B_support_projection_20261024/`，并独立保存日志与receipt；当前尚未执行。

新增控制后，本地8项中5项通过、3项缺SciPy跳过；服务器8项控制待执行。首批7项服务器通过的历史证据不冒充新代码已通过。

## 结果与边界

本阶段有两段部分几何候选，无完整导出审计或策略改善证据；零新增训练、策略推理、积分或GPU渲染。R03最近实核为10月24日01:40:53，原controller762428/start3606081278与worker762431/start3606081380同boot非僵尸，4300/12000追加轮、68800新交互、累计7300轮，冻结条件通过，GPU39摄氏度/空闲16671MiB、容器31041597440字节/OOM14；恢复仍重新核对，不能把本快照当实时证明。

## 保存与下一步

源码、报告与后续小JSON/日志按`Q1`保存，提交前审查暂存区和payload，显式推送并独立比对远端提交。大NPZ及模型只留服务器LOCAL_ONLY，不取回Mac。

下一只对出拳做有界CPU平移不变性定位，确认计数/深度差异和数值量级后再决定独立新候选；原失败目录不复用。如几何改善同时动力学残差变坏，保留失败结论，不因足底接触增加自动接入。主线等待原R03结束后的完整配对评测、独立重算及V02渲染，以移动/腿部/全身误差和实际画面判断可用性，不以候选生成代替策略成功。
