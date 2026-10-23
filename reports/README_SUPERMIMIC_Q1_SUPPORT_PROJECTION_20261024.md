# Q1独立根高支撑参考候选

记录日期：2026-10-24，北京时间。阶段：SM-Q1-D05/D05B。状态：首批D05失败保留；D05B三片生成、54项导出审计与276项动力学统计重算完成。几何接触改善但走路逆动力学残差明显变坏，不直接接入训练；R03继续原固定预算。

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

## 首批失败定位与D05B

只对出拳678帧执行CPU定位、0积分/inverse/策略/optimizer。逐帧自碰撞计数完全一致；6帧最大深度差超过原1e-10m检查，最大4.4866912e-7m，低于模型既有`ccd_tolerance=1e-6m`。逐个核对70个机器人几何体，平移误差2.220446e-16m、朝向差0；第一份诊断的all-geom误差包含不应随机器人移动的固定floor，后续明确排除唯一world geom0，不把它误判为机器人变形。

定位收据`logs/supermimic_q1_D05_punch_invariance_probe_20261024.json`8381字节/SHA`cd65e7c3a6405472a55c7e7a4af00aa56b70b9cd603ffeba984d731daaa92625`，补充`logs/supermimic_q1_D05_robot_geometry_probe_20261024.json`1287字节/SHA`18594b7a983a317c43207b683dedd8e4d2ebfe0802fdbb21b501035bf9eb064c`。出拳原/候选均有70/678帧（10.3245%）自碰撞，最深12.4684mm、7.68秒、左髋/左腕；根高候选没有解决自碰撞。

两份实际定位命令保存在`logs/session_records/supermimic_q1_D05_probe_commands_20261024.json`，不包含认证。原控制/退出/两份定位小收据已取回`data/manifests/supermimic_q1_D05_*_20261024.json`，两份执行日志大小/SHA与原生收据一致；未下载NPZ。

独立D05B将数值深度不变性检查绑定模型原有CCD容差，自碰撞计数仍要求精确一致；新增控制验证0.45微米差在1微米内通过、超过容差或计数改变必须拒绝。几何候选设置不变，原D05失败目录不重用。D05B使用新输出`data/supermimic_for_q1/D05B_support_projection_20261024/`，并独立保存日志与receipt。

新增控制后，本地8项中5项通过、3项缺SciPy跳过。修正代码`c340e6831a87f23d9a9014b4a2ae4b80de8f6984`已推送并独立核对Q1远端，原部署SHA核对后只更新analysis目录。新服务器8项控制实际通过、0跳过；收据/日志独立保存在`logs/supermimic_q1_D05B_controls_20261024.{json,log}`。首批7项历史证据与新8项分开保留。

01:54:44原生worker770040/start3607008986启动，实际退出0，三片3487帧均生成。收据32517字节/SHA`36ee154b35844d01920e3e38a75b4082378961e0f77ee08d106d1597df5dfd80`；日志`logs/supermimic_q1_D05B_projection_20261024.log`6790字节/SHA`1efad466930f68c58ea225fcd020e467d79df07b5d9d6bb2c21a387c790979f7`。走路/浅蹲与原失败批次已完成的对应NPZ SHA完全一致，出拳新候选1059548字节/SHA`816ce863ed73e81e62acc38fd29f8372647b3c77956edbda6dd69dc69f434b99`。三片实际足地接触覆盖均100%，未接入训练；随后完成的两项独立审计见下节。

## D05B独立审计与结果

两项审计均实际原生退出0。`audit_support_projection.py`54项通过，确认原字段/水平路径/全部关节/时间保持、根竖直速度与新根高差分一致、身体位置/速度满足平移恒等关系、实际支撑带及自碰撞计数保持；最大检查差3.3998243e-7，小于相应2e-6导出容差。`audit_reference_dynamics.py`对新候选3475分析帧/6950次inverse的276项统计和差分重算通过，最大差1.7763568e-15，不复算原D04、不增加积分或NN优化器更新。

```bash
$PY script/supermimic_q1_analysis/audit_support_projection.py \
  --root "$PWD" --receipt data/supermimic_for_q1/D05B_support_projection_20261024/receipt.json \
  --output data/supermimic_for_q1/D05B_support_projection_20261024/export_audit.json
$PY script/supermimic_q1_analysis/audit_reference_dynamics.py \
  --root "$PWD" --receipt data/supermimic_for_q1/D05B_support_projection_20261024/dynamics.json \
  --output data/supermimic_for_q1/D05B_support_projection_20261024/dynamics_audit.json
```

两个已完成命令均拒绝覆盖，不作为恢复时自动重跑入口。`export_audit.json`2542字节/SHA`b7fb53cbd059b3e012bfa8baae8d297ecdb5d5481b45839e7c51589cbe0ef241`，`dynamics_audit.json`13558字节/SHA`2e8c9e520a455883f48c9b1e1ccf74766a290e47093c3da38edbd2894438e98a`。比较命令保存在`logs/session_records/supermimic_q1_D05B_comparison_command_20261024.json`，比较JSON7229字节/SHA`980ba9eaeda67cb566c96613e69ec4227a6bea930213c1ff931247944f857365`，绑定原/新收据和两项审计。

下表为中央三点差分、每片排除首尾各2帧的相同区间；自碰撞比例另按完整片段统计。

| 片段 | 活跃足地接触覆盖原→新 | 根所需外力中位数/体重原→新 | 任一关节所需力矩超cap的帧比例原→新 |
| --- | --- | --- | --- |
| 走路 | 0.06068%→100% | 1.0000→0.9165 | 0.06068%→29.4296% |
| 浅蹲 | 0→100% | 1.0000→0.9290 | 0→0.2602% |
| 出拳/复用开发 | 0→100% | 1.0004→0.9111 | 12.3145%→22.1068% |

走路根所需外力矩P95由2.66794升至96.08892Nm，最大主动关节所需力矩/cap由1.57653升至25.61968；中央五点也呈同方向变化，超cap帧29.2476%、外力矩P95 96.38887Nm。几何贴地没有建立合理接触动力学。身体绝对原目标位置误差也增大，走路P95为4.3447cm（原2.0969cm）。出拳自碰撞70/678帧及最大约12.47mm穿透继续保留。

决定：D05B仅保留为诊断/失败对照，不直接替换训练参考或启动新训练。根外力和逆动力学来自精确姿态、速度、差分加速度与软接触，并未优化接触力分配，因此这些结果不证明近似动作跟随不可能，也不证明R03必然失败；它们明确不支持“只修根高已解决参考动力学”的结论。下一参考方向应同时核查支撑脚速度/姿态及自碰撞约束，再做匹配检查。

D05B98项输入身份前后保持，未导入Torch；实际峰值RSS756320KiB，容器31074164736→31786549248字节、OOM始终14，未触发30GiB保护。零新增物理积分、策略推理、NN优化器更新或GPU渲染。候选仍全部`qualified_for_training_pilot=false`，大型NPZ只在服务器LOCAL_ONLY。

## 结果与边界

首批失败与D05B已完成诊断均保留，尚无新策略改善证据。02:03:18收尾核验98项输入身份仍一致，控制/启动/原生退出/两项审计及比较证据逐项大小/SHA绑定；`completion_verification.json`5027字节/SHA`123157d5e1d2ea397f0f6ee9e1d067dd4aca75b25a4f6587b3e3ceedfe436052`。

R03最近实核为10月24日02:03:18，原controller762428/start3606081278与worker762431/start3606081380同boot非僵尸，4950/12000追加轮、79200新交互、1929组WM/1879次actor、累计7950轮，冻结条件通过，GPU39摄氏度/空闲16671MiB、容器31081000960字节/OOM14，无终态收据；恢复仍重新核对，不能把本快照当实时证明。

## 保存与下一步

D05B13份小收据/日志已取回，逐项本地大小/SHA与服务器完成清单一致；转移核验`data/manifests/supermimic_q1_D05B_transfer_verification_20261024.json`。Git证据为`data/manifests/supermimic_q1_D05B_*_20261024.json`及`logs/session_records/supermimic_q1_D05B_*_20261024.{json,log}`，首批失败独立保留。大NPZ及模型只留服务器LOCAL_ONLY，不取回Mac。

源码已发布`82bcbb3`及修正`c340e68`，均独立核对Q1远端；本次最终结果记录随阶段提交，提交前审查暂存区和payload，显式推送并独立比对远端提交。完整提交号由Git历史提供；不在提交前把新记录记作已同步。

本阶段定位/候选/审计已经结束，不重复运行。主线等待原R03结束后的完整配对评测、独立重算及V02渲染，以移动/腿部/全身误差和实际画面判断可用性。若R03仍不足，再准备同时约束支撑脚速度/姿态与自碰撞的独立重定向候选及匹配比较；当前不把D05B接入训练，不增加R03预算或恢复旧STOPPED实验。
