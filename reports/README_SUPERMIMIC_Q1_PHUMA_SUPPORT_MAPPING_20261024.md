# PHUMA足底高度映射诊断

记录日期：2026-10-24，北京时间。阶段：SM-Q1-PHUMA-D02/D02B/D02C。状态：原PHUMA R01全部完成后，D02C八训练片/1536帧分析原生退出0，32项几何一致性及独立264项数组统计重算通过；121冻结文件未变。此前D02导入失败和D02B内存保护退出1保留，两次均无样本输出。没有新训练或修改输入。

## 目标与范围

[D01](README_SUPERMIMIC_Q1_PHUMA_DIAGNOSIS_20261024.md)确认112候选中85片支撑比例失败、44片自碰撞失败（重叠），但尚未区分PHUMA源轨迹本身、机器人目标映射和IK残差各自对支撑不足的影响。本阶段只读八个固定训练集样本，比较同一50Hz时间轴上的源G1碰撞足底、Q1目标脚刚体姿态和原始Q1 IK参考足底。development/test不用于本次诊断选型。

新入口位于 `script/supermimic_q1_analysis/analyze_phuma_support.py`，不在当前R01冻结集合内。直接复用固定PHUMA解码/重采样和独立足底查询函数；不执行IK、积分、策略推理、训练或修改任何参考。源ZIP不重复全包顺序读取，仅按已固定成员读取并逐份核验原始大小/SHA；固定源46文件和R01的121冻结输入重新比对。

## 方法与控制

三种足底曲线各保留原始世界高度。另用每条整片曲线一个固定平移比较时间变化；源曲线乘既有腿长比例表达为Q1米。该诊断平移不写回轨迹，不是逐帧根高投影，也不证明动力学可行。Q1目标脚是任务给定的刚体姿态，不保证双脚/全部关节能同时到达。

新刚体几何查询支持sphere/box/mesh，四项控制覆盖带局部平移的旋转球、旋转盒垂直投影、网格顶点/多几何最低点和未知/空几何拒绝。Mac和服务器均原生退出0、4项通过。真实分析拟逐帧把刚体查询与MuJoCo实际几何比较，同时复现原PHUMA源FK和既有支撑统计；两次真实尝试均在样本处理前失败，不能宣称逐帧检查通过。CPU进程禁用CUDA，至少1GiB初始余量，距原30GiB训练保护线不足256MiB时停止诊断，OOM增量亦拒绝。

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

首次预定输出`PHUMA_D02_support_mapping_20261024/receipt.json`和修正后D02B目录均未创建；D02B日志包含真实stdout/stderr，已取回并核对大小/SHA。未来独立D02C的NPZ仍只留服务器，Mac仅取小型JSON/日志，不重复既有输出目录。

## 版本与复现

当前训练版本仍 `ed4217384e7abfc8a9cea0b23b95592d4a3e19b0`。本诊断代码提交由本次Q1 Git历史提供，发布并独立核验后才部署运行；运行收据另记实际诊断发布号。Python沿用服务器已有环境，不安装依赖。复现入口：

```bash
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" \
  script/supermimic_q1_analysis/analyze_phuma_support.py \
  --root /opt/data/private/YuanQi_Q1 \
  --motions /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_P03_references_20261024 \
  --training-spec /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_R01_B128_3000_20261024/spec.json \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_D02C_support_mapping_20261024 \
  --code-commit "$DIAGNOSTIC_COMMIT" \
  --clips train_locomotion_1bc87468bec5a472 train_locomotion_26832240824d6963 train_locomotion_fe9abef3042c3c39 train_locomotion_1f75cd47fbf61bb7 train_locomotion_6cafc84ec90e5186 train_locomotion_007047b45054fd5a train_squat_04ebe3a321c47d2f train_upper_a5678394f936531f
```

上述D02C入口已在原流水线完成且重新核验内存后执行，原生退出0。输出排他创建，不能重跑覆盖同一目录。原始NPZ留服务器，Mac仅保存小收据和日志；以下结果不替代真实动力学跟随评测。

## 首次执行失败与修正

诊断发布`210969ad6b2ee5e438d50c86e5f2d80e61eae2a1`已推送并独立核验Q1远端，两份源码部署SHA一致，服务器四项刚体控制原生退出0。首次真实执行原生退出1：导入`verify_reference.foot_bottom`时ImportError，既有函数实际名为`lowest_foot`。错误发生在创建输出目录和读取真实样本之前；原始`logs/PHUMA_D02_support_mapping_20261024.log`仅tee stdout而为空，stderr由原生工具会话返回，已明确区分留档于[失败记录](../logs/session_records/supermimic_q1_phuma_D02_failed_20261024.log)。不能将单元几何控制当作真实端到端分析通过。

本修正只改变分析脚本三处函数引用，不改R01冻结的既有验证器、参考或训练入口。后续采用独立`PHUMA_D02B_support_mapping_20261024`输出目录和日志，先在服务器执行包含真实依赖的导入检查，再分析原定八片。首次失败的退出码与原日志必须保留。

13:47:38只读核查原controller799647/worker799650仍为同一父子进程、同一启动时间和boot；1625轮/26000控制转移/599组WM/549次actor。GPU40摄氏度、cgroup30,838,591,488字节、OOM仍14。该快照不代表最终效果。

## D02B资源保护停止

修正发布`248c2cf26ae179be50e989486e0763bd0adc2d77`已推送并独立核验Q1远端；服务器分析源码SHA`8b445549ce41fc850600ca872f0f689a8602fd6fa0763363ac35afa13fa648c3`与发布一致。13:52:03真实依赖导入、原boot/start_ticks/父子进程及至少1GiB余量检查通过：cgroup31,133,540,352字节，OOM14，训练1775轮。

D02B实际运行原生退出1，在源/目标MuJoCo模型创建后的`health()`因距训练30GiB保护线不足256MiB主动中止。它尚未创建输出目录、读取八个选定成员或形成样本结果；没有新IK、物理积分、策略调用、optimizer更新或参考修改。错误日志未打印触发时具体内存，因此不编造峰值。两次失败均保留，不覆盖重跑。

13:52:53立即复查原训练仍为同一controller/worker及启动身份，status RUNNING、1775轮/28400控制转移/659组WM/609次actor；cgroup已回落30,843,056,128字节，OOM仍14。D02和D02B两个预定输出目录均不存在。结构化[失败及训练核查](../data/manifests/supermimic_q1_phuma_D02B_memory_guard_20261024.json)、[原生失败日志](../logs/session_records/supermimic_q1_phuma_D02B_memory_guard_20261024.log)已保存，日志588字节/SHA`2dc224c05a9e28476af412b873b5837eb790d30ac8050e5926b91d8a94d45c46`与服务器一致。

决定：停止在本轮训练期间并行尝试这项模型诊断，不放宽训练/诊断保护、不干扰其他作业。等待原3000轮及匹配评测流水线结束，再用独立D02C目录和最新资源核验恢复。D01的支撑/碰撞筛选计数仍有效，但源轨迹、目标映射和IK残差的具体归因尚无新证据。本阶段代码和失败记录随Q1提交，完整提交号由Git历史提供；大型源数据/权重仍LOCAL_ONLY。

## D02C真实完成与独立重算

14:57:51恢复预检通过：同一boot、原controller为完整七阶段完成、原进程组无活跃成员，cgroup27,362,304,000字节，距30GiB保护线4,849,950,720字节，OOM14、GPU38摄氏度。分析使用已发布`248c2cf26ae179be50e989486e0763bd0adc2d77`，部署源码SHA与前节一致；输出与日志此前不存在。R01/V01报告已同步服务器并逐份SHA与本地一致。

D02C原生退出0，状态`PHUMA_SUPPORT_MAPPING_DIAGNOSIS_COMPLETED`；8片共1536帧，逐片源FK与旧trace完全一致，Q1刚体足底与MuJoCo几何差不超过3.61e-16m，原筛选支撑比例和最大间隙精确复现，共32项布尔检查通过。15:00:48独立进程重读8份NPZ，重算六种曲线各五项统计、两个曲线RMSE和踝部任务P95，共264项一致；时间轴均50Hz、数组有限/大小/SHA一致、121冻结文件未变。

下表的源与映射支撑比例采用各整片一次固定平移到最低点为零的诊断曲线，源乘既有尺度表达为Q1米；解后列为原参考。没有逐帧投影或写回参考。

| 训练clip后缀 | 原资格/失败项 | 源固定落地15mm内 | 映射固定落地15mm内 | 原IK参考15mm内 | 映射/源曲线RMSE(mm) | IK/映射曲线RMSE(mm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1bc87468bec5a472 | 合格移动 | 100% | 100% | 100% | 5.974 | 1.454 |
| 26832240824d6963 | 合格移动 | 100% | 100% | 100% | 1.737 | 2.071 |
| fe9abef3042c3c39 | 移动/仅支撑失败 | 25.50% | 47.65% | 42.28% | 4.539 | 3.040 |
| 1f75cd47fbf61bb7 | 移动/仅支撑失败 | 44.51% | 10.44% | 10.44% | 6.847 | 1.023 |
| 6cafc84ec90e5186 | 移动/仅碰撞失败 | 100% | 100% | 100% | 2.590 | 1.367 |
| 007047b45054fd5a | 移动/仅碰撞失败 | 100% | 97.75% | 98.20% | 5.462 | 2.989 |
| 04ebe3a321c47d2f | 蹲/仅支撑失败 | 53.77% | 34.67% | 38.69% | 1.714 | 0.617 |
| a5678394f936531f | 上肢/仅支撑失败 | 43.22% | 7.54% | 8.54% | 5.719 | 0.558 |

原始源G1较低足底最小高度为-10.40至-45.69mm，原世界坐标下8片都100%不高于15mm；该单边条件把穿地也计入，不能直接叫物理接触合格。对四段仅支撑失败的片，源曲线固定落地后已有17.68–20.45mm波动范围，映射后为20.81–27.61mm；原IK曲线相对目标的形状RMSE0.56–3.04mm。因此证据支持源足底高度变化与目标映射已贡献该小样本的支撑筛选缺口，不能把它全部归为IK求解误差。两段仅碰撞失败片没有因此变合格，碰撞原因尚未定位。

两段已用于R01的移动参考仍100%在15mm内，最大间隙14.29/11.77mm。它们仍分别早停或移动不足，故本诊断不能解释全部策略失败，更不能直接证明修正源落地就会学会行走。固定平移、运动学近地和动态可行是不同判据；不降低现有质量门、不接入拒绝片、不重复旧D05B逐帧根高投影。

诊断采样cgroup最大28,834,037,760字节，结束重算时27,399,184,384字节，OOM仍14。0新IK、物理积分、策略调用及optimizer更新，参考未修改。原始输出在服务器`data/supermimic_for_q1/PHUMA_D02C_support_mapping_20261024/`，日志`logs/PHUMA_D02C_support_mapping_20261024.log`；全部大型原件仍LOCAL_ONLY。

[诊断收据](../data/manifests/supermimic_q1_phuma_D02C_support_mapping_20261024.json)为21763字节/SHA`c212010f76426ebc43091dade10e7d01a32444937ab1171950e129dd09a26302`；[原生日志](../logs/session_records/supermimic_q1_phuma_D02C_support_mapping_20261024.log)15992字节/SHA`57ea4ca55f9830fef876afee0437cd0f226e6dfdf1ac39d4a1030ef072e541b0`。两份取回核对，另存[预检与独立重算](../data/manifests/supermimic_q1_phuma_D02C_independent_check_20261024.json)及[重算源码](../logs/session_records/supermimic_q1_phuma_D02C_reconciliation_source_20261024.txt)。本阶段记录审计后提交并显式推送Q1，独立核对远端。

下一项先针对PHUMA R01的训练/开发动作检查奖励分量和世界模型H1/H8/H32预测，区分姿态奖励收益与根移动误差；不把旧A3来源实验的模型诊断直接套用到PHUMA，不先延长预算。下一训练须独立登记，当前未启动。
