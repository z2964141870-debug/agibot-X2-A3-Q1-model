# AMASS P04 Q1候选重定向与独立几何检查

记录日期：2026-10-25，北京时间。状态：16段真实Q1候选转换与独立检查完成，0/16通过全部几何门，训练准入未完成。

## 目标与范围

复用已有SuperMimic for Q1数据接口，将[P03](README_SUPERMIMIC_Q1_AMASS_PILOT_20261025.md)已验证的16段SMPL-X人体参考转为Q1候选，并独立重算原几何/导出门。保留原120→50Hz原速、16betas和世界Z约定，来源别名与受试者划分未完成前不准入训练；不修改正在运行的R02输入、469冻结文件或已停止实验。

## 完成内容与方案

新增`script/supermimic_q1_sources/retarget_amass_body.py`，复用原`validate_egolocate.leg_scale`的双腿骨链长度比、PHUMA已用的grounded_solver/initialize_first_pose、Q1Retargeter及export_q1。人体FK和Q1模型分别在不同进程执行；不载入SMPL-X网格来做IK，不另造训练算法。候选状态保持不可训练，即使转换器几何门通过。

新增`verify_amass_reference.py`，从固定BodyReference重新计算腿长比例、世界Z朝向对齐、全身目标及根姿态，再独立计算Q1速度、FK、导出、足底与自碰撞。原19项几何/导出阈值保持，另加比例、映射和根朝向共22门。检查通过只记geometry_qualified；qualified_training_count固定0，split固定UNASSIGNED_DIAGNOSTIC_ONLY。

两项输入合同控制在Mac和服务器均通过：错误时间/维数/训练标志拒绝，同来源换名称重复拒绝。独立映射控制在服务器通过：30度倾斜标定下，水平位移不转为高度，首帧同姿映射及朝向正确；Mac缺SciPy，该项跳过，不声称本地数值通过。尚未执行真实Q1参考重定向或独立检查。

两个入口均沿用MemoryBoundBudget/guarded_run：共享28.75GiB、0.25秒检查、OOM14、100GiB磁盘余量、每个新输出/来源目录512MiB、900秒无进度保护，仅回收自己的worker。不放宽原训练30GiB保护或几何门。

## 版本、命令与产物

主机`huanyu:/opt/data/private/YuanQi_Q1`；原Python环境`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。人体准备版本`19944a9b9660f7e2d082a1a86b70ff248d9a1319`，Q1新代码在本次阶段提交后从Git历史恢复；发布前不称已推送。

源码两端SHA：转换`ebb7566385f72632097b8e09b21a2acedc2d9f738775e597f839dc727ba46ee5`、测试`935c4f5f9cca0b1199238f9b2f3c91e3faa86cc498cc50b0050b0e7fc92a82ef`；独立检查`8be4936da532b4f1ea31b3f5902ba9fa10b9fe879df55d705c8be038489f6b0d`、映射控制`412493368e2878de19502931db88387fe32735093f95fbde7fb0ef614518302c`。既有leg_scale依赖原已部署且SHA`ca2f174feb5638e5693b68e66044f5b758599ffe640aeea4836d35a01439cca8`一致，不覆盖冻结代码。

控制命令在项目根执行；分别匹配test_retarget_amass_body.py和test_verify_amass_reference.py：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python \
  -m unittest discover -s script/supermimic_q1_sources -p test_retarget_amass_body.py -v
```

以下首批真实命令尚未执行；输出新建且不恢复已有目录：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  script/supermimic_q1_sources/retarget_amass_body.py \
  --root /opt/data/private/YuanQi_Q1 \
  --body-manifest /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P03_body_B001_20261025/body/manifest.json \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P04_Q1_B001_20261025 \
  --expected-oom-kill 14

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  script/supermimic_q1_sources/verify_amass_reference.py \
  --root /opt/data/private/YuanQi_Q1 \
  --motions /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P04_Q1_B001_20261025/references \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P04_check_B001_20261025 \
  --expected-oom-kill 14
```

输出各目录guard.json；转换完整manifest/diagnostics留references/，独立检查留reference_check.json。stdout为`logs/<目录名>.log`。大工件LOCAL_ONLY，Git只保存脚本、配置、报告及小收据。

## 边界与下一步

先4段真实转换及独立检查，确认资源和接口后再处理B002的12段。拒绝项按原阈值登记，不为增加时长放宽门。来源别名/受试者隔离、完整合库及最终动力学效果仍待；几何通过不替代策略移动成功。当前PHUMA R02继续原有限预算，最终配对评测待流水线完成。

## B001真实转换与独立检查

本阶段准备发布`da7c8be02e60e97bdf0ff0a6fd8a5e55b068f42e`已独立核远端Q1。01:04:51至01:05:01，转换guard829939/worker829940退出0并回收，4/4真实转换、0异常，转换核心5.294秒。01:05:40至01:05:46，独立检查guard829972/worker829973退出0并回收；0/4通过全部22门，0训练准入。两个阶段OOM14，结束内存约29262532608/29262442496字节，未触发保护。

失败门重叠计数：cartesian_residual 3、support_within_15mm 3、support_gap_at_most_30mm 2、converter_gates 3。各片IK P95为57.123/69.823/54.086/38.947mm，足底15mm内比例0.0945/0.1095/0.9851/0.2388；最大足底间隙48.089/47.978/17.767/26.733mm。自碰撞穿透均0。原阈值未放宽，4片不能计入Q1合格时长。

独立目标映射误差最大4.441e-16m、根位置导出误差最大5.943e-8m，时间/比例/根朝向等门通过；这排除了本批已检查的时间轴与根运动丢失，但不证明重定向具有足够保真或动力学可行。下一按原固定名单处理剩余12段，仍不合库、不动R02输入。

## B002与16段结果

01:07:08至01:07:28，B002 guard830035/worker830036转换12/12成功、0异常并回收；01:08:39至01:08:53，检查guard830099/worker830100退出0并回收。两批合计16候选/64秒，0/16通过全部22门，0新增Q1合格时长、0训练准入。所有候选split仍UNASSIGNED_DIAGNOSTIC_ONLY；源到Q1适配失败不能据此泛化为AMASS整体无用或判定当前PHUMA策略训练原因。

16段失败门重叠计数为cartesian_residual 11、support_within_15mm 15、support_gap_at_most_30mm 10、converter_gates 11；其余18门每片均通过。IK位置P95范围38.947至102.108mm；目标映射误差最大4.441e-16m、根位置导出误差最大5.943e-8m，未丢失根运动。首批挑选四秒温和移动、人体FK正确和Q1参考可用是不同条件，当前不扩大同配置转换以追求表面时长。

独立检查原件各位于B001/B002 check目录的reference_check.json；[B001原样小副本](../data/manifests/supermimic_q1_amass_P04_check_B001_20261025.json)12215字节/SHA`81c4807b5f74b9a977d8146b90587669cd4476cbb3dd3a542e0d594764106422`；[B002原样小副本](../data/manifests/supermimic_q1_amass_P04_check_B002_20261025.json)33434字节/SHA`2a8e2a86458a449df7d48fc67f7aac7a27baaeec3b080c05f70f16a80285ad57`。每份记录22门与独立FK/速度/足底/自碰撞数值。

01:10:25合计检查、四个worker原生退出0/回收及资源汇总完成：[小收据](../data/manifests/supermimic_q1_amass_P04_summary_20261025.json)6850字节/SHA`7aec1a0c3b39eaf81c931227659c101a152a327126a0e7edd6659eb503363971`，原件`data/supermimic_for_q1/AMASS_P04_summary_20261025.json`。两批转换/检查目录合计实际分配11608064字节，身体参考约2.49MB，未新增大型Mac资产；原件LOCAL_ONLY。

同次读取原R02仍RUNNING：3650轮/233600交互、WM1409组/actor1359次、7279.59秒（121.33分钟），GPU40摄氏度/OOM14，共享内存29270376448字节、盘余1965455032320字节。469冻结在00:56:59已全量核对未变，本阶段新增脚本不在其内；没有重启、改输入或追加原训练预算。尚无R02最终动力学结果。

下一先对AMASS本批的源/目标/IK足底及残差做小范围归因，再决定独立映射改进；不将0/16失败参考喂入训练，不为增加数据量放宽门。HumanML来源索引已于01:07重新接收成功，见[P03更新](README_SUPERMIMIC_Q1_AMASS_PILOT_20261025.md)，下一需实际核对跨源对应和受试者划分。原PHUMA R02继续原有限预算与终态配对评测；30分钟/1小时合格库和移动策略成功目标仍未完成。
