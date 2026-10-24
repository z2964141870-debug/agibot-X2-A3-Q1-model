# AMASS P04 Q1候选重定向与独立几何检查

记录日期：2026-10-25，北京时间。状态：代码准备及服务器控制通过，真实Q1转换待执行。

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
