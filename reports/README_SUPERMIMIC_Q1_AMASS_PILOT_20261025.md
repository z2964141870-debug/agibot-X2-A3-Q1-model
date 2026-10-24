# AMASS原速移动候选准备

记录日期：2026-10-25，北京时间。阶段：SM-Q1-AMASS-P02。状态：两端4项控制及00:06真实原始窗口选择完成，43次探查选出16个KIT原动作/64秒；00:08八项小清单重算通过。0新Q1参考/训练，原PHUMA R02继续。

## 目标与范围

用户停止BONES获取，要求利用现有资源继续SuperMimic for Q1。原PHUMA R02继续12000轮有限预算；P06并行转换B017已触发提前内存保护，详见[P06](README_SUPERMIMIC_Q1_PHUMA_CONCURRENT_EXPANSION_20261024.md)。本阶段不重启该worker或放宽内存线，先利用已有AMASS标准SMPL-X原始库做小规模、无torch/MuJoCo的来源及数值准备。

当前AMASS完整头部索引17265标准动作，男女/中性16维身体模型已到位，见[模型接收](README_SUPERMIMIC_Q1_SMPLX_INTAKE_20261024.md)。23:59:59索引轻量查询找到990个名称含walk、16betas、至少481帧、文件不超过16MiB的候选；此数只按名称/头部条件，不是动作语义或Q1质量证明。PHUMA额外名单的来源键包含humanml编号，AMASS与PHUMA混用前须核对原动作对应，不能只比较不同格式文件SHA就宣称跨库隔离。

## 完成内容与关键决定

新增`script/supermimic_q1_sources/select_amass_pilot.py`，复用既有`inventory_amass.inspect_motion`与完整文件SHA身份函数。按seed20261025加相对路径的SHA顺序固定探查，排除名称明确含run/hop/jump/leap/box/stair/backward/crouch的项；最多64个原文件探查、预期16个候选，每个只用最早481帧/4秒的原始120Hz窗口。原文件最多2500帧/16MiB，完整pose与五个分字段在窗口内一致且有限，16betas全部有限、不截断；源头部/mtime/字节和读取前后SHA核对。

候选原始平移初门为最大水平位移0.2至2m、水平速度P95不超过1.5m/s、Z平移跨度不超过0.2m。这仅用于优先准备较温和移动，不验证身体关节/足底/碰撞；未改PHUMA原19门。planned 50Hz/201帧只登记计划，没有实际重采样、FK或Q1导出；不把64秒候选原料记成合格训练时长。

所有候选明确`split=UNASSIGNED_DIAGNOSTIC_ONLY`、`qualified_for_q1_training=false`；未建立训练/开发/测试划分，未纳入原R02。完整原动作身份、不同裁片与HumanML对应以及跨源/受试者隔离待后续登记。不在未解决别名关系时宣称独立未见移动测试。

每个文件前后检查共享cgroup内存低于28.5GiB、OOM14及100GiB磁盘余量；无torch或机器人模型载入，原文件只读，读后仅对该文件使用posix_fadvise释放本轮读缓存，不改系统缓存设置。输出新目录排他创建，失败保留failure.json。

## 版本、命令与证据

权威工作副本`/tmp/yuanqi-supermimic-q1-20261023`、Q1，准备前已发布并独立核验`9981c092727943d22136ef93720b066f9ed25ab5`。本阶段新增源码的实际SHA和部署/执行结果在下方登记；不修改原R02的469冻结文件。

本地4项控制通过：解析精确四秒/481个120Hz源帧与16betas、静态和高速平移不被选入、错误帧率/10betas/pose字段不一致被拒绝、原索引元数据变化被拒绝。命令：

```bash
python3 -m unittest discover -s script/supermimic_q1_sources -p test_select_amass_pilot.py -v
```

实际服务器命令如下，部署/代码SHA/资源/输出不存在核验后已执行，原生退出0：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  /opt/data/private/YuanQi_Q1/script/supermimic_q1_sources/select_amass_pilot.py \
  --root /opt/data/private/YuanQi_Q1 \
  --source-root /opt/data/private/yuanqi_ops/script/amass_bridge/data/amass_raw/AMASS_smplxg \
  --index /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P01_inventory_20261024/all_files.jsonl \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P02_moving_candidates_20261025 \
  --count 16 --maximum-probes 64 --seed 20261025 --expected-oom-kill 14
```

## 结果、边界与下一步

准备时真实选择尚未执行；真实结果记在下方。后续用已验证16维身体模型和现有原速SMPL-X/站姿标定接口做独立小样本FK/Q1重定向及质检，保持原训练优先，资源不足时继续原训练并保留未完成状态。

本阶段文件原件仅服务器LOCAL_ONLY，Mac仅脚本/报告/小清单。完成结果后同步报告索引、当前状态和指定GitHub Q1分支，并独立核验远端。原R02训练保存检查已通过，仍无最终跟随效果结论。

## 真实选择与独立小清单重算

入口SHA`6fbf505c6fc90a6f08338880ba5e9159e8ec21469b12640180e51ef1007b4f23`、测试SHA`0ae627da6d64e30eb99c26d7c24bca775a96a96efd5f301aff36a6625b73fbf5`及依赖inventory_amass.py SHA`96af20e269f5c71b91cd209aaf614c892e4661680b2982e5d7340c6de6e9dae3`，两端一致。两个新增入口不在原R02的469冻结清单内；服务器4项控制全部通过，未安装依赖。

00:06:11真实选择原生退出0，用时0.945秒；较990个宽松名称候选，进一步按2500帧上限及排除词后829个可探查。固定顺序检查43个原文件，16个通过原始平移初门、27个未通过此初门；无字段/身份异常导致退出。选中16个均为KIT，女7/男9，各4秒，合计64秒原始窗口；源字节SHA均唯一，但不称受试者独立或多数据集合覆盖。全部UNASSIGNED_DIAGNOSTIC_ONLY、Q1训练标志false。

共享内存开始29045911552字节，最高/结束29053472768字节，OOM14；无torch、SMPL-X或MuJoCo载入，未生成Q1参考/动力学步/optimizer更新。源索引12198595字节/SHA`501e723765a71167e7186f90df0791b506191cb83fbe98a103810d86caf2a233`保持一致。模型、输入及正在训练的R02未修改。

真实输出`data/supermimic_for_q1/AMASS_P02_moving_candidates_20261025/selection.json`，81611字节/SHA`acffdda17a9c6cc3467cf69322b0cd4e4be4dfd8bb1bafaf553e6e4115c9736f`；[原样小副本](../data/manifests/supermimic_q1_amass_P02_selection_20261025.json)。stdout在`logs/AMASS_P02_moving_candidates_20261025.log`。未复制原AMASS动作到Mac或其他数据目录；输出只有来源、统计与计划，候选原料64秒不增加原Q1合格时长。

00:08:22另一个轻量只读进程完成八项清单重算：16条/源SHA唯一/各初门通过、481帧真实4秒/16betas、未分配split且未训练准入、64秒算术与0新Q1参考/物理/训练。此过程没有重复载入原始数值或模型，不能称独立FK或质检。

同次核验原训练guard825182/controller825183/trainer825189的boot及start_ticks和/proc命令、活跃状态均匹配，R02到1875轮/120000交互、WM699组/actor649次，训练3605.05秒（60.08分钟）、GPU40摄氏度/OOM14，storage_guard仍RUNNING。完整[资源/进度小快照](../data/manifests/supermimic_q1_existing_resources_progress_A_20261025.json)为7548字节/SHA`0aac315e6842de655435b4665cb146f6057c34551f8ccd904b40e7a305720364`，原服务器路径为R02目录`resource_progress_20261025_A.json`。

下一先查HumanML来源别名与已固定PHUMA三划分；AMASS只做候选级FK/映射质量准备，在合库或建立独立移动评测前完成来源/裁片/受试者隔离。PHUMA P06待原训练释放资源后新目录从offset160完整继续，旧B017失败目录及原R02输入不覆盖。当前30分钟/1小时合格库、未见移动成功及最终动力学视频目标均未完成。

## P03跨源元数据与人体参考准备

本轮恢复已实核00:18:01原R02三个/proc进程、boot及start_ticks一致，2125轮/136000交互、WM799组/actor749次，GPU39摄氏度/OOM14，storage/controller仍RUNNING；内存29026942976字节。前一目标轮完成真实数据选择和保存检查，属于实质进展，本轮继续未完成的原速接入，不重启训练或已停实验。

已读HumanML3D官方README、MIT LICENSE和原始预处理Notebook；公开来源映射为index.csv。通过git ls-remote固定main提交`9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23`。元数据接收入口`intake_humanml_metadata.py`（SHA`fd6416ec8385609c05622f232dbf00a285bb341701e5553d6ce7346d1a34acdb`）在服务器HTTP读取超时，exec64503原生退出1；目录`data/references/humanml3d_metadata_9176e8fb_20261025`保留，日志`logs/AMASS_P03_humanml_metadata_20261025.log`。Mac curl同一固定index.csv连接重置；随后partial clone连接GitHub 443超时，原生退出128，未获取索引。暂缓相同网络重试，尚未接收/解析索引，不宣称跨源隔离完成。仅尝试小型公开索引/文档，0动作下载，不继续BONES。

新人体参考入口`prepare_amass_body.py`复用原dataset_input的SMPL-X16 FK、完整forward三帧核对、合成同形体站姿标定与egolocate_input原速120→50Hz插值。拆分人体FK与后续Q1模型加载以降低同时驻留内存；当前仅实现，真实执行待代码/资源核验。候选固定最早四秒，50Hz应201帧；split未分配且训练标志false，人体参考不是Q1参考。

新入口的父进程沿用既有MemoryBoundBudget和guarded_run，28.75GiB/0.25秒检查、新OOM停止、只回收自己的worker组；至少100GiB盘余、输出最多512MiB、身体模型目录最多2GiB、900秒无进度保护。Q1 robot model不会在此FK阶段载入；原训练30GiB线保持。所有派生身体数组与16维形体数值留服务器被忽略目录，Git仅收不含betas的completion小摘要。

下一先两端控制，再独立B001的offset0/count4（覆盖男女两个模型）做真实人体FK，资源通过后处理余12个候选，最后独立Q1映射与几何质检。原HumanML/PHUMA别名和受试者隔离未完成前不合库、不建立训练/测试资格。

00:33:56再次读取原R02：2575轮/164800交互、WM979组/actor929次、5099.42秒，三个原进程boot/start_ticks及命令活跃匹配；GPU39摄氏度/OOM14，共享内存29028413440字节、盘余1966006308864字节，存储guard/controller仍RUNNING。一次只读查询误用cgroup v2路径而退出1，已改读实际v1路径后得到本快照，没有写入或改变作业。

新增人体入口SHA`b97aef00241cf819c100aedbff729811b9ca520cbdf81102d5d0f5f5d4b2d2b9`、测试SHA`bc666b967d57a8591fccf343e6f47124bd22ff44b0899238ce6715aa569aa9d4`已部署并两端一致。dataset_input、egolocate_input、inspect_egolocate及q1_retarget/模拟/映射依赖逐SHA一致；三个旧输入模块原已部署，无须覆盖。新增入口不在469冻结清单内。Mac合同控制通过、插值控制因缺SciPy/MuJoCo跳过；服务器两项控制全部通过（0.472秒），没有安装依赖、加载身体模型或新增动力学步。

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python \
  -m unittest discover -s script/supermimic_q1_sources -p test_prepare_amass_body.py -v
```

B001以下命令在本次准备记录时尚未执行，使用新目录排他创建；真实结果须另行登记，不把代码控制通过当成原数据FK完成：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  script/supermimic_q1_sources/prepare_amass_body.py \
  --root /opt/data/private/YuanQi_Q1 \
  --selection /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P02_moving_candidates_20261025/selection.json \
  --model-intake /opt/data/private/YuanQi_Q1/data/references/body_models/smplx_lockedhead_20230207/verification.json \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P03_body_B001_20261025 \
  --offset 0 --count 4 --expected-oom-kill 14
```
