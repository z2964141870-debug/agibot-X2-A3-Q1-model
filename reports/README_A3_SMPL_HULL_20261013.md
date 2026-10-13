# A3真实脚凸包地面约束E23

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_HULL_20261013_E23，VERIFIED_KINEMATIC_CORPUS，6/7开发参考通过原运动学门，下蹲拒绝。此前E22保留，不重复或修改旧注册文件。

## 目标与唯一变量

E22真实脚不穿地且无停步，但位置门7/7失败。保存构型诊断已确认外接盒包含真实mesh以外的空角点，接触期真实脚高于包络约1.6cm。E23仅以真实mesh凸包实际顶点替换AABB角点；既有SciPy ConvexHull选择原始顶点，退化或Qhull失败时保留全部真实顶点，不抖动或近似几何。

继续原每帧目标、代价、120初始化/后续8步/240Hz、7段各300帧、1mm间距/.85增益、非线性积分验收及全部质量门。原5cm位置P95/15cm最大值、贴地、滑动、root/关节速度与限位门不放宽。几何安全不等于动力学平衡；本轮不训练编码器或策略。

## 实现、版本与复现

源基线HEAD9df0ef75776fb478d8796abaf30d06f6101cb374。新增smpl_hull_floor.py、test_smpl_hull_floor.py、smpl_hull_probe.py，历史E14–E22文件不修改。QP沿用已核验的位移/角速度Jacobian；独立验收从保存qpos复算FK/实际mesh/凸包最低点、CSV与原门。注册绑定版本/代码/输入及SciPy Qhull二进制SHA。

计算只在ZHY /opt/data/private/hechengzhu/ZHY，原hp3090仅代码/Git/标准库传输。既有环境、单线程CPU、CUDA不可见。5秒监测、CPU90/GPU85°C连续两次停止、4GiB保留、900秒上限、首故障停止；不恢复或重复实验。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.test_smpl_hull_floor
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_hull_probe register
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_hull_probe run
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_hull_probe verify
```

工件data/experiments/a3_zhy_smpl_hull_20261013_E23/，日志logs/a3_zhy_smpl_hull_20261013_E23/，测试日志logs/a3_smpl_hull_tests_20261013_E23.log；小清单data/manifests/a3_zhy_smpl_hull_20261013_E23.json。原始工件LOCAL_ONLY，Mac仅小代码/报告/清单。

## 验证状态与下一步

3项CPU几何测试0.137秒通过，日志SHA256 `72f0eaf24b319c77e9ab4e9abbebd4b07be55e39b5fb9f99cf24b761d3a9040c`。首次登记准备在写job前发现SHA辅助函数要求Path而不是str，已修正；当时目录未创建/计算未启动，不是重启或负载故障。正式注册09:24:35；计算09:25:10–09:26:12共62.14秒，09:35:11独立验收。

12foot geom原始8225顶点减少为2739真实凸包顶点/平面行，无合成角点或退化回退。2100保存帧逐geom真实mesh/凸包最低点差均0；独立FK/CSV/原门/输入SHA及17584积分证据通过，每段2512积分、缩步/保持均0。质量对象`floor_collision_pairs=6`是旧pair元数据，实际行数以geometry_contract为准。

| 片段 | E22位置P95 cm | E23位置P95 cm | 最大cm | 接触脚底L/R P95 mm | 结果 |
| --- | ---: | ---: | ---: | --- | --- |
| zj walk | 5.699 | 3.138 | 12.084 | 1.000/1.004 | 通过 |
| zj warmUp | 5.828 | 4.603 | 7.796 | 1.489/1.364 | 通过 |
| fjw walk | 5.501 | 2.845 | 6.755 | 1.172/1.107 | 通过 |
| zls walk | 5.300 | 3.154 | 9.950 | 1.126/1.000 | 通过 |
| yy squat | 7.228 | 6.290 | 13.943 | 1.578/1.056 | 仅位置P95失败 |
| yy walk | 5.394 | 2.908 | 9.668 | 1.000/1.000 | 通过 |
| zzx walk | 5.446 | 3.067 | 12.431 | 1.000/1.000 | 通过 |

7/7不穿地，限位/速度、滑动/贴地、root修正/速度/跟踪全部通过。合格6，optimizer0、policy rollouts0、无CUDA；下蹲仅位置P95 6.290cm超5cm，拒绝。`training_ready=true`只表示合格片段覆盖原三分区的运动学合同，不表示策略/部署或未见最终验收通过；七段均已用于开发。

15健康样本UTC01:24:35.947–01:35:11.006，同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`，CPU最高47/GPU26°C，可用内存最低9,323,794,432字节、OOM0，无关键内核/遥测错误。最大538.59秒空档是执行结束到验收之间，不算持续运行保护覆盖；正式计算有5秒保护。任务已退出，不自动恢复。

新增`smpl_hull_diagnose.py`只读保存trace，核验SHA与误差复算，不重跑IK/策略。下蹲局部P95：左膝13.178cm/左髋9.716cm/右膝9.169cm/右髋7.124cm，分别85/34/48/24帧超过5cm；这些部位最差帧在2.833–3.067秒。脚底门已经通过，下一拆腿目标、关节轴/限位可行性及代价冲突；以上是误差定位而非根因证明。保存`body_error_diagnosis.json`与[小清单](../data/manifests/a3_zhy_smpl_hull_20261013_E23_body_errors.json)。认证临时连接在分析启动前失效，重新连接后确认同boot，未重跑E23。

下一阶段冻结官方035测六段合格参考，另拆下蹲；运动学不等于平衡或人体世界轨迹准确。路线见[持续工程](README_A3_LONGTERM_20261013.md)，保存后审计提交推送main。

用户长期工作请求已通过Codex app automation_update工具创建本聊天工程heartbeat zhy-a3，ACTIVE、30分钟跟进；从最新README与实际作业恢复，持续推进有证据的下一变量，不开机恢复或重跑故障分支。原yuanqi-a3/hp3090暂停、每日研究yuanqi不变。官方文档检索本次未返回内容，功能状态以工具创建回执为准。本阶段代码/README/两项小清单随Git提交保存，提交号由Git历史提供，同步以实际HEAD和远端main一致的核验为准。
