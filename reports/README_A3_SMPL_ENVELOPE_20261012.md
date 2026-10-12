# A3 多顶点脚地面约束 E22

实验执行日期：2026-10-12，北京时间。实验：A3_ZHY_SMPL_ENVELOPE_20261012_E22。状态：VERIFIED_QUALITY_REJECTION，位置门7/7失败，合格pair0。用户继续指令沿用无需新人体编码器路线，先拆参考转换；本轮不训练编码器、策略或运行机器人。

## 目标与固定变量

E21的候选积分几何检查使7/7脚不穿地，但频繁停步导致7/7位置跟踪失败，合格pair0。E22仅把旧最近碰撞点的脚/地面限制替换为每个实际脚geom局部AABB的8顶点平面限制，12个geom共96行。直接采用位移QP形式`-Jz * delta_q <= .85 * (point_z - .001)`，不再把Mink速度h作为当前平面边界。点Jacobian包括geom中心平移与角速度造成的偏移，先用有限差分核验。

每帧目标、root/关节代价、120初始化步、后续8步/240Hz、7段各300帧、旧质量门均不变。保留E21真实脚几何非线性验收/最多12次缩步，仍有不安全候选则保持。AABB比mesh保守，可能过度抬脚或妨碍跟踪；该代价必须按原贴地/位置/root门判定，不能只看穿地改善。

## 版本、运行与保护

源基线HEAD`df8552466b7ea72e875f2cf705f16a4493bde244`。新增`script/a3/smpl_vertex_floor.py`、`smpl_envelope_probe.py`与`test_smpl_vertex_floor.py`，无vendor或历史E14–E21代码改动。原hp3090仅Git、stdlib传输；计算只在ZHY`/opt/data/private/hechengzhu/ZHY`，既有`data/environments/a3-sonic/bin/python`、单线程CPU、CUDA不可见，不加载神经网络权重。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.test_smpl_vertex_floor
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_envelope_probe register
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_envelope_probe run
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_envelope_probe verify
```

注册须同boot已核验E21、无并行计算、可靠温度/内核遥测。5秒健康采样、CPU90/GPU85°C连续两次保护、4GiB内存保留、900秒上限、首故障停止，无重试/自动恢复。注册固定代码/父工件/输入SHA，不重置已有目录。

产物`data/experiments/a3_zhy_smpl_envelope_20261012_E22/`，日志`logs/a3_zhy_smpl_envelope_20261012_E22/`，小型清单`data/manifests/a3_zhy_smpl_envelope_20261012_E22.json`。独立验收不重跑IK，从保存qpos复算FK、实际脚几何、目标/CSV/质量门并核验SHA与逐步积分证据。原始NPZ/CSV/日志保持服务器LOCAL_ONLY，Mac仅小代码/README/清单。

## 实际结果与拒绝理由

两项CPU契约于0.116秒通过，日志`logs/a3_smpl_envelope_tests_20261012_E22.log`，未重复运行旧E21测试。2100帧于23:56:44–23:57:26计算完成，23:58:12独立核验通过执行完整性，质量拒绝。

7/7真实脚几何不穿地，7/7关节限位/速度、支撑滑动、root修正及root速度通过。每段2512次积分，合计17584次，全候选直接接受，缩步/保持均0；E21每段348–923次保持已消除。仍7/7位置P95超5cm，2/7最大值超15cm，squat贴地/根跟踪失败，不能训练。

| 片段 | 位置P95 cm | 最大cm | root位置P95 cm | 失败项 |
| --- | ---: | ---: | ---: | --- |
| zj walk | 5.699 | 15.368 | 1.502 | 位置P95/最大 |
| zj warmUp | 5.828 | 9.278 | 0.800 | 位置P95 |
| fjw walk | 5.501 | 10.164 | 0.748 | 位置P95 |
| zls walk | 5.300 | 12.371 | 0.566 | 位置P95 |
| yy squat | 7.228 | 13.796 | 6.702 | 位置P95/接触贴地/root跟踪 |
| yy walk | 5.394 | 11.995 | 0.755 | 位置P95 |
| zzx walk | 5.446 | 16.493 | 1.465 | 位置P95/最大 |

位置P95范围从E21的5.44–24.90cm降为5.30–7.23cm，全部保留原门槛。根速度P99为0.661–1.222m/s；squat左/右接触脚底高度P95为20.065/19.308mm，左脚超原20mm门。合格pair0、optimizer0、策略回放0，无CUDA初始化。

11健康样本UTC15:55:22.701–15:58:12.320，同boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`，CPU最高43/GPU26°C，可用内存最低9,294,499,840字节、OOM0，无关键内核/遥测错误。最大82.07秒采样空档位于注册到执行的阶段间，不宣称全程连续监测；计算期间沿用5秒保护。作业已退出，无自动恢复。

## 保存构型的几何诊断

新增独立只读CPU分析`script/a3/smpl_envelope_diagnose.py`，未改执行时注册代码。按已核验trace的SHA读取2100个qpos，复算真实脚底并与已存高度比较（1e-12容差），再计算AABB最小高度，逐帧验证真实mesh被包络包含；不重跑IK或策略。实际模型为12足geom、96平面行。质量对象内`floor_collision_pairs=6`是替换前继承的碰撞pair元数据，不是本轮实际平面行数；历史清单原样保留，当前约束覆盖以几何诊断为准。

所有七段接触时段的AABB高度中位数约1mm，但真实左/右脚底中位数16.856/17.343mm。差值中位数分别15.856/16.343mm；squat差值P95升至19.065/18.308mm。因此“外接盒贴地而真实脚悬空”已有数值证据，外接盒虚拟角点过于保守确实引入了竖向偏移；尚不能把全部位置误差都归因于它。

诊断产物`data/experiments/a3_zhy_smpl_envelope_20261012_E22/envelope_geometry_diagnosis.json`，小清单`data/manifests/a3_zhy_smpl_envelope_20261012_E22_geometry.json`，记录验证结果/代码/各trace SHA。复现只读分析：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_envelope_diagnose
```

运动学合格也不等于动力学平衡、准确的人体世界轨迹或超过官方。七段已用于开发诊断，不作为未见最终泛化集。下一单变量建议采用每个真实foot mesh凸包的实际顶点平面约束，保留非线性检查和所有旧门槛；使用现成凸包实现并核验其在任意平面方向上保留真实mesh最低点，不能通过把地板向下挪或放宽门槛掩盖空角点。该下一实验尚未实现/登记/运行。

本阶段完成后立即更新此README、报告索引/current state，审计scoped diff、运行`script/project_tools/audit_git_payload.py`，提交推送main并核对远端。结果通过再准备冻结官方闭环；失败则保存具体约束/跟踪冲突，先定位，不训练畸形标签。恢复见[E21](README_A3_SMPL_CHECKED_20261012.md)与[衣服/接触路线](README_A3_CLOTH_CONTACT_ROUTE_20261012.md)。
