# X2 流水线资源分配 R2

2026-11-13，PREPARED_NOT_DISPATCHED。用户允许评估适当放宽20ms，要求先继续优化资源分配。R1吞吐29.853Hz但策略超时和参考年龄回退，保留完整失败取舍；原961GPU帧预算结清，不重复原实验。

## 独立变量与实现

新增`script/x2_joint_alloc`，包装已冻结R1的GPU/CPU/发送入口，不重写原作者递归、模型、物理、180ms预览或衣服默认。第一步仅改变自有任务的CPU affinity，以便先测小范围调度分配能否缓解争用；进程隔离仍为后续候选，没有宣称已实现或已定位GIL原因。

| 自有执行任务 | CPU编号 |
| --- | --- |
| native只读observe | 0,1 |
| GPU主机提交/衣服校准 | 2,3 |
| HumanFrame转换/发送 | 2,3，独立线程 |
| FGP LIP/GP/原生物理 | 4,5 |
| Morph策略服务 | 6,7 |

运行前必须实核继承可用CPU含0..7；不足则拒绝，不扩大系统cpuset。若原作者实际创建兼容物理子进程，只包装其精确`-m stage_tran_gp.worker`命令分配4,5，命令内容不改；原native后端只允许observe。每执行角色记录真实TID与核掩码。其他系统服务、ROS控制、IRQ、电源和时钟不改，绑核不代表独占或实时保证。

本地六项合同实际通过，耗时0.007秒，两CLI通过；这是接口/分配合同，Mac没有执行Linux真实绑核，更没有板端模型调用。首次测试发现统计工具重复传入budget_ms参数，已修复并重测六项全过，未影响任何旧工件。

总在途2、确认后归还、状态后不丢弃保持；七项原并发合同不重做。新独立六项合同检查受限核集合拒绝、线程实际掩码、native observe和worker精确重写、旧R1门不能解锁新组、放宽统计不改历史接受事实。

## 20ms 与 30Hz

20ms是原50Hz策略周期，现有native_controller.cpp硬编码该限制。用户允许稍放宽，因此新增保存日志分析同时报告20/25/30/33.333ms超时数。它是事后时限敏感性，不可把旧拒绝改成真实接受，也不证明真实30Hz。当前R1完整RPC样本仍设备，25/30/33.333精确比例待原日志分析，不能由p95/p99猜计数。

Mac原串行最好组的8982791B完整native.jsonl（SHA `6d4abfb65aa35c5b1547d7c593d8eb8fb16b7aa606ee22e0e87aa7f269ba4c4c`）已完成保存数据分析，无新推理。1492个ready请求在20/25/30/33.333ms下分别超过6/2/1/0次；历史实际接受1486不改。结果`data/analysis/X2_JOINT_PIPELINE_20261113/deadlines_serial_best.json`。R1是不同并行负载，不能套用此计数。

为隔离资源分配变量，R2首次比较仍原50Hz请求、原20ms实际门并列全部新门槛；若选择30Hz策略，后续独立observe构建将周期/超时成对设置，writer高频循环与状态新鲜度门分别保留，未更改现有终端2/3。放宽时限不会减少180ms预览或实际源年龄。

## 新预算与执行门

新预算独立于R1：先唯一`X2_JOINT_ALLOC_20261113/prefix_r2`，复用同串行实际93输入/64输出，完整pose/root≤2e-5及新allocation/affinity来源、三角色TID/CPU核门过后，才唯一`joint_r2`同929源最多900GPU帧。总≤964真实帧+4graph初始化；100秒控制器/120秒外层、4GiB可用/80度/10GiB磁盘、原项目非阻塞锁和原observe保护保持。无engine构建/训练/新BLE/电机或MC操作。

先源码审计发布/独立远端40位、Mac/设备精确SHA与原247源24资产及R1终门不变、实际身份boot/任务/资源/锁准入。每个新输出/日志/派发收据/进程先防重。不能因SSH未确认就重派。完成后核原来源终门/自有进程清理，再比较实际吞吐、间隔、所有ready RPC、分段CPU时间、源年龄与四门槛，不只看接受率。

```bash
# 设备项目根；先来源/资源与防重门，通过后各执行一次
env -u PYTHONPATH -u PYTHONHOME timeout --signal=INT --kill-after=10s 120s data/environments/fgp-preview-py310/bin/python -B -m script.x2_joint_alloc.run prefix --fixture data/analysis/X2_JOINT_LATENCY_20261113/fixture_r1/imu929.npz --reference data/analysis/X2_JOINT_PERF_20261113/all_sleep1_fk --output data/analysis/X2_JOINT_ALLOC_20261113/prefix_r2
env -u PYTHONPATH -u PYTHONHOME timeout --signal=INT --kill-after=10s 120s data/environments/fgp-preview-py310/bin/python -B -m script.x2_joint_alloc.run joint --fixture data/analysis/X2_JOINT_LATENCY_20261113/fixture_r1/imu929.npz --parity data/analysis/X2_JOINT_ALLOC_20261113/prefix_r2 --output data/analysis/X2_JOINT_ALLOC_20261113/joint_r2
```

## Done / To Do

用户：首测已确认吞吐改善但延迟回退；资源分配新实现和多门槛分析已准备，尚无R2实测收益。下一实际准入后测绑核，之后按证据决定进程隔离或30Hz部署候选。

Agent：原40802第二跳明确Connection aborted后回Windows，随后一次原生run@10.0.1.41重连TCP超时，Windows会话仍在；不泛化整桥接/机器人停止。上次原机器人boot df9b9a2e及R1完成/原件已回Mac仅历史，R2部署/当前板端资源UNKNOWN。新恢复条件后先原生身份/同实体根/boot、R1结束/锁/温度内存磁盘、允许CPU与原源码SHA，不重做R1或旧串行组。未下发、0新GPU，不记录为运行中。Q1暂停/goal不用/新agent不用/采纳null。
