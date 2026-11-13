# X2 FGP 有界流水线

2026-11-13，状态 PREPARED。用户明确要求使用流水线优化。新增独立`script/x2_joint_pipeline`，保留原模型、准确mixed_gemm TensorRT engine、LIP/GlobalPose递归状态、30Hz物理和180ms预览；默认衣服启动和终端2/3不切换，SDK/电机/MC/训练/改频均0。连接沿Mac原生Windows桥接，再原生run@SoC1交互密码。

## 目标和实现

已成串行最好配置all_sleep1_fk为24.805Hz，FGP中位34.422/p95 38.972ms，Morph全部ready RPC p95 14.866ms；原结果复用、不重跑。新方案将GPU CANO/校准、CPU LIP+GlobalPose+原生物理、SMPL转换/ACK分成三个执行阶段。GPU执行context始终只有主线程使用，CPU及发送各单线程串行，ZMQ client由发送线程创建和关闭。每个线程使用所需的独立inference_mode上下文。

全流水线总共最多两帧在途，生产者获得额度后才选最新已到达原始帧；CPU状态更新后不丢帧、不换序，发送确认后才归还额度。过载只在原始输入准入前取最新，逐例记录跳帧；不以增加FIFO积压换吞吐。两帧上限仍可增加排队，必须一起看参考年龄和CPU/发送等待，不能只看Hz。

CPU路径调用原作者RealtimePipeline.step，只将其中的denoiser替换为一次性已完成结果适配器。原始CANO实例由生产者独占，窗口与EMA只推进一次；CPU继续使用原LIP/GP/物理/有效性门和无Unity偏置的gp.last_result。连续已测录制条件专用，不能把这个入口当完整真实BLE断流/重标定实现；适配器拒绝跨线程重置。

## 验证与预算

本地五项CPU合同已过：双阶段顺序和两帧上限、线程归属、CPU/发送失败传播与回收、准备结果单次消费及原native observe命令保持。新worker/probe CLI通过。真实设备GPU暂未执行。

先独立PIPELINE_PREFIX门：精确复用串行最好组前93个实际处理输入（29历史+64输出），不替换为连续理想索引；最多64次真实GPU帧+原graph两次初始化，CPU完整物理顺序保留，与保存64个pose/root比最大绝对差2e-5。无真实策略请求/反馈/BLE/电机；两个保存来源SHA绑定。100秒守卫/120秒外层，失败根保留。

此前缀门及原资源/来源/锁通过后，唯一新joint_r1根做同929原始数据/理想30Hz释放、真实ROS只读反馈和原native observe。最多900次GPU帧+两次graph初始化，复用原100秒控制器/120秒外层、55秒native寿命和20ms接受边界，4GiB可用/80度/10GiB磁盘门及项目非阻塞锁；不中途扩限或重启旧结果。GPU帧总上限964，不训练、不构建engine、不运行电机。

真实壁钟阶段保留每帧GPU阶段、CPU排队/执行、发送排队/FK/ACK和source年龄，采样capture时间不提前或重标新鲜。不把分阶段耗时相加当并行wall；原fgp_ms定义为当前输入开始到CPU姿态就绪的完整单帧时延，另报实际输出间隔。所有native ready与拒绝/p95/p99/最大值保留。达到30Hz还须检查跳帧和完整率；本短测不认证真实BLE、人体精度、机械响应或长时稳定。

## 可复现命令

设备项目`/home/agi/x2_morph_human_20261113`，现有fgp-preview-py310环境。先新根：

```bash
env -u PYTHONPATH -u PYTHONHOME timeout --signal=INT --kill-after=10s 120s data/environments/fgp-preview-py310/bin/python -B -m script.x2_joint_pipeline.probe --fixture data/analysis/X2_JOINT_LATENCY_20261113/fixture_r1/imu929.npz --reference data/analysis/X2_JOINT_PERF_20261113/all_sleep1_fk --output data/analysis/X2_JOINT_PIPELINE_20261113/prefix_r1
```

只在首门真实通过后：

```bash
env -u PYTHONPATH -u PYTHONHOME timeout --signal=INT --kill-after=10s 120s data/environments/fgp-preview-py310/bin/python -B -m script.x2_joint_pipeline.run --fixture data/analysis/X2_JOINT_LATENCY_20261113/fixture_r1/imu929.npz --parity data/analysis/X2_JOINT_PIPELINE_20261113/prefix_r1 --output data/analysis/X2_JOINT_PIPELINE_20261113/joint_r1
```

已存在根不得重派。完成后只读保存数据分析、与原all_sleep1_fk比较、生成六宫图/CSV及证据包。source integrity前后同和自有进程/锁回收通过后发布结果。

## Done / To Do

用户版：流水线代码和本地并发检查已备；尚无本轮实际提速结论。下一先64输出一致性门，再做同机壁钟短测。

Agent版：六新文件、原控制器/策略worker复用，最多两帧额度，依赖文件不热改；原四组/929串行结果不重做。先发布源码与精确SHA包、沿Windows桥接run核同boot/原来源/无自有测试任务/锁/资源，再板端五合同、唯一prefix根及其guard终门，成功后唯一joint根。异常保留失败，不擅改原作者/权重/物理或放宽原final门。原Q1训练保持暂停，goal/新agent不使用。
