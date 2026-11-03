# Q1共享训练与目标监督

2026-11-04。应用目标实读active，原心跳q1已通过应用工具更新并读回ACTIVE、每10分钟、同一线程。监督指令已转向当前共享阶段，保留Primebot终态，不再等待已结束的旧PID。整个五阶段目标尚未完成，用户采纳null。

## 首轮与转接

Primebot已在10000轮正常早停，固定9/9、全393/393和Mac完整交付通过；完整转接门重算passed/authorizes_morph_training=true。跟踪改善不等于自然步态合格，低脚底离地和承重滑移仍在。详见[首轮结果](README_PRIMEBOT_Q1_FINAL_RESULTS_20261104.md)。

## 首次共享入口失败

原根data/supermimic_for_q1/MORPHMIMIC_Q1_CONDITIONAL_SHARED01_20261103于00:45:59启动控制器、00:46:04结束。e1024/profile/guard.json明确STOPPED_STORAGE、worker_started=false、worker_reaped=true；没有启动训练worker，没有新增物理转移或优化。pipeline保留FAILED和原异常，不解释成容量CUDA OOM，也没有进入512回退或正式训练。

来源是通用Primebot guard的64MiB源码目录上限；实际MorphMimic发布目录分配142503936B、逻辑141858407B、341个inode。现场磁盘空闲525553401856B；这是源码目录预算不适配，不是磁盘耗尽。新GPU UUID和boot匹配，25度/recovery=None/1MiB，cgroup OOMkill0。原始guard、status和trace继续保留在原失败根。[诊断收据](../logs/session_records/q1_shared_source_budget_20261104.json)。

## 修正范围

guard增加可选maximum_source_bytes，原默认64MiB保持，参数必须为正整数且最多256MiB。共享V02配置显式设置256MiB，覆盖实测约136MiB的固定发布树，仍继续统计整个目录并拒绝超限。100GiB磁盘余量、24GiB启动/30GiB运行、OOM0、86度、GPU锁和设备身份检查保持。

controller将配置传入共享guard并记下源码预算。新配置script/q1_five_stage/conditional_shared_v02_20261104.json使用独立MORPHMIMIC_Q1_CONDITIONAL_SHARED01_V02_20261104根，其余数据、网络、环境和训练配方不变。首次失败发生在worker创建前，修复后仍按原授权完成首次实际共享验证；不续失败根或profile权重。

本地4项源码预算回归、7项controller与10项contracts检查通过，共21项；1项真实Torch保存测试本机缺Torch跳过，发布后服务器CPU补验。新测试复现原64MiB拒绝、256MiB接受实测树、超过256MiB仍拒绝、非法参数拒绝，同时核原资源线不变。

## 执行顺序

发布修复、独立40位远端和三端工件核验后，服务器CPU检查实际目录预算和既有门。通过后启动V02各1024环境、393条共同数据、8轮完整更新；两rank共享参数、五优化器80/8/160/8/160、CPU保存审计与实存磁盘核算全部通过，控制器才自动全新初始化正式160000轮。只有明确容量CUDA OOM且正常回收、输入同、OOM0和GPU健康才允许各512/320000；正式异常不自动重启。

新启动命令在唯一ZHY根执行，VERIFIED_FULL_COMMIT必须来自本阶段独立远端核验：

```sh
data/environments/morphmimic_original_20261029/bin/python -u script/q1_five_stage/controller.py run --config script/q1_five_stage/conditional_shared_v02_20261104.json --code-commit VERIFIED_FULL_COMMIT
```

此页修复准备时尚未启动V02；实际执行与发布身份以后续launch/publication收据为准，不把计划写成训练进度。原生Q1 motion/KP、G1初末评估、原速媒体和综合比较仍须完成。早停Primebot缺少等410009600曝光检查点如实缺失，不追加训练。

只复用hp_han Windows原生SSH至25362的既有会话78774；本次连接可用，未重建SSH或重启服务器。目标模式持续监督当前实际pipeline、guard、PID/starttick、完整metrics、GPU/cgroup和保存；无变化安静，有新异常、质量变化或完成主动报告。不开agent，不改其他自动任务。
