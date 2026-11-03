# Q1五阶段条件转接门禁

2026-11-03。正式目标已由应用get_goal实读为active，本线程q1心跳ACTIVE、每10分钟。当前仍监督原Primebot正式训练；本阶段只增加独立门禁纯函数和反例测试，尚未把共享训练控制器、原生评测和综合比较全部接通。

## 本阶段结果

新增script/q1_five_stage/contracts.py及test_contracts.py，Mac标准库CPU测试10项全部通过，0新物理步、0optimizer、0agent。它们不在原Primebot104份冻结输入内；训练执行版本仍b8f920cbba9be3f7247bddbe6535febfbc06186b。

最终转接必须重新检查：Primebot训练/交付终态、固定三片九次和共同有效前缀两项误差、全部393片初末评测、六份初末原速视频、初末检查点审计，以及与交付清单完全相符的Mac大小SHA/逐帧解码/目视收据。临时观测不能授权Morph；缺失、重复、非有限、提前跌倒、无共同前缀或不完整本地交付均拒绝。

共享配方从既有完整原生模板构造，保留完整网络、H32/K8、WM每轮10批、五优化器和同片同帧latent对齐。首档各1024环境/WM512/motion与KP1024/160000轮/WM-only2000/预热400；容量回退各512环境及对应减半batch/320000轮/WM-only4000/预热800。两档各型目标都是410009600真实转移，更新次数差异保留，不续profile权重。

八轮验证必须两rank都完成，实际更新与保存Adam80/8/160/8/160一致、非有限0、实际共享参数SHA与配对抽样一致、机器人presence buffer独立。只有正常回收的WORKER_FAILED、原输入未变、cgroup OOM增量0、GPU recovery=None且明确CUDA out of memory，才允许配置容量回退；内存守卫、信号终止、设备故障和通信异常均不允许。

整轮磁盘预算用两rank真实保存体积计算，包含编号检查点、latest及并发临时副本、8GiB证据余量、25%空间余量，另外保留100GiB空闲。原生v0.1 Trainer不保存replay，不借用Primebot的replay存储公式。

## 实际监督快照

21:50:30北京时间，原controller29435/start2148365、guard29965/start2153863、worker29970/start2153901仍在，4259完整轮、43773952真实转移，WM42590/actor4259/critic85180。guard RUNNING，GPU22244MiB/67度/recovery=None，容器17458606080字节、OOM0。此为有时间戳的快照；未重复全部冻结文件内容审计，仅核新包不在原104输入内。

已完成的1000轮观测仍是固定9/9但无正常抬脚换步；下一次既定物理观测5000轮，当前未早停，不能据训练曲线称自然步态改善。

## 复现与下一步

```sh
python3 -m unittest discover -s script/q1_five_stage -p 'test_contracts.py' -v
```

原始收据：logs/session_records/q1_five_stage_contracts_20261103.json。后续必须补齐真实文件身份检查和串行controller、原生motion/KP/G1评估及比较，再用本门禁约束启动；本纯函数模块不直接启动任何作业。保持唯一hp_han Windows原生25362入口、原boot/GPU、24/30GiB内存/86度/100GiB与GPU独占锁；异常正式训练不自动重启。

整个目标保持active，用户采纳null；不能把本组件测试通过或等待训练记作五阶段已完成。执行发布和服务器测试以旁收据为准。
