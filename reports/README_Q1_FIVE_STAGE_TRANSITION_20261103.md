# Q1五阶段条件转接门禁

2026-11-03。正式目标已由应用get_goal实读为active，本线程q1心跳ACTIVE、每10分钟。当前仍监督原Primebot正式训练；新增共享串行控制入口、Mac交付核验和被动初始保存检查。真实共享八轮/正式启动仍须Primebot最终全门和本地交付通过；原生评估及综合比较尚未接通。

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

22:00补记：d9e733a7ca08fbf5f51b0ed7bff053560ee7d2d6已独立40位远端核同，8份480980字节Git/Mac主根/25362逐大小SHA一致；服务器10项CPU测试也全部通过。未新增物理或优化，正式训练冻结版本保持。[发布核验收据](../logs/session_records/q1_five_stage_contracts_publication_20261103.json)。

## 串行入口准备

新增controller.py、delivery.py、snapshots.py及独立条件配置。controller在Primebot最终服务端评估和Mac完整验收全部通过前，拒绝创建共享实验目录或Runner；启动前重新核393条配对数据、冻结子集、原v0.1源码/实际资产和既有新卡G1/Q1原生控制结果。使用原guard.py及同24/30GiB、86度、100GiB和GPU锁，不修改运行中的Primebot104输入。

实际顺序为每型1024八完整更新→两rank实际共享和五优化器核验→独立CPU保存审计→按两份真实检查点体积核算整轮磁盘→全新正式160000轮。只有正常回收、输入未变、cgroup OOM增量0、GPU recovery=None且明确CUDA容量OOM，才另起每型512八轮及320000轮配方；设备、信号、内存保护或审计错误均不降档/重试。正式作业异常保留，不自动重开。

train_native.py仅增加显式passive_checkpoint_audit开关；旧计划未启用时保持原行为。新计划保存实际转移和五优化器计数；正式零更新保存前后独立比较全部模型/归一化器参数与buffer、Python/NumPy/Torch及已初始化CUDA RNG，不变才通过。检查点0及终点均独立CPU核实际元数据、Adam步数和模型/优化器有限值，不续profile权重。

delivery.py生成完整数值工件及六初末原速视频清单。Mac逐文件大小SHA、视频尺寸/帧率/完整RGB帧解码，并要求另外提供绑定每份视频SHA的目视收据；程序不会自行冒充目视已验收。所有NPZ和检查点保留服务器。root相对冻结子集路径显式归一化到ZHY，避免误用旧559条清单。

本地27项新包CPU测试中24通过、3项真实Torch保存/审计检查因Mac无Torch跳过；原native回归10项中9通过、1项同原因跳过。服务器使用既有Morph环境的CPU验证待本阶段发布后执行，无新增物理/策略推理/优化或共享启动。controller正常结束标为SHARED_TRAINING_COMPLETED_PENDING_EVALUATION，不结算应用goal；Q1 motion/KP、G1逐片和综合比较工具仍须继续实现及实测，不能称五阶段已自动贯通。

服务器最终门通过并完成Mac交付后，才按发布后独立核验的40位执行提交调用：

```sh
data/environments/morphmimic_original_20261029/bin/python script/q1_five_stage/controller.py run --config script/q1_five_stage/conditional_shared_20261103.json --code-commit VERIFIED_FULL_COMMIT
```

当前Primebot仍运行，该启动命令尚未执行，不为验证入口额外启动仿真。
