# MorphMimic 与 PrimebotTech SuperMimic-Q1 接入

2026-11-02，用户在完成 X2/T2 发布策略回放后要求继续尝试 Q1 训练，并提出结合 PrimebotTech SuperMimic-Q1 与 MorphMimic。本阶段落实这一授权，不创建新应用目标，不改变其他自动任务。

## 方案与边界

- 使用 MorphMimic v0.1 的共享世界模型、策略/critic、独立机器人 replay 和同源同帧 latent 对齐；先做 G1+Q1 两本体接入。新的共享网络从零初始化。
- Q1 使用 PrimebotTech SuperMimic 的分组软件 PD、50 Hz 控制、2 ms 物理步长、位置目标延迟及转速限矩契约，资产使用已核的 Q1 V3 力矩 motor 模型。该库示例指向外置 V2，V3 接入须单列版本差异，不能称完整复刻其未随库分发的资产。
- 对照为相同 Q1 资产、数据和控制契约的 Q1 单体训练。共享训练必须记录每个机器人实际转移数、optimizer 次数及共同动作；仅训练 Q1 或先后训练独立模型都不算 unified。
- 数据由同一批 SOMA 分别做真实 G1/Q1 重定向；旧 AP03 与共同 1h 的 559 条仅有 9 条同源交集，不能直接作为完整成对输入。衣服演示仍搁置。
- 仍保留原算法奖励，先核按尺寸缩放的具体位置、避免重复缩放；不把两个代码库的奖励权重直接相加。短期不追加 PPO 残差实验。

## 已查明的接口差异

MorphMimic 原版统一 29 关节、14 跟踪 body、240 维状态、93 维 proprio、670 维 motion goal。Q1 V3 实际只有 22 个 hinge，缺少 `waist_pitch_joint` 和左右各三个腕关节，但固定腕 body 仍可用于跟踪。

原 `Translator`、action 的 `perm` gather/scatter 和 morphology 的 log 执行器特征均假设 29 槽真实存在。直接填 `-1` 会错误引用最后一个真实关节；仅在 motor 输出端丢弃多余动作也会让世界模型利用不存在的自由度。因此需要同步约束参考、动作/探索噪声、last_action、观测/encoder bias、世界模型预测和训练误差。浮点缺失槽用显式选择置零，不能依赖 `NaN * 0`。

拟将缺失关节 mask 明确加入 morphology，形成 154 维新契约（原 125 + 29 mask），并仅使用存在槽拟合 min/max。此架构不兼容发布 ONNX 的 125 维 morphology，不能据发布策略已验证就宣称 Q1 零样本可用。

自训 G1 旧 aa40 版本没有 morphology 且 KP goal 版本为 2；v0.1 为 10。PrimebotTech 没有随本次固定源码提供 Q1 训练 checkpoint，MorphMimic 发布包只有部署策略，不能恢复完整 WM/critic/optimizer 共享训练。

## 固定来源

- MorphMimic v0.1：`41c8382f127a6f5109bfe047445a9180f71f1fc0`。
- PrimebotTech/SuperMimic：`118338b19dcd14f768d1e7de5a48d04b78987688`。
- SOMA retargeter：`cbdb2b494a2af82137bf9b6f15122d60167281c4`。
- 服务器资产：`/opt/data/private/ZHY/data/references/q1/model_20261023_P01/q1_v3/xml/q1_v3_mc.xml`。
- 小源码审阅副本：`data/references/morphmimic_q1_review_20261102/`，仅审阅，不修改原冻结树。

## CPU 接口验证结果

准备提交 `a3be66d419be7d5b5bcbb904dc4adf73bc22131d` 已独立核远端 Q1，9 文件 415585 字节在权威 Git、Mac 主根、25386 逐大小/SHA 相同后执行。2026-11-02 19:33:47 CPU 实模核验完成，结果见 [模型契约](../data/manifests/morphmimic_q1_cpu_contract_20261102.json) 和 [12 项测试](../logs/session_records/morphmimic_q1_cpu_tests_20261102.log)。

- 12 测试全部通过，包含缺失槽 NaN 不进入真实关节、缺失动作梯度为零、腰 joint 顺序重排、signed/offset 一般映射、limits、状态/观测/goal/context mask、真实关节误差分母和 descriptor 缺失槽边界。
- Q1 V3 为 `nq=29/nv=28/nu=22/nbody=27/ngeom=71`；14 个跟踪 body 存在。64 组原生关节姿态的 canonical 往返误差及静态 FK body 位置变化均为 0。
- 22 关节零位正轴与 G1 方向余弦最小约 0.93969，两侧零位前臂方向余弦约 0.99965，支持此模型使用正号、零 elbow offset 的候选映射；不沿用 X2 的 elbow `-pi/2`。这不是两个机器人 FK 相等或任意动作可达的证明。
- 固定 Primebot 源码通过 AST 只加载两组 PD 表和三个 tensor 函数，未实例化其完整 simulator。128 组速度/目标 × 10 子步 × 22 motor 共 28160 控制数值与源公式最大差 0；涵盖按方向转速限矩、延迟目标选择、gear 和 XML ctrl bounds。尚未验证 mjlab 原生 actuator 回调、随机化 encoder bias/PD 因子或求解器轨迹等价。
- 154 维 descriptor 仅验证缺失槽和缩放算法，实模测试中的 geometry 使用人工全 1，不能当作 Q1 的实际形态向量；真实几何和 G1/Q1 训练群体边界待提取、冻结。
- 五个明确列出的只读来源前后 SHA 相同，未建立 CUDA context；新增动力学积分、策略推理、optimizer 均为 0。19:35:49 原 boot/GPU 核验相同，GPU 15 MiB、无计算进程、OOM0。

实现位于 `script/morphmimic_q1/`，没有注入原冻结 Trainer。原始 CPU JSON/log 在服务器与 Mac `logs/` 保留；回传小包 3614 字节，SHA256 `af9930ae25489099eff1e256f0010dfe3b2faed4cfeb77b2a787786e1b398344`。

## 当前进度与下一步

### 4096首轮全更新完成，次轮物理图回放CUDA OOM

E4096-V01最终外层和worker均FAILED，保留原根/错误栈。一轮完整更新实际完成，五optimizer hook为10/1/20/1/20、非有限0；首轮1058.685秒包含编译，peak allocated20578716160B/reserved23441965056B。第二轮在collector的Warp物理graph回放报`CUDA error 2: out of memory`，没有完成8轮或最终Adam状态审计，不能把第一轮成功写成稳定容量通过。[原始失败](../logs/session_records/morphmimic_q1_e4096_v01_results_20261103.json)/[结果](../data/manifests/morphmimic_q1_e4096_v01_results_20261103.json)。这不是否定先前另一Q1实现的4096实测，也不是训练质量失败。

原controller602072/parent602086/worker602087均已退出，GPU15MiB；容器OOMkill0但本次CUDA OOM为真实错误，两者不可混称OOM0。1812份冻结输入在失败后独立重算大小/SHA全部不变。未恢复/扩时或复用profile权重，正式训练仍未开始，采纳null。

按用户“高env/充分显存利用且保证训练量和效率”，下一独立实验仍保留E4096、WM2048x10、motion/KP4096、H32、网络/数据/种子/五optimizer；仅把上游已支持的`compile=reduce-overhead`换成`default`，测试训练CUDA图/显存池驻留这一执行因素，不能预先当根因已隔离。新根`MORPHMIMIC_Q1_FULL1H_PROFILE_E4096_DEFAULT_V02_20261103`/8全更新轮/10800秒资源时限/原守卫，发布核验后执行；原V01失败不改。[配置](../script/morphmimic_q1/full1h_single_profile_e4096_default_v02_20261103.json)/[准备](../data/manifests/morphmimic_q1_e4096_execution_v02_preparation_20261103.json)。配置对比须精确只有cfg.compile不同，实际稳态吞吐由新实验记录，不仅看显存填充率。

完整E512双rank后六轮中位2.8405秒，等曝光320000轮的迭代耗时估计约252.5小时，尚未计编译/保存，不当正式运行承诺。这说明低规模虽然通过执行却不宜直接当效率选择；后续以实测高容量和完整训练总耗时比较，而非缩小训练量换短日志。

### 原规模4096完整更新资源测试已启动

准备提交`e0586a1f3c3faf69df1ec521e49a82e9740db86b`已push Q1并独立git ls-remote确认全40位，9文件534089B在Git/Mac/25386逐大小SHA一致。随后北京时间00:56:16唯一新profile启动，controller602072/start55222136、worker602087，原boot/GPU/唯一根；[启动现场](../data/manifests/morphmimic_q1_e4096_launch_20261103.json)。执行冻结一直为e0586a1，后续报告提交不改变它。

配置为真实559片/单Q1 E4096、WM2048x10、motion/KP imagination4096/H32、原完整网络/compile/BF16、8全更新轮。日志409600预热转移15.1秒，首轮仍编译；落盘0轮/optimizer0不是内存尚无更新的证明。现场GPU14193MiB/24GiB、容器总约14.35GB/OOM0，只是编译期间采样，不能当最终峰值。原12GiB初始/30GiB运行保护与GPU串行保持，不修改正在运行的Python/输入或重开同根。

三个刚拟定而未发布执行的E1024配置草稿已移除，旧已发布训练记录不改。接下来等待本次五optimizer、Adam状态、完整迭代耗时及allocator峰值，然后独立决定高容量共享配置；正式训练仍未开始，不以预热或资源profile称步态改善，采纳null。用户清除了应用goal，本阶段不创建或恢复目标，不改其他自动任务。

### 11月3日完整数据共享资源通过，优先恢复4096原规模

冻结`e2cae1d8c47d401ba0c691f2c47855bfc89124aa`下，完整559条同源同帧集的E512-V02于北京时间00:32:24正常完成。外层COMPLETED、guard WORKER_COMPLETED/回收0；两rank各8轮，五optimizer及Adam状态各80/8/160/8/160，非有限0，1810份冻结输入前后不变。两个rank的WM/策略/两critic完整参数SHA相同，实际共同片帧抽样相同，morphology presence与replay保持独立；原controller599450/parent599457/rank599458和599459均已退出，GPU15MiB无计算/OOM0。[结果](../data/manifests/morphmimic_q1_full1h_profile_e512_v02_results_20261103.json)/[原始收据](../logs/session_records/morphmimic_q1_full1h_profile_e512_v02_20261103.json)。旧两片V01的外层资源失败原样保留。

每rank实际61440转移，含51200预热；前两轮编译约132.9/379.7秒，后六轮约2.61--3.09秒，末轮2.612秒。两rank peak allocated约5.20GB/各reserved约5.65GB，这不是整卡同一时刻精确峰值。该结果只证明完整数据共享执行与当前规模的资源，不是正式训练或步态改善。

用户指出3090先前可以4096，并进一步要求env尽量高、显存尽量利用、保障训练量和效率。[旧完整批量实测](README_SUPERMIMIC_Q1_E4096_20261025.md)确有E4096/WM2048x10/actor4096-H32、峰reserved17760780288B=16.54GiB；此前直接默认E1024缩放证据不足。旧记录属于另一Q1实现/资源profile，不能证明当前两个v0.1 Trainer同时驻留也能各4096；同样不能仅凭上游28GB估计就否定本卡容量。

下一独立根`MORPHMIMIC_Q1_FULL1H_PROFILE_E4096_V01_20261103`按[发布配置](../script/morphmimic_q1/full1h_single_profile_e4096_v01_20261103.json)测单Q1：E4096、WM2048x10、motion/KP imagination4096/H32、原网络、BF16与compile、8全更新轮、559片、从零初始化。12GiB初始/原30GiB运行线、GPU锁/boot/GPU/OOM/温度保护保持；上限10800秒只约束资源测试，不套旧300/1200秒物理门。profile权重不续训。共享同卡容量随后单独实测，按整体吞吐选择高容量而非以低显存为目标；三个未发布执行的E1024草稿撤回默认地位。

正式训练参照E4096x40000：每机器人含100步预热共410009600转移、WM400000次更新、500轮WM-only后motion及当前return-KP各39500/critic各790000。任何缩放必须列明采样、batch和Adam步差异，不把相同曝光称完全等价；优先不缩原batch。当前接入使用return KP/jitter，新Q1控制/缺失槽/命令记忆也是改动；v0.1 README当前COMMON为copy且60000轮，论文展示的是40000轮shared，不混称逐字复刻最新版全部配方。[决策清单](../data/manifests/morphmimic_q1_high_capacity_preparation_20261103.json)。当前高容量profile尚未启动、正式训练未启/采纳null；不创建goal/agent或改变其他自动任务。

### 11月3日完整一小时559片原过滤/真实配对通过

执行来源`e5898a7597dd07ef970ae8673dcddc04a561c33d`已独立40位远端及10文件449639B三端核验。首次独立SSH查询超时保留，随后HTTPS branch ref及独立git ls-remote均确认同一提交；未改认证或Git持久配置。[发布收据](../data/manifests/morphmimic_q1_full1h_retarget_publication_20261103.json)。服务器4项资源准入CPU测试实际全部通过，随后controller598471完成原CPU转换/过滤/配对，guard正常reaped0/1248冻结输入未变、未创建CUDAcontext、0积分/optimizer。

559/559个真实Q1参考均通过原过滤，559个人体侧文件逐片相同帧数；与原G1一小时集的交集仍为559片、3604.72秒，无排除或替换。1677帧G1实模FK最大位置差6.990e-7m。1118个Q1/human NPZ共404751654B只留服务器，原G1参考未覆盖；同源表和大小SHA保留在服务器result.json，Mac只保存小收据与YAML内容。[结果清单](../data/manifests/morphmimic_q1_full1h_data_results_20261103.json)/[原始收据](../logs/session_records/morphmimic_q1_full1h_data_20261103.json)。

空闲缓存维护按原8GiB/120秒/1800秒冷却执行一次，实cache13356945408→6177824768B，总量14645428224→7885205504B，未读取大文件内容/删数据/全局drop_caches；后续空闲实测7319539712B，满足新12GiB初始上限。下一步先发布本数据结果，再新根完整集E512双rank/8轮资源验证，保持30GiB运行线；旧两片V01资源失败不改。仍未正式训练/证明自然步态或跨本体收益，采纳null。11月2日“转换待执行”等为历史。

### 完整559条真实Q1重定向完成，转换/配对待执行

前阶段结果`cfc95bba8c4d8883d10aad1a296197cf5484cd0f`已独立远端/7文件514699B三端大小SHA，见[发布收据](../data/manifests/morphmimic_q1_shared_profile_v01_publication_20261102.json)。随后原controller596724/start54764026于23:39:55启动完整559条，原Newton batch10/接触后处理23:49:30正常结束，worker596731回收0，669冻结输入前后不变。559个真实Q1 CSV共142661847B，仅服务器，原BVH未重新提取；当前仅重定向完成，不能把文件数直接当通过原过滤的共同训练片数。[实测](../data/manifests/morphmimic_q1_full1h_retarget_results_20261102.json)/[原始摘录](../logs/session_records/morphmimic_q1_full1h_retarget_20261102.json)。

两原PID已不存在，同boot/GPU根、GPU15MiB无计算/OOM0。下一步原`prepare_data.py`转换/生成human sidecar/过滤并核原G1同50Hz同帧和实模FK，不替换失败片。资源入口新增可选`--initial-admission-gib`，默认24GiB且只可收紧；新完整数据双rank profile明确12GiB初始上限，为V01实测约14.6GiB增量留余量。运行时30GiB/锁/OOM/温度等线未改，4项本地CPU边界测试通过；服务器测试待本阶段发布/镜像后执行。旧V01与全部错误日志保留，不续其权重。候选`full1h_shared_profile_e512_v02_20261102.json`仍为8轮资源检查，不是正式训练计划。

### 共享8轮实际更新通过，外层资源守卫失败保留

原V01的G1/Q1两个worker均完成8轮：每边五optimizer为80/8/160/8/160，Adam状态步数与hook一致；每边实际61440转移（含51200预热）。WM/motion及人体策略/两critic共享参数完整SHA一致、实际共同片帧抽样一致，原rank校验通过，机器人morphology/limits/状态normalizer/replay仍独立；全部非有限更新为0。前两轮含编译为1051.6/433.5秒，后6轮约2.32--2.89秒，每rank CUDA reserved峰值约5.45GB。仅两片profile，不证明步态质量或完整一小时资源可行。

两个worker的收据已COMPLETED，但外层controller在退出收尾时检查到33270964224B，超过原30GiB，按MemoryLimit正常TERM回收parent=-15，因此整次profile明确FAILED，不能将worker成功覆盖成整体通过。OOM0，131份冻结输入大小/SHA重核全部不变；controller588306/parent588313/rank588314和588315均已退出，GPU15MiB无计算。两型ckpt8/latest各342722571B仅stat，未加载或用于正式训练。原guard/错误栈/完整收据保留，见[结果](../data/manifests/morphmimic_q1_shared_profile_v01_results_20261102.json)/[原始记录](../logs/session_records/morphmimic_q1_shared_profile_v01_20261102.json)。

空闲后23:27:16按既有缓存维护默认值运行一次：17.82秒/最多8GiB advice，实测cache19503054848→12499193856B，总量20949975040→14334533632B，OOM0；未删文件、未全局drop_caches、不读取大文件内容。V01初始总量17592258560B、失败峰值比初始高15678705664B，故24GiB普通准入不足以保证双rank编译/保存余量；不是GPU显存溢出或算法已失败的证据。后续保留原30GiB线，完整数据后另做新根资源验证，启动前按这次实测增量预留额外余量，不恢复失败V01/不续其权重。

完整559条准备`06b67aadc8658858f4ab6ab26ed76c250c6c7b0c`已独立远端/7文件423365B三端大小SHA，[发布收据](../data/manifests/morphmimic_q1_full1h_preparation_publication_20261102.json)。下一步先发布本次真实结果，再串行完整重定向/原转换过滤/配对；正式训练尚未启动，采纳null。下方“profile运行中”为历史过程。

### 共享资源 profile 运行中与完整一小时数据准备

原生共享 profile 于北京时间22:57:33启动，controller588306/start54509869，执行冻结仍为`cba206093e35854ad07991eb3a321436912b6b0b`。截至23:11:14，G1和Q1均完成各51200条预热转移，正在编译首轮完整更新，收据仍为0个已完成迭代；不能将尚未刷新的optimizer字段解释为内存中没有开始更新。当前容器29506048000B、GPU9577MiB/37度，未触发30GiB守卫；尚未证明两rank共享更新完成。运行根`MORPHMIMIC_Q1_G1_SHARED_PROFILE_E512_V01_20261102`、runtime plan在`logs/morphmimic_q1_shared_profile_e512_v01_plan_20261102.json`，继续监督原进程，不修改冻结Python。

新增[完整数据准备清单](../data/manifests/morphmimic_q1_full1h_preparation_20261102.json)及`script/morphmimic_q1/full1h_batch_20261102.json`，指向已提取的559条共同BVH，不重新提取归档。后续使用同一校准V02/原Newton batch10和原接触后处理，独立新输出根；原120Hz到50Hz转换、human sidecar与原过滤后，只取与原G1一小时集同片同帧的真实交集，保留排除项。当前仅配置，完整重定向/转换/训练均未启动；需共享profile结束并保存结果、GPU释放与原资源准入通过后执行。正式训练轮次待实测，不以两片profile权重续训或据此称步态改善。

前阶段`52a45c741e2f9f5e2faa5400d720ba31a7db38f0`的6文件440060B三端核验另存[发布收据](../data/manifests/morphmimic_q1_paired_g1_native_gpu_publication_20261102.json)。

### 同片配对和 G1 原生对照完成

执行来源 `cba206093e35854ad07991eb3a321436912b6b0b` 已独立远端/9文件441368B三端大小SHA核验，服务器既有Torch环境4项测试实际全部通过。22:46:32既有NPZ配对检查结束：两片23.68秒均在原G1一小时集/50Hz逐片帧数相同，6帧真实v0.1 G1 FK最大位置差4.184e-7m，源NPZ/YAML前后身份相同；未重新转换。

22:48:22 G1原生检查结束：16环境/24控制步/96批量物理子步/384转移/0optimizer；原六actuator分组独立延迟0--4步，延迟目标/干净观测差0，环境与WM解析reward最大差2.608e-8。第21步有1个环境提前终止，重置命令记忆归零已核，不把这项固定测试动作当策略步态评测。两个guard均reaped0/冻结源不变，原boot/GPU与OOM0保持。

证据：[配对和G1实测](../data/manifests/morphmimic_q1_paired_g1_native_gpu_results_20261102.json)、[原始收据](../logs/session_records/morphmimic_q1_paired_g1_native_gpu_20261102.json)。下一步仅按已发布候选运行两rank E512/8完整更新资源profile；尚未正式训练或证明Q1步态改善，采纳null。

### 同片配对与共享 Trainer 执行准备

已完成结果 `5b59018801bff1e4413dcf3808aa6d43141e9eb0` 独立核远端，6文件435113B三端大小/SHA相同，见[发布收据](../data/manifests/morphmimic_q1_pilot_native_gpu_publication_20261102.json)。新增复用已有NPZ的配对入口，只核G1同片50Hz/逐片帧数/关节与body身份和三帧真实v0.1 FK，不重复重定向/转换或覆盖原结果。配对排除原因逐项记录；正式全量转换也使用相同检查。

新增 `train_native.py` 调用原v0.1 Trainer，在同一GPU上两个原生rank，share all/同片同帧latent/原平均梯度后clip-Adam；分别记录五optimizer hook与Adam状态次数、实际总采集转移、共享参数完整SHA、独立buffer和共同帧抽样。修复子rank失败被另一个存活rank掩盖的退出等待，并累计非有限更新；3项本地反例通过。尚未实际原生共享训练。

执行前另核出原 hazard 含标量参数，完整参数SHA入口先展平后按原始字节散列；新增数值反例在既有服务器Torch环境运行，本机无Torch时明确跳过，不把纯标准库检查当张量实测。

冻结资源候选为每机器人E512、WM256、motion/KP imagination512、H32、原完整网络、8全更新轮、原compile和BF16。这个profile从零初始化、只用于执行与资源检查，权重不用于正式训练；两固定片不足以当正式训练集或证明跨本体收益。先CPU配对与G1零更新执行，再实际两rank profile；通过后扩559共同SOMA并冻结足量训练配置。正式规模/轮次尚未决定，旧300/1200秒门不适用，24/30GiB等原保护不变。

### 两片真实参考和 Q1 原生 GPU 检查完成

固定入口 `494ff6e2db6641304f65371e854a2afb551820a4` 的 V02 Newton pilot 于北京时间 22:03:55 正常结束；原接触和脚部后处理已执行。两片前进/起走参考均通过原 120 Hz 到 50 Hz 转换和原过滤，共 23.68 秒，两个 human sidecar 与机器人逐片帧数一致。这里的 paired_count=2 尚未核 G1 对应物，不代表 G1/Q1 成对数据已完成。原 V01 构造失败保留。

22:08:05 Q1 原生 GPU 检查结束：16 环境、24 控制步、240 批量物理子步、384 环境转移、0 optimizer。实际软件 PD/转速限矩回调 241 次，最大公式差 2.027e-6 Nm；延迟目标和干净观测最大差 0，环境奖励与 WM 解析奖励最大差 2.236e-8。三个 guard 均 WORKER_COMPLETED/reaped0，冻结来源前后相同；本轮现场复核同 boot/GPU/唯一根、GPU15MiB无计算/OOM0。

证据：[结果清单](../data/manifests/morphmimic_q1_pilot_native_gpu_results_20261102.json)、[原始结果与守卫摘录](../logs/session_records/morphmimic_q1_pilot_native_gpu_20261102.json)。CSV、NPZ、BVH仅留服务器。上述仅证明真实参考生成和控制/模型接口执行，不是策略步态改善；G1原生对照、真实同源同帧数据、两rank共享Trainer及正式训练仍待，采纳null。

### Pilot V01 初始化失败与导入来源约束

准备 `bc697df3d48eecdf9623d9ef49cb22f5dcee08c7` 独立远端及11文件426708B三端核验后，V01原batch在构造pipeline时返回 `could not convert string to float: 'size'`，guard正常回收退出255，未生成CSV/未训练。原根 `MORPHMIMIC_Q1_SOMA_PILOT_GUARD_V01_20261102` 保留。该次启动的commit元数据手工抄成 `bc697dfc568db9236b97e6e32962db73f31c5f77`，是记录错误；实际部署文件已按正确已发布commit逐SHA核验，补正不覆盖原记录、不重算作训练。

对同一固定源码和配置的独立CPU构造及有守卫CUDA构造均通过。普通retarget环境会从site-packages导入SOMA，因此新入口强制固定源码/PYTHONPATH、核实实际模块路径，并只给原constructor添加异常栈。此时尚未隔离初始化失败的单因；V02使用独立输出/guard根，不改原源文件或重用失败产物。

### SOMA 校准 V02 完成与真实 pilot 准备

路径修复 `42c1bf4797637592cf6b16a6c4259509a46d31f5` 已独立远端与 6 文件 410156 字节三端大小/SHA 核验。新 V02 CPU 校准成功，15 映射点偏移回算通过，真实 free+22 模型为 29 coordinates/28 DOFs，pelvis 比例为 0.4148515761（此值是 Q1/零姿态人身高比例，不是 Q1/G1 腿长比）。原 XML SHA 未变，原 V01 加载失败与日志保留。结果见 [校准原始结果](../data/manifests/morphmimic_q1_soma_calibration_results_20261102.json)。

新增固定两片 `pilot_batch_20261102.json`，调用固定 SOMA 原 Newton 重定向，输出真实 Q1 CSV；新增 `prepare_data.py` 直接调用原 120Hz→50Hz converter/原过滤与原 human sidecar converter，核有限22关节/逐片帧数；成对阶段仅取原 G1 子集与通过原过滤的 Q1 交集，逐项保存排除原因，不把 G1 NPZ 改名当 Q1 几何。

`verify_native_gpu.py` 准备16环境、24控制步零 optimizer 验证：实际 Q1 actuator callback 的 PD/方向转速限矩、每子步延迟目标、上一实际命令递推、缺失槽 NaN 保护、仿真后态与 WM 解析 reward/干净观察一致。G1 同入口核原分组独立延迟。当前仅语法检查，pilot/原生 GPU 尚未执行，不能称步态改善或完整 Trainer 已通过。19:35以后再次现场核同 boot/GPU、15MiB/26度、cgroup18552647680B/OOM0后准备串行执行；训练轮数仍待实际两rank资源。

### SOMA 校准路径修复

准备 `0b5623cd489c9d33bb14592787f0d702bf042c29` 已独立核远端，12 文件 450753 字节三端大小/SHA 相同。首次 CPU 校准在 Newton 加载 mesh 前失败：相对 XML 路径派生出的 `meshdir` 被再次相对新输出目录解析。原 V01 根和 `logs/morphmimic_q1_soma_calibration_v01_20261102.log` 保留，未产生校准或动力学/optimizer。

入口现在先绝对化 source/XML/output；新 V02 独立根 `data/supermimic_for_q1/MORPHMIMIC_Q1_SOMA_CALIBRATION_V02_20261102` 待校准，不覆盖失败根。仍使用固定原 configurator 函数、真实 Q1 模型和默认 IK。校准后执行两固定 BVH 原 Newton pilot，再转换/真实环境；尚未正式训练。

### 真实原生 CPU V02 通过

`b343fefc0a887a7310da61b3e1d99b7fcb8c89c9` 独立远端及 7 文件 414037 字节三端核验后，13 组检查全部通过，见 [CPU 原始结果](../data/manifests/morphmimic_q1_native_cpu_results_20261102.json)。实际 Q1/G1 大腿长 0.211656/0.340923 m、小腿长 0.170652/0.300010 m、质量 16.917288/33.341142 kg；真实形态 154 维不再使用人工几何。Q1 腿长比例 0.596486，环境位置核宽 0.178946 m、线速度核宽 0.596486 m/s、高度终止 0.149122 m，与世界模型配置一致。命令记忆、多步递推、缺失动作梯度、NaN 缺失槽拟合、replay、policy/critic、2ms/10 子步及禁重复延迟通过；仍未原生 GPU Trainer 或实际 actuator 回调验证。

新增 `prepare_soma.py` 以真实 Q1 V3 创建外部 SOMA manifest，使用原 configurator 的偏移/回算与接触后处理函数，按原默认 IK 权重从 Q1 零姿态校准；不借 X2 尺寸、偏移或 elbow 参考。校准先在 CPU 运行，再两固定片原 Newton pilot、50Hz 转换与原生环境检查，合格后扩共同数据。`run_native.py` 沿用 GPU 锁、24/30GiB、OOM/boot/GPU/温度/磁盘保护及源码前后 SHA；profile/train 的 optimizer 数以 worker 实际收据为准，不能沿用旧零更新标签。

G1 延迟改为每个原 actuator group 独立 0--4 步，Q1 维持每环境共享连续秒数后量化到 2ms，默认历史差异保持明示。这一 action 实例化仍待原生物理检查；未因 CPU 通过声称步态改善或正式训练已开始。

### 原生适配准备 V01

准备 `da3bf3e738f85383a60541dc9c8d24eb2f6a6994` 已独立核远端，10 文件 428221 字节三端大小/SHA 相同。首次真实 CPU 已通过实际资产/FK、形态、replay/策略/critic 前置检查，随后在拟合损失因包自动注册任务提前捕获旧 `split_state` 别名失败；原日志保留为 `logs/morphmimic_q1_native_cpu_v01_20261102.log`。未发生动力学积分或 optimizer。V02 显式重绑定已加载模块中身份完全匹配的 registry/split/join 别名，并记录逐项名单；使用新结果路径重新检查，不覆盖 V01。

用户进一步明确沿此思路继续。新增 `assets.py`、`native_action.py`、`model_adapter.py`、`bootstrap.py` 与 `verify_native_cpu.py`，通过进程内显式绑定接入固定 v0.1；原冻结源码文件不改。当前仅完成实现与本地语法检查，真实描述/原生模块 CPU 验证还未执行，不宣称 Trainer 或物理训练已通过。

选定状态 269 = 原物理 240 + 上一拍实际 canonical 命令 29；context 62 = 原 33 + 每槽实际量化延迟秒数 29，缺失槽为 0。世界模型每步将执行命令精确写入下一状态，critic 使用 297 = 原 268 特征 + 命令记忆，避免多步预测沿用过期的上一目标。记忆不是学习预测，也不加入物理拟合误差。Q1 22 关节的动作/探索执行、状态输出、拟合误差、RMSE 分母、策略 decoder 及观测噪声均需投影；新 CPU 入口含 NaN 缺失槽反例。

Q1 entity 从 V3 XML 的原几何/惯量/限位与 armature 构建，移除场景 floor、旧传感器和旧 motor，由 mjlab 注册软件 PD/方向转速曲线 motor；足 mesh 和四个点保持碰撞几何，并命名用于 friction 随机化与参考过滤。尚待实际编译验证这一构建与原 FK 一致。真实 morphology 的候选固定边界来自 G1+Q1 两型，各自实际 PD/动作尺度/峰值力矩与原 9 项几何/质量；Q1 单体对照也使用这同一边界，不能各自重新缩放。

延迟统一在 action 的物理子步应用，关闭底层重复延迟。Q1 连续 0--20 ms 后 round 到 2 ms；G1 保留 5 ms 离散 0--4 步采样和原 PD，统一在 episode 开始以上一命令 0/默认姿态作为历史。G1 原 DelayBuffer 首次命令回填当前目标，新接口改为显式默认历史，这是新增的启动语义差异，须写入 baseline/shared 两组配置，不称原版逐轨迹等价。训练仅接受 episode-fixed 延迟且不超过一控制周期。

位置/线速度 tracking 核宽和高度终止按腿长比例同步到环境及世界模型，保留原权重与角度核宽；需 CPU 检查实际 reward/termination 配置，不能只看函数实现。新契约与旧 G1 checkpoint、发布 ONNX 均不兼容，从零初始化。当前没有新增动力学、策略仿真或 optimizer；下一步先发布/镜像后运行真实 CPU 契约，再做 Q1 SOMA 校准及串行 GPU 原生预检。

独立缺失关节模块与 CPU 契约验证已完成；进程内 Q1 注册已由 CPU 验证，原生 GPU Trainer、重定向、物理 rollout 和训练仍待。完整结果写入 `data/manifests/morphmimic_q1_integration_20261102.json`。当前资产 XML 自带 1 ms timestep，适配显式设置 2 ms/10 子步；原 XML 不修改。

延迟还涉及世界模型状态契约：Q1 源控制会在子步中使用上一时刻位置目标，原 MorphMimic 的 33 维 context 只含摩擦、encoder bias、CoM offset。必须明确延迟值和上一目标在模型中的可见性及一步更新规则，不能只把 PD 换掉就宣称完整接入。奖励/终止按尺寸缩放和 22 关节误差权重也须在仿真与 imagination 两侧一致；CPU bridge 中的真实关节 RMSE 不代表训练损失已经替换。

按顺序完成：缺失槽/梯度与真实模型 FK/控制契约验证；Q1 专用 SOMA 配置和成对轨迹；原生 Trainer 两 rank 同 GPU 的通信、资源及五 optimizer 更新验证；冻结实际训练配置并发布核验，再启动足量训练及固定原速视频评估。环境规模和迭代数由真实共享资源测试确定，当前不预填已通过数值。沿用 24/30 GiB、boot/GPU、OOM、GPU 锁和温度守卫；不套用旧 300 秒退步门或 1200 秒短预算。

用户采纳仍为 `null`；本阶段没有 Q1 步态改善结论。
