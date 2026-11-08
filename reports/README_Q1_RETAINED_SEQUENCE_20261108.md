# Q1旧能力保持与动作序列对照

2026-11-08，北京时间。独立实验`F0_RETAINED_SEQUENCE_PAIR01`，不恢复原共享160k训练。[序列边界审计](README_Q1_SAVED_SEQUENCE_AUDIT_20261108.md)已完成，旧采集、教师短分叉和物理评估不重做。

done：两组各3000新增Adam真实完成，source PASSED／守卫回收0／成本191秒已结算；两组都选新增update0，CPU确认与此前已训练的causal_history3000策略精确相同。两训练六宫PNG／PDF／CSV及513225B证据ZIP已核验并目视。详细结论见[结果](README_Q1_RETAINED_SEQUENCE_RESULT_20261108.md)。

todo：下一先以训练源检查旧／纯学生标签及目标梯度冲突，形成不同机制的独立单项方案，再决定是否新训练；不延长本配方或重跑相同0/7策略。真实人体流、未见事件、完整软件链与自然性仍未完成。

## 假设与唯一主变量

纯学生标签配对上一轮降低新状态MSE，却使旧三验证回退21%至40%，全部非零选点被拒。教师短段能在部分偏离状态纠正，但并非总能恢复；因此不继续加长相同逐状态MSE。本次两组都加入相同的旧输出保持项，唯一差别是匹配教师40ms动作增量。

两组均从实际`causal_history/selected3000`非零适配器继续，仍为930本体历史→128ELU→latent四张量；人体编码器、共享decoder、WM、教师和normalizer冻结。训练点目标均为同样的旧／纯学生标签各半；同组同来源抽四标签窗口，保持相同初始化、来源slot、索引摘要、更新数与曝光。没有同时更改网络或输入条件。

- `retained_point`：原点动作MSE＋20×旧train状态上与初始学生输出的MSE。
- `retained_delta`：相同目标，再＋0.5×相邻40ms的预测动作增量与教师增量MSE。

保持项是软正则，不保证旧能力不退；真正选点仍硬要求原三项validation风险均不超过初始值，且包含update0。增量损失匹配教师动作变化，不是把动作压平，也不是声称已移植GigaBrain奖励。40ms固定采样下该项与速度差平方只差常数，但本实验报告规范化动作增量单位，不当作物理关节速度。

## 数据、选择和因果边界

严格复用审计的旧train6581窗口／pure387，validation2304／209窗口；每窗f、f+2、f+4、f+6，跨度120ms。两条终止动作原行保留，仅不纳入有效窗口。来源／组train-validation分离，不读test、新留出事件或按A033等验证失败挑训练帧。窗口重叠，不当独立样本。

每时刻只输入该时刻的f0人体十步历史与930本体历史；teacher未来只作标签，next_state不加载，terminal只作mask。旧策略输出缓存仅在旧train状态生成。原始人体离线预处理仍`offline_unverified`，不因此新增实时因果性保证。

验证选择完全沿用上一配对：匹配来源旧状态与pure状态风险50:50，原全旧／R1／mixed三风险均不退，含0。验证pure仍使用原219行，绝不通过剔除验证坏行抬高结果；序列窗口仅影响共同训练批。两组均3000新增Adam，每批16旧窗＋16pure窗＝128点目标，另64点旧输出保持；种子2026110806、lr1e-4、梯度裁剪1、每500更新保存一次指标。

四次前向来自保存真实状态，是open-loop拟合；没有模型内递推或真实物理闭环。旧WM一步FP32梯度检查不能替代多步可靠性，所以本轮不使用未验证多步WM优化。训练loss、序列loss和验证risk均不能代表自然遥操合格。

## 资源与发布

独立根`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_F0_RETAINED_SEQUENCE_PAIR01_20261108`，最多600秒/128MiB，原非阻塞GPU锁、22GiB准入/24GiB退出、显存18GiB/80度、余盘100GiB/private500GB及自有子树回收保持。原账本4599=1614(r1)+1614(r2)+1371(r3)，无预留、r2余2886；不重置四小时或阶段预算。

来源包括原1172保护输入、已成序列summary及其实际控制／数组、六新增执行文件、manifest和固定base plan；均执行前后核同。先原运行时六合同、实际序列统计和资源成本准入，再发布准入回执，最后唯一output/log/receipt/精确进程防重派发。reserve→守卫确认worker回收→settle，原共享及冻结来源不改。

训练后必须交付两组六宫图／PDF／CSV和实际成本。新权重先CPU完整性与选点审计，未改变策略直接复用原失败；非零候选才独立逐片闭环和GPU原速媒体。保留完整率、离地换脚、承重滑移、侧倾、上肢时序和突变，不把软保持项或旧0/7称为成功。

本目标仍包含新留出事件、可获得真实人体流、完整软件链与成本质量结论，尚未达到；用户采纳null。主要来源和预算继承[重新定向研究设计](research/2026-11-07/LOW_LATENCY_SUPERMIMIC_REBASE.md)，本次序列增量项为独立候选，不新增论文归因。

## 原运行时与真实准入

执行源码`c3a826e3d6457d9ed455f165869fa60b8b3e7218`保留在合并提交`eb66f43849f0a3ff8dd665d9d6deeb038eb45ea0`中；并行FRDV研究记录完整保留。清单发布`a66c91bed6a571e75f77d258d57fb231e69d6db8`已显式push origin Q1、独立40位远端核同。Mac十二文件593026B大小SHA一致。

16605B十成员代码包经过原Linux会话45014的14个精确ACK新增部署，原1172保护来源前后同。六项原Morph运行时CPU检查通过，进程4.584067秒；1216保护输入、原暂停／GPU锁、4599已结算账本与资源准入11.785079秒通过。实际服务器新CUDA前向／训练／物理／渲染均0；CPU检查内小型合成Adam不作真实训练。

准入总8255746048B／实际限34292629504B、OOM0、GPU1MiB25度、余盘183413096448B。三份原始回执5344B经2181B包已Mac大小SHA核同；包SHA `4a80ac4218dc996ef549a1ea555ee0f5f1394e658c6778a04f220c6f5b94ede1`。代码包SHA `268e1b10a2346a49f4822341cab780db1e261bbb03e37666195ae21a7b3bffd1`。固定入口及已获批SSH三值90／60／5已自主实核，现有会话可用，不请求用户代测。

原件：[部署](../logs/session_records/q1_retained_sequence_deployed_20261108.json)／[六检查](../logs/session_records/q1_retained_sequence_native_checks_20261108.json)／[准入](../logs/session_records/q1_retained_sequence_admitted_20261108.json)。本段发布以后才唯一防重单派；后续实际dispatch和完成状态以独立回执为准。新大索引服务器镜像仍PENDING，原共享不恢复。

## 真实训练与选点

准入发布`f6b02230c1837c102af9b5a85e939fe575756f5f`经独立40位远端及Mac六文件556576B核同后，唯一防重门通过。timeout390213／controller390214/start41469547单次启动；两组各3000Adam，四计数均3000，各384000点目标，初始化／容量／来源抽样摘要完全相同。全任务191秒已SETTLED，guard正常回收0、1216保护来源同。

| 指标，末轮相对本轮起点 | retained_point | retained_delta |
| --- | ---: | ---: |
| 综合验证动作MSE | -9.176% | -9.046% |
| 纯学生状态验证MSE | -12.728% | -12.519% |
| 全旧状态验证MSE | +6.438% | +6.409% |
| 固定R1验证MSE | +7.249% | +7.078% |
| 固定mixed验证MSE | +4.854% | +4.776% |
| 合格非零保存点 | 0 | 0 |
| 最终选中新增更新 | 0 | 0 |

所有非零保存点均未同时保住原三验证风险。软保持20倍没有保证旧能力不退；增加动作增量项在本次配对中没有产生可采纳收益。只能否定本次具体设置，不能推广为所有序列方法无效。相较上轮回退幅度变小，但本轮窗口采样也不同，不把跨轮差异单独归因于保持项。

这里的update0是本次继续拟合的起点，即以前已训练3000更新的f0本体历史适配器；不是重新初始化教师或原共享模型。冻结教师仍为32k。四份selected／last保存均CPU有限；两组选中策略的8个base与4个adapter张量、statistics及teacher／decoder精确同初始，CPU审计2.545536秒。直接复用该策略原0/7完整结果；新物理、视频与训练后前向均0，不再渲染相同失败。

原始31743B十五成员数值包SHA `2b57573f4d5251dcbe867bfe1ec5c055d03a61e64db16a3ad2d1dae6a072b85d`已Mac核同。新只读交付入口`deliver_retained_sequence.py`仅处理这些保存原件，不在本轮六训练源码中；两PNG均已目视，无文本重叠。ZIP513225B／23成员／22相对链接通过CRC和逐文件大小SHA；SHA `2780037612da8eac57cc5b67a214bc859025a597aa93c2aca8e7d542c3abd26e`。每500更新采样曲线，不虚构逐更新loss或episode return。

原件：[派发](../logs/session_records/q1_retained_sequence_dispatch_20261108.json)／[CPU及回收完成](../logs/session_records/q1_retained_sequence_completed_20261108.json)／[交付清单](../data/manifests/q1_retained_sequence_evidence_20261108.json)。Mac工件位于`data/analysis/Q1_RETAINED_SEQUENCE_FIGURES_20261108`及`data/deliveries/Q1_RETAINED_SEQUENCE_EVIDENCE_20261108.zip`，模型与数组仍仅服务器。

累计成本4790=1614(r1)+1805(r2)+1371(r3)，无RESERVED，总余9610秒／r2余2695秒，原四小时和阶段预算未重置。峰总10561945600B／GPU544MiB／OOM0，末总8541753344B／限34292629504B、GPU1MiB27度、余盘183398744064B。timeout／controller／worker及原五训练PID均已退出，原GPU锁和计算为空，无缓存回收。原共享暂停／原final门／采纳null保持，不能把此阶段结算当自然遥操目标完成。
