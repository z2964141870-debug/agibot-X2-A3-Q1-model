# Q1冻结世界模型的BF16与FP32独立对照

2026-11-07北京时间19:18，独立精度对照已实际完成：`WM_PRECISION_COMPLETED`、diagnostic_screen/source_integrity均PASSED。19:21完成退出、来源、资源和成本复核。原共享训练继续保持用户暂停，原生物理梯度FAILED与采纳null不变。

## 用户版done与todo

done：相同权重、256条保存转移、四个锚点和原阈值下，FP32物理梯度有限差分88/88通过，原BF16为4/88。原生预测及四个VJP逐值复现，原生FAILED保留；预测校准和逐source质量保留门均通过。

done：376次WM.step、8次标量反向、0新物理/渲染/Adam，持锁34秒已结算，累计1463/14400秒。原值逐文件大小SHA核同，新的诊断六宫PNG/PDF及两份CSV已生成并目视。

todo：已登记固定FP32世界模型的独立物理特征蒸馏设计；先审计已完成action-only控制的精确设置、同批次曝光和初始化，再实现发布及独立准入。学生仍只看f0/history，该设计尚未开始训练。

## 真实完成结果

控制器358354/start36318263与worker358426均已退出，guard为WORKER_COMPLETED/return0且worker_reaped；178执行/200冻结/754保护输入退出前后同。缺失七关节梯度与扰动物理响应仍严格为零，四组存在关节均22/22通过。88表示每关节至少一种预登记epsilon通过，不表示全部176个epsilon逐项通过，也不表示完整物理Jacobian。

| validation状态来源，各六片等权 | BF16关节误差rad | FP32关节误差rad | BF16身体误差mm | FP32身体误差mm |
| --- | --- | --- | --- | --- |
| 原教师状态 | 0.002325322 | 0.002254676 | 0.448462 | 0.443072 |
| 教师学生混合状态 | 0.002896251 | 0.002843029 | 0.545368 | 0.543459 |

关节误差分别下降3.038%/1.838%，身体误差下降1.202%/0.350%。完整逐source、六项物理指标和每个epsilon保留，不能用这些均值替代闭环质量。独立FP32影子仅关闭本地BF16 autocast，TF32保持关闭，全部权重/normalizer不变；该对照支持本批样本的数值梯度实验，未证明BF16造成此前f0倒地，也未认证原训练动力学、自然遥操或硬件延迟。

退出复核：峰cgroup10615828480B/GPU460MiB，末总8024719360B/限34292629504B，GPU1MiB26度/OOM0/计算进程空，原锁可获取后释放，旧五训练PID仍退出。成本本次34秒、累计1463/14400、r3已342/7200秒，全部SETTLED，无重置。

[真实派发](../logs/session_records/q1_wm_precision_dispatch_20261107.json)、[控制器结果](../logs/session_records/q1_wm_precision01_result_20261107.json)、[完整诊断703660B](../logs/session_records/q1_wm_precision01_precision_result_20261107.json)、[守卫](../logs/session_records/q1_wm_precision01_guard_20261107.json)和[退出实核](../logs/session_records/q1_wm_precision_completed_20261107.json)已原字节传回Mac。结果包112395B/SHA `206898aa49f5e68e787b499c82293a43fb5f69fe806bc31d5e98c5259f20e5cb`全部成员核同，不重传或重做。新图与压缩包位置见本阶段figures_delivered收据。

新证据ZIP为`data/deliveries/Q1_WM_PRECISION01_EVIDENCE_20261107.zip`，311946B/SHA `c902f6c707f23db424bed7f98039fca81e423a39498bbd07eeafbe25e7f0471b`，16成员CRC/逐件大小SHA及14相对链接通过。包内含简版README、六宫PNG/PDF、两CSV、原生与FP32完整原值、守卫/派发/成本及两预登记方案；本次没有新增训练曲线或视频。[交付收据](../logs/session_records/q1_wm_precision01_figures_delivered_20261107.json)。

## 假设与公平比较

BF16 autocast的量化可能使小扰动中心差分为零而自动微分非零。这是项目假设，当前没有因果证明。新计划在任何FP32调用之前登记，见[精确JSON](../script/supermimic_teleop/wm_precision_probe_plan_20261107.json)。

原生模型严格载入32k原始metadata/normalizer/morph/presence/height/frame/residual及所有buffers。独立`deepcopy`只用`dataclasses.replace(cfg,bf16=False)`更改本地影子开关，所有张量SHA相同，两个模型均冻结，TF32关闭。既有模型、策略、配置文件、采集动力学及保存频率不改。

原PROBE02的已结算result/guard/authorization及全部保护来源必须先核同；重建相同原教师/教师学生混合train/validation的256行，顺序、clip/frame/dataset-row、实际执行动作和终止mask与原诊断一致，不读取test。命令末29维仍排除在物理特征之外。

两模型各进行四批保存预测，各256行。原生仅重算四锚点物理VJP并要求逐关节精确复现，原生差分直接复用原结果，不重复。FP32影子进行四锚点VJP及相同22存在关节的epsilon0.01/0.001中心差分，缺失七关节保持严格零导数和物理扰动响应。预计376次WM.step、8次标量反向；不是完整Jacobian，也不是CUDA primitive kernel数。

## 预登记门

差分门沿用绝对0.002加相对5%，每个关节至少一种epsilon通过，四锚点全部通过；各存在关节在某锚点有大于1e-7的物理导数。保持原生VJP逐值精确复现及保存预测逐值一致门，不把原失败改写为通过。

影子仍须通过原validation calibration：两origin各source先平均再等权，关节与身体位置WM/identity均值不超过0.95、任一source不超过1.25。额外要求相对native预测保留：每validation origin均值不超过1.10倍native误差加1e-6，每个source不超过1.25倍加1e-6；完整逐source和六项物理误差保留，不能以总均值掩盖单source退步。

全部通过才记`WM_PRECISION_COMPLETED/diagnostic_screen=PASSED`；科学筛查失败仍可正常完成并结算成本，执行错误独立记录。对照成功只证明本批保存数据下的数值梯度可用，仍不能认证原训练动力学、长滚动/闭环质量或自然遥操。

## 实现与边界

独立`wm_precision_probe.py`控制器、`wm_precision_worker.py`及`wm_precision_math.py`，复用原严格WM载入、数据重建、原物理指标、guard和持久成本账本。原175执行/200冻结源码不改。新聚焦检查覆盖影子隔离、拒绝可训练模型、单次VJP复现、禁止均值掩盖source回退和禁止缺失/重复source；实际CUDA调用及结果见上方完成记录。

唯一新输出：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_WM_PRECISION01_20261107`。原PROBE01/02输出及失败证据不改；派发前核output/log/dispatch/receipt及实际进程，观察失联不能作为重派依据。

总300秒/64MiB，最多512次WM.step/8次反向/256条保存转移，新增物理/渲染/优化均0。22GiB准入/24GiB退出、GPU18GiB80度、100GiB磁盘余量、private500GB、原非阻塞GPU锁/自有子树回收和原暂停检查保持。复用R2缓存边界，不重复advice/drop或删除证据。成本从累计1429/14400秒、r3已308/7200秒继续，不能换新根重置。

上一步结果发布`e96479de93646a2fbd4b8e626ab33a875ba6d186`已显式push、独立40位远端与Mac十三文件1164479B核同；[原结果](README_Q1_PHYSICAL_WM_PROBE_20261107.md#真实诊断结果)与[证据交付](../logs/session_records/q1_physical_wm_probe02_figures_delivered_20261107.json)保留。新阶段发布、部署、原运行时、准入与实际结果分别以随后收据为准。现有33391由Agent自主复用，断开后按用户固定source/ssh入口自行尝试，不再让用户先测试。

## 下一步

精度与预测保留已同时过门，下一[预登记设计](../script/supermimic_teleop/wm_physical_distill_plan_20261107.json)只加0.1倍同状态、同当前坐标系的教师物理特征蒸馏，排除末29维命令缓存及缺失feature。教师目标为同状态下教师动作的冻结WM预测，不错用其他行为动作的日志下一状态。冻结teacher/decoder/WM和原R2统计；不增加未来信息或新辅助头。

必须先精确审计原PAIR02 action-only控制的初始化、设置、每验证点batch SHA和source曝光，不一致就阻止派发并修订发布计划。最多3000更新/1740秒拟合/1800秒保护/1GiB，选择仍只用原validation动作风险且包含update0。若选中策略与R2精确相同，直接复用0/7证据，不重复失败物理；实际实现及训练尚未完成。不把这项独立假设称为GigaBrain损失的忠实复现。

## 真实部署与准入

独立源码`bacf186397c7f21aa35f0817ccd8aa1aa9a09830`与178执行清单`96539f226772229bce5612f09c243aa351d18b2a`已显式push/独立40位远端核同，Mac十一文件806739B大小SHA全同。六成员代码包20198B/SHA `685b601dc9265a77729536cc249eeb74b515b7bf8879104f963d4bc27fd93511`经33391原会话部署；旧748份来源前后核同，只新增清单外代码与已登记计划，没有改写原175/200来源。

北京时间19:15，原Morph运行时五项新CPU检查全过，测试本体0.079秒/进程2.431331秒；没有加载WM或初始化CUDA。实际独立准入重新通过178执行/200冻结/754保护输入，临时独占检查原锁后释放；总8111087616B/限34292629504B、GPU1MiB26度/计算空/OOM0，原暂停及五PID退出重新确认。累计1429秒/r3已308，全SETTLED，没有重置或提前预留新训练。

[部署](../logs/session_records/q1_wm_precision_deployed_20261107.json)、[原检查](../logs/session_records/q1_wm_precision_native_checks_20261107.json)、[完整准入参数](../logs/session_records/q1_wm_precision_admitted_20261107.json)三原文件6030B与2333B包已Mac逐字节核同，包SHA `6afa9beada915f9e461140b6a984f60f382d9936b94dc667391227892ffb9886`。派发/输出/日志在此准入时均未创建，发布本准入后单次派发PRECISION01；真实新CUDA仍0，服务器新说明镜像待。
