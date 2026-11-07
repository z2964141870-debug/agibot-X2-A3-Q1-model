# Q1冻结世界模型的BF16与FP32独立对照

2026-11-07。实现准备，尚未部署或执行真实精度对照。原共享训练继续保持用户暂停，原生物理梯度FAILED与采纳null不变。

## 用户版done与todo

done：上一步实际世界模型预测校准通过，但BF16物理梯度有限差分仅4/88项通过。原结果、六宫PNG/PDF和两份CSV已归档，没有直接开始依赖该梯度的新训练。

todo：在相同权重、相同256条保存数据和四个锚点上，仅将独立影子的autocast改为FP32，检查梯度和预测质量；原生BF16结果始终保留。先代码发布、服务器聚焦检查和来源/资源准入，再唯一有界执行。

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

独立`wm_precision_probe.py`控制器、`wm_precision_worker.py`及`wm_precision_math.py`，复用原严格WM载入、数据重建、原物理指标、guard和持久成本账本。原175执行/200冻结源码不改。新聚焦检查覆盖影子隔离、拒绝可训练模型、单次VJP复现、禁止均值掩盖source回退和禁止缺失/重复source；真实CUDA尚为0。

唯一新输出：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_WM_PRECISION01_20261107`。原PROBE01/02输出及失败证据不改；派发前核output/log/dispatch/receipt及实际进程，观察失联不能作为重派依据。

总300秒/64MiB，最多512次WM.step/8次反向/256条保存转移，新增物理/渲染/优化均0。22GiB准入/24GiB退出、GPU18GiB80度、100GiB磁盘余量、private500GB、原非阻塞GPU锁/自有子树回收和原暂停检查保持。复用R2缓存边界，不重复advice/drop或删除证据。成本从累计1429/14400秒、r3已308/7200秒继续，不能换新根重置。

上一步结果发布`e96479de93646a2fbd4b8e626ab33a875ba6d186`已显式push、独立40位远端与Mac十三文件1164479B核同；[原结果](README_Q1_PHYSICAL_WM_PROBE_20261107.md#真实诊断结果)与[证据交付](../logs/session_records/q1_physical_wm_probe02_figures_delivered_20261107.json)保留。新阶段发布、部署、原运行时、准入与实际结果分别以随后收据为准。现有33391由Agent自主复用，断开后按用户固定source/ssh入口自行尝试，不再让用户先测试。

## 下一步

先执行这项小样本精度对照。若数值梯度与预测保留同时过门，再登记固定教师/decoder/WM下单项物理误差训练，与同曝光action-only比较；若未过，保留反证并改用不依赖错误物理梯度的适配方向。不得从本计划自动开始训练，也不重复已完成的84事件、七条同策略失败物理或同数据latent拟合。
