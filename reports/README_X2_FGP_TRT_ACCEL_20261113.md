# X2 FGP CANO TensorRT 加速

记录日期：2026-11-13。阶段：R1至R7本轮预算均结清；当前准确候选中位29.086ms，仍8/32超33.33ms。原始S1权重身份已核，稳定30Hz/衣服与Morph并跑尚未通过。

## 目标与边界

用户要求继续检查 FGP 加速可行性，并说明第二、第三终端工作已完成。本轮只处理第一终端的人体推理；不修改已完成后端，不启动 BLE/新标定/反馈订阅/电机/MC，不恢复 Q1 训练。`publish_allowed=False`、`hardware_commands=0`。

上一阶段 ARM64 完整链已通但 S1 中位 129.324 ms，CANO 稳态中位 112.741 ms。选用作者原样 `onnx_s1/cano.onnx`，只替换 CANO 的执行器为 TensorRT；LIP、GP、物理、标定、30 帧历史、单步 Euler、EMA 与失联处理保留。原作者源码、模型和 `script/x2_fgp_realtime` 基线入口均不编辑。独立适配器只在本次进程中替换 flow model 工厂，并在 finally 恢复。

## 已核设备条件

Mac 原生有线 SSH 直达 SoC1 `10.0.1.41`；boot `8f960fd3-d5da-4630-b5de-4d61d7251901`。设备已安装 TensorRT 10.7.0.23/CUDA 12.6.68，原 `fgp-preview-py310` 环境通过本进程追加 OS TensorRT 包路径能导入 10.7.0，不更换 numpy/torch 或安装新 GPU runtime。

16:44 的实际可用内存约 13.05 GB/总 16.42 GB，swap 使用 0，GPU 约 63 C，CPU/Tj 约 65 C。`trtexec --help` 实读了 `--noTF32`、`--fp16`、workspace、builderOptimizationLevel、maxAuxStreams 及 skipInference 选项。它们是已装版本的能力，不是尚未运行的加速结果。

## 有界实验与门

1. 原样 ONNX 分别构建 FP32（关闭 TF32）和允许 FP16 的两个独立 engine。每次新输出目录，最多 600 秒，workspace 256 MiB，优化级别 2，辅助流 0。workspace 不是全部内存上限；监视真实 MemAvailable，低于 4 GiB或温度达到 80 C 时只终止本次自有子进程组。至少 10 GiB余盘，项目专用非阻塞锁只协调本项目加速任务，不冒称全局 GPU 锁。
2. 每套沿用现有 61 帧夹具，29 帧历史/32 输出，最多 32 次 GPU CANO；CPU LIP/GP/物理原样执行。计时含 CPU→GPU→CPU 同步传输和保存 CANO 输入的开销，初始化单列。外层 120 秒保护。
3. 完成计时后，才对同样的 32 个实际 CANO 输入各跑一次 CPU 参考，避免 CPU 比较污染 GPU 时间。全链与已保存同夹具 `full_s1_t4_r1` 输出比较，不重跑整个旧 CPU 物理链。
4. 数值筛查限值预先固定：FP32 residual 最大绝对差 `2e-4`、相对 L2 `1e-3`、全链最大关节角差 0.5 度、根平移最大差 5 mm；FP16 分别为 `5e-3`、0.02、1 度、10 mm。所有值有限且 32 帧完整；超阈值保留失败，不据速度选择。限值是工程一致性筛查，不是真人精度或硬件安全认证。
5. 分列全部输出与去前两次中位/p95/最大、超 33.33 ms 帧数，以及 CANO/pose/物理。30 Hz 只能按整链判断；30 帧历史是启动历史，不是持续未来等待。无 BLE/网页/网络/机器人延迟，不能冒充端到端。

本轮上限为两个 engine、64 次 GPU CANO、64 次 CPU CANO 对照，0 训练更新。失败的新根保留；不扩大预算补跑，不进行 INT8/稀疏权重改写、全局 cache 清理或系统功耗/频率变更。若构建不支持算子，先记录具体 parser 错误，不静默跳层或更改模型。

## 自有代码与验证

新增 `script/x2_fgp_accel/` 五文件：有界构建、静态 TensorRT/CUDA session、完整链 wrapper、六项 CPU 合同与包入口。使用已安装 TensorRT 核心，不自行实现神经网络算子；CUDA 内存/流只属于本进程。CPU 合同在既有 `q1-teleop-mps-20261106` 环境六项通过，Python 编译与 diff 检查通过。一次命令使用了不存在的短环境路径，未运行测试；随后使用实际环境通过，不能把路径错误算成合同失败。

准备记录：[x2_fgp_trt_prepared_20261113.json](../logs/session_records/x2_fgp_trt_prepared_20261113.json)。基线：[ARM64 移植与原始报告](README_X2_FGP_REALTIME_ARM64_20261113.md)。重要来源、自有代码先审计/发布，再部署原运行时合同及资源准入，之后依防重输出单次执行。

设备项目根 `/home/agi/x2_morph_human_20261113`，独立输出根 `data/analysis/X2_FGP_TRT_ACCEL_20261113`。构建入口：

```bash
env -u PYTHONPATH -u PYTHONHOME \
  data/environments/fgp-preview-py310/bin/python -m script.x2_fgp_accel.build \
  --onnx data/runtime/clotho_fgp_realtime_20261113/models/onnx_s1/cano.onnx \
  --precision fp32 \
  --output data/analysis/X2_FGP_TRT_ACCEL_20261113/build_fp32_r1
```

## R1 实际完成

准备提交 `6dff33e5473467e6082e358a147b4d6f95d72228` 已显式 push，并独立 ls-remote 核同40位后执行。五源码包 6,110 B/SHA `c2a7dba5c7ca18210e6fa3a8d3c38b2dc6ad7ce949ed23177e90bda8567067c7` 上板核同，五源码逐件 SHA 同；原环境六合同 .002 秒过。Mac 扩展头 provenance 被 GNU tar 忽略，五个实际文件完好，不重传重测。

| 完整链 | 中位 / p95 / 最大 ms | 超33.33ms | 一致性 |
| --- | --- | ---: | --- |
| 保存 CPU ONNX 基线 | 129.324 / 134.535 / 139.025 | 32/32 | 参考 |
| TensorRT FP32 CANO | 50.552 / 57.927 / 61.998 | 32/32 | 通过 |
| TensorRT FP16 CANO | 29.482 / 36.185 / 37.821 | 7/32 | **失败，不采用** |

FP32 residual 最大绝对差 `8.34465e-7`/相对L2 `6.20151e-7`，全链关节角差最大 0.05053 度、根平移最大 `6.30436e-7 m`；角差公式对 float32 近似旋转有数值底噪，不把0.05度当真人误差。FP16 residual 最大差 .078412/相对L2 .047729，最大关节角差13.9659度、根平移 .0308103 m，明显超预定门。中位29.5ms也不能掩盖7/32超时或一致性失败。

FP32 构建24.190秒、engine 96,506,588 B/SHA `eda242045eab76c5dc0b59db54ac7a8f51a5b9f90107a6419b47b03eb13bb739`；FP16 构建110.552秒、engine50,281,180 B/SHA `48e1a69b91cdbb59a09fad8ca9ae23d1bf49a8bfba90ca7691427d71f83a2b53`。最低真实可用内存10.963/9.838 GB、最高温66.843/67.406 C，两个构建子进程退出0。原ONNX SHA `adf4e93a607c45ee1a18d4e12a54648afbb709e5d5a769f9ed61743e5d581649`。

实际恰好两个engine/64次GPU CANO/64次CPU同输入参考，0训练和硬件命令。原基线代码、247上游源和24作者资产测试后再次SHA全同。32输出完整且tracking；原作者GP训练来源警告保留。CPU全链基线属于重启前历史，GPU测试属于新boot，非同场随机配对性能试验；逐输入CANO数值对照则在本次GPU计时之后执行。

原始 JSON/NPZ/log 已原生SCP取回，`X2_FGP_TRT_R1_RESULTS_20261113.tgz` SHA `fad36b1cfbd5fd8a35c821da5e4edffa6f9a8ab97902497207e86b63644cc2cd` 两端同，12份253,068 B原件及五执行源码SHA同；engine不回传，保持设备本地并登记SHA。只读分析/CSV/六宫图在 Mac `data/analysis/X2_FGP_TRT_ACCEL_20261113/analysis_r1`，PNG已目视，无新增模型调用。[R1六宫图](../data/analysis/X2_FGP_TRT_ACCEL_20261113/analysis_r1/six_panel.png)。

## R2 独立混合精度方案

R1 的两个engine和64调用预算已结清，不把失败FP16改写为通过或覆盖其根。只读解析网络实际有5,885层，包含 LayerNormalization、Softmax、两个ReduceSum；没有 parser 不支持错误。根据精度敏感性，新增**独立一个** FP16 engine，强制所有 `*LayerNormalization`、`*Softmax`、`/ReduceSum*` 的计算精度和输出为FP32，其余允许FP16；这是待验证假设，不能断言已定位FP16误差因果。

新增独立预算为一engine/600秒/256MiB workspace/32 GPU与32 CPU参考，资源门与R1相同；输出 `build_mixed_r2`、`probe_mixed_r2`，复用原保存CPU全链，不重跑R1两组。严格沿用FP16原数值阈值，不放宽。`--precisionConstraints=obey`，若不能满足则构建失败，不软回退。原作者/旧基线不改；仅自有build/probe支持新模式与一CPU合同，准备发布后再部署执行。第二第三终端保持不动，0新训练/硬件。

## R2 实测与当前设备分工

R2 已按独立方案完成，未新增第四个engine或重跑R1。构建90.462823秒、退出0，engine49,978,860 B/SHA `a10c99e9a5657dc617573bbc2be1a303277325b98aa117be9cb927602e223ea1`；最低可用内存9,932,763,136 B、最高温67.593 C。32个完整有限tracking输出的中位30.144/p95 37.725/最大57.336 ms，6/32超33.33ms；去前两次仍中位30.064/p95 35.820/最大39.551ms，5/30超时。

混合精度residual最大差 .0787252、相对L2 .0478101，关节角最大差13.95935度、根平移最大差 .0304361 m，**一致性失败，不采用**。强制上述三类运算FP32未消除偏差，不能据此定位单一误差原因，也不能把约30ms称为可部署30Hz。实际累计三个engine、96次GPU CANO与96次同输入CPU参考，0新训练/硬件命令。

用户关心两模型共享GPU压力，已在Mac与SoC1双端核实当前实际策略入口 `data/runtime/x2_morph_human/upstream/deploy/sim2sim/policy_node.py:325` 明确指定 `CPUExecutionProvider`。此前真实反馈6秒的接受RPC p95 13.690ms是**CPU ONNX策略的请求往返**，不是GPU前向；全部295请求有1次22.570ms超20ms并拒绝，不能把仅接受请求的p95当全部请求保证。见[原生writer结果](README_X2_MORPH_NATIVE_WRITER_20261113.md)。

当前优先分工为**FGP的CANO用GPU、Morph保持CPU、FGP其余LIP/GP与原生物理仍用CPU**，这是待并跑验证的部署建议，没有修改第二第三终端。即使分开处理器，整机CPU调度、内存和功耗竞争仍须实测；现有数据没有证明两模型同机50Hz稳定，也没有证明GPU容量不足。本轮闲置快照可用内存约12.42GB，不能代替并跑峰值或实时预算。通过一致性门的FP32 FGP整链单独运行已约50.6ms，首先还需满足其自身30Hz/33.33ms预算。

后续同机测试应先用录制人体与合成反馈、保持advisory，分列FGP完整链耗时/队列年龄、Morph全部RPC p95/p99/最大与20ms超时、实际内存峰值和温度，再决定是否需要双GPU方案。180ms人体预览仍属输入需求，不能通过CPU/GPU分工消除。当前未运行该并跑测试，未订阅新反馈或调用电机。

R2七份141,226 B原件和五源码已Mac/设备SHA核同，原247上游源/24资产终检再次通过；78,570 B结果包SHA `65df84e757afb4c9d4a82fae62577af557235c06a48cc1d4d669f4a7ca38f70e`。Mac只读四方案[六宫图](../data/analysis/X2_FGP_TRT_ACCEL_20261113/analysis_r2/six_panel.png)、CSV和summary已生成并目视，不新增推理。原件与engine留设备，Mac不复制engine。小记录见[x2_fgp_trt_r2_completed_20261113.json](../logs/session_records/x2_fgp_trt_r2_completed_20261113.json)，设备文档镜像状态以publication为准。

## R3 有界精度分配方案

用户进一步要求混合精度探索并以30Hz为最低目标。不重跑R2同一配方；新增实际TensorRT解析层计划，只给浮点数据矩阵乘法FP16计算，其余浮点计算及矩阵乘法输出保持FP32，整数/shape层不强改类型。计划读取同一个原ONNX，不重写模型；精确列出实际parser层名，构建日志须逐项确认计算精度与输出类型，否则失败。旧norm/softmax门已真实生效但不够，残差流精度敏感仍只是待验假设。

新增至多两engine：`mixed_gemm`及允许TF32的FP32对照，每engine最多600秒、256MiB workspace、独立32次GPU与32次CPU同输入参考。R3总计最多64+64调用、0训练；前两轮三engine/96+96不重置。原4GiB可用/80 C/10GiB余盘和自有进程回收门不变。新根 `build_gemm_r3`/`probe_gemm_r3`、`build_tf32_r3`/`probe_tf32_r3`。mixed_gemm保持原1度/10mm/.005/.02限值，TF32沿FP32更严的.5度/5mm/.0002/.001限值；不放宽质量换速度。

30Hz短样本门要求完整FGP链逐帧不超过33.33ms并且一致性通过；中位或p95过门而仍超时不称稳定30Hz。此32输出录制筛查仍不是长时间衣服/网页/机器人并跑认证。新增两计划合同连同原七项，本地9项.001秒通过；先发布和原运行时验证，再唯一新根执行。原作者/旧基线/第二第三终端不改，不接电机。准备记录见[x2_fgp_trt_r3_prepared_20261113.json](../logs/session_records/x2_fgp_trt_r3_prepared_20261113.json)。

## R3 结果与原始权重核同

实际parser为564个浮点计算层，其中97个矩阵乘法。首计划因TensorRT内部自动层名带字面星号而在构建前拒绝，随后仅允许其固定命名且模式精确唯一匹配，旧失败保留；没有因该解析失败新增推理。修正源码 `86eb2524ed4371d245023ccfcd41038409f609e1` 已push并独立核远端，Orin九CPU检查.002秒过。mixed_gemm构建170.802276秒/52,455,276 B，564项计算与输出约束全有日志回执。

| R3完整链32输出 | 中位/p95/最大 ms | 超33.33ms | 相对CPU参考 |
| --- | --- | ---: | --- |
| 逐层mixed_gemm | 31.970/36.742/67.497 | 11/32 | **一致性通过**：姿态最大.073327度，根最大.515055mm，residual最大.00224163/相对L2 .00170904 |
| TF32对照 | 41.201/48.585/58.899 | 32/32 | 严格FP32门失败：residual最大.00042522>.0002；姿态.033112度、根.059869mm仍保留 |

mixed_gemm去前两次中位31.818/p95 36.579/最大36.830ms、9/30超时，CANO中位15.178/LIP2.168/GP＋物理13.068ms；首帧CANO54.189ms单列。不能以平均约31Hz或去首帧称完整30Hz。TF32不提高到30Hz，也不为它放宽预定数值门。实际本轮两engine/64GPU64CPU参考结清，累计五engine/160GPU160CPU参考；原作者代码与模型未改。

用户新提供 `/Users/yu/Downloads/cano_ep120.pth.1`，285,461,969 B/SHA `288fa807273b183eb75f85e3b9aeeabe3715cd14e14c26c273de0feb17a882aa`，与当前ONNX原manifest的cano checkpoint SHA**完全一致**。`torch.load(weights_only=True,map_location="cpu")`成功，元数据model_size=S1、epoch120/global_step213720；144模型张量、23,773,449元素，FP32模型本体95,093,796 B（约90.7MiB）。272MiB文件含Adam状态，不因文件大而判另一模型或B版。原始参数已可用于后续导出优化，但当前ONNX本就来源于该权重，无需重复上传大文件或无理由重导出；不把S1称另取到纯S权重。

## R4 CUDA Graph与CPU线程准备

在精度已过门的同一mixed_gemm engine上，不改权重/作者链或增加engine。只捕获固定TensorRT kernel执行到本进程独有CUDA stream/graph，逐帧CPU→GPU输入和GPU→CPU输出仍完整同步；每帧仍真实计算，不缓存姿态或插帧。启动使用零缓冲对无状态CANO作一次GPU预热及一次capture enqueue，不推进LIP/GP/物理/人体历史，初始化耗时和额外调用独立记录。capture失败即保留失败，不软回退后宣称图加速。

新增预算最多两组：CPU4线程＋graph、CPU1线程＋graph，每组32真实输出/32CPU同输入参考/120秒，保守含初始化每组最多34GPU调用，总≤68GPU＋64CPU参考，0新engine/训练。旧R3预算不扩。资源门不变；输出 `probe_gemm_graph_t4_r4`、必要时 `probe_gemm_graph_t1_r4`。完整链全32帧≤33.33ms且原mixed一致性限值同时通过才称该录制短样本过门，仍需后续长序列与Morph CPU同机测试。新增graph包装和线程开关仅在自有加速命名空间；不修改第二第三终端。

## R4 已保存结果

源码`3ba380dd644c2f75bc85064d773d0d1ba756ea60`已发布核同。CPU4+graph完整链中位30.186/p95 36.131/最大37.080ms，11/32超33.33ms；CPU1+graph为30.101/37.896/38.868ms，8/32超时。两组一致性均通过，姿态最大差.073327度、根.515055mm，与R3完全相同。启动GPU准备各.390/.392秒单列；两组共64真实GPU输出、两warmup及两capture enqueue，64CPU同输入参考，保守68预算结清。新增engine/训练/硬件命令0。

原始两目录已原生SCP取回，结果包`data/deliveries/X2_FGP_TRT_R4_RESULTS_20261113.tgz`的SHA为`e5c847952ee4b12c0674f24b66c0496d9b23b88d06364d5851a9795b04c8338d`，Mac与SoC1同。CUDA Graph未解决完整链30Hz长尾，不以中位数宣称稳定；下一变量只移动GP神经网络。

## R5 GP网络GPU准备

复用同一准确CANO mixed_gemm engine，新增作者原样`gp.onnx`的TensorRT FP32（禁TF32）engine及CUDA Graph。GP输入features[1,1,267]、h/c[3,1,512]，输出prediction[1,1,9]及h_out/c_out；30Hz元数据对应每输出一次GP调用。每帧实际传输当前features和隐藏状态、同步输出后由作者网络更新状态；初始化零输入仅建图，不推进外部人体或物理状态。LIP、特征、原生物理和第二第三终端不改。

独立预算一engine/600秒/workspace256MiB，一组CPU4完整32输出/120秒：64真实GPU网络调用＋两warmup＋两capture，保守上限68；计时后32CANO＋32GP同输入CPU参考。沿用全链1度/10mm和CANO .005/.02门；额外GP三输出合计最大绝对差≤2e-4、相对L2≤1e-3且所有值有限。不放宽门，不重跑R4。GPU前向与GP隐藏状态均保存，短样本整链全32帧≤33.33ms才过速度门。

新根`build_gp_fp32_r5`和`probe_gemm_gp_graph_t4_r5`。原GP SHA`1a292e5fac116ff81fca39a0900317db286a94fa304ad00628ae02b3f3bfdb61`，原来源/资产保持。可用内存4GiB、80度、10GiB余盘及自有项目锁边界延续；17:49实核原boot、可用12.375GB/最高66.843度/余盘374.170GB且无加速进程。十一CPU合同.001秒通过（含GP输出顺序/隐藏状态复制），新设备GPU执行尚未开始。准备记录见[x2_fgp_trt_r5_prepared_20261113.json](../logs/session_records/x2_fgp_trt_r5_prepared_20261113.json)。

## R5 实际结果

准备源码`e1fc991`和索引`7f04c2dca6eeadd51eadabb3ba8085ce9950daec`已push/独立40位远端核同，八源码上板SHA同，十一板端合同.003秒通过。GP FP32 engine构建5.046682秒/23,319,452B/SHA`7b9d10bd971ac1d8fea8fa900d9904a8f51eb3b97f8b07e751d32780f4e3361c`，最低可用11.258GB/最高66.968度，退出0。

完整32输出中位30.407/p95 40.809/最大43.837ms，8/32超33.33ms；去前两次仍8/30超时。一致性通过：CANO最大差.00224163/相对L2 .00170904，全链姿态.073327度、根.515058mm；GP预测及h/c同输入CPU参考最大差3.814697e-6、相对L2 1.355153e-7。作者日志仍打印其CPU类名，实际session已被显式GPU适配器替换，记录以适配器引擎SHA和对照为准。

GP+物理阶段去前两次中位R4为12.487/R5为12.458ms，原生物理分别6.336/6.502ms；该小样本未证GP搬GPU有明确收益，不将R5替代当前优先方案。累计新一engine/64帧GPU调用＋两warmup＋两capture/64CPU参考结清，0训练硬件。原始包`data/deliveries/X2_FGP_TRT_R5_RESULTS_20261113.tgz`为1,225,814B/SHA`3f660e4d612d7f6e490cb4b3cfc7298718b3617b930392acc665c8ea9f759434`已Mac/设备核同；engine仍板端，不为图表重跑。

## R6 构建搜索独立方案

先复用R3 exact precision计划和原始ONNX，只将TensorRT builderOptimizationLevel从2改为5，GP恢复原CPU路径，CPU4+同一graph包装保持。板端安装的trtexec帮助明确支持0至5，更高级别只允许构建时更多优化选项；是否加速待实测。数值门不变，不改权重、历史长度、Euler次数、原生物理或系统频率/功耗。

新增最多一engine/600秒/workspace256MiB，一组32真实输出＋一warmup＋一capture（保守34GPU）、32CPU同输入参考/120秒，资源边界沿用。新根`build_gemm_opt5_r6`、`probe_gemm_opt5_graph_t4_r6`。所有564项精度/输出约束仍须日志逐项确认；32完整链各帧≤33.33ms且一致性通过才过短样本门。R5和R4不重做；本方案准备时新GPU0，先发布和原运行时合同后执行。

R6实际于18:09触发600秒原时限：`BUILD_FAILED`/`TimeoutError('600 second build limit')`，600.756189秒，子进程返回-15且已回收；无engine（0B/SHA null），因此没有R6模型probe或CPU参考，不把日志中子图生成当完整engine。最低可用9,223,389,184B/最高68.843度，仍在资源边界内。失败根/log保持，不扩预算或重启构建；该观察仅说明本次opt5未在预算内完成，不证明其潜在速度好坏。

## R7 相同人体特征复用准备

只读核实作者`GlobalPoseTracker.update`在GP头fps=30/substeps=1时，会两次对相同pose/acc/gyro调用`PoseFeatures`；后者仅使用固定SMPL offsets做FK和拼接，无可变时序状态。两来源固定SHA为features.py `8b6a4a67d4a4d054b139ceebc36ba8501e84dcebb0ca0768a7c7fb621e89a207`和tracker.py `324e93beeb21ab86c9ea8f9dfb32031cb127161b3bdb35ea3e5734c94d1ccc58`。此为源码机会，不是已测加速收益。

新增独立单条缓存，仅完整shape/dtype/逐字节相同且有限输入才命中；保存键及输出副本，调用方更改返回值不会污染缓存。只在本进程30Hz tracker实例挂包装，退出恢复工厂；不改原作者文件、GP网络更新、接触/物理步数或动力学。每帧仍运行CANO/LIP/GP和一次物理，缓存只省重复FK/特征计算。十三CPU合同.001秒过，新增两项覆盖所有权和不精确/NaN键。

独立R7固定复用R3 engine＋CPU4/graph/原CPU GP，以已保存R4 CPU4原件为精确对照，不与R6混合。新根`probe_gemm_featurecache_graph_t4_r7`，0新engine，最多32GPU＋一warmup＋一capture（保守34）/32CPU参考/120秒。原混合精度门和全部32帧≤33.33ms门保留，额外要求pose与translation相对R4逐位相同、64次特征请求至少32次命中且最多32次真实特征计算；来源不符或任何差异则失败。R6运行时不部署R7，不热改在跑文件；先发布，再等R6终态和独立准入后单派。

## R7 实测与当前结论

准备`9f40dd4f2b82012b94994d52382d3e0be199a138`和有符号零合同`904789a45cc0b1e8e39d3a0931f56e2537d819aa`均已独立远端核同。R6结束且原锁空后部署三文件，5,948B代码包SHA`9d13bc32094dc9073731ed9fbc7dc3f1e13faa43ac34565c276bb3a2a8726678`及三个源码SHA全同；十三板端合同.004秒过。单派原120秒内退出0，64特征请求/32命中/32真实计算，32帧pose与translation相对R4逐位相同；不是近似缓存或跳过物理。

| 保存方案 | 完整链中位/p95/最大 ms | 超33.33ms | 一致性 |
| --- | --- | ---: | --- |
| 历史CPU参考 | 129.324/134.535/139.025 | 32/32 | 参考 |
| 原准确混合＋graph CPU4（R4） | 30.186/36.131/37.080 | 11/32 | 通过 |
| GP也搬GPU（R5） | 30.407/40.809/43.837 | 8/32 | 通过，但未证速度收益 |
| 相同特征复用（R7） | 29.086/35.884/36.429 | 8/32 | 通过，且与R4输出逐位同 |

R7去前两次仍8/30超时，完整32输出平均30.149ms，倒数约33.17Hz仅为不定时回放吞吐，不能当连续30Hz认证。去前两次CANO/LIP/GP+物理中位14.011/2.029/11.629ms，GP+物理较R4少约.858ms；不同场次的性能差不单独当因果证明。相对原CPU参考最大姿态差.073327度/根.515055mm是数值一致性，不是真人重建误差。

实际R7新增32帧GPU＋一warmup＋一capture/32CPU同输入参考结清，0训练/新engine。至此成功engine共6（其中GP一个），R6失败构建保留；累计真实帧网络调用320GPU与320CPU参考（CANO各288、GP各32），额外GPU初始化10次；TensorRT builder自己的profiling不计作帧推理。没有扩大失败预算。18:12后自有build/probe进程全退，项目锁释放。

原247上游源/24资产、原native模块和历史结果由独立只读`verify_stage`终检全部通过，未重复原推理或物理。新包`data/deliveries/X2_FGP_TRT_R6_R7_RESULTS_20261113.tgz` SHA`ea98bea3d9a90b7e071692b7df15d48d08736b83f5f585b10ee905ba6db13821`已原生SCP回Mac核同；失败R6日志与R7 JSON/NPZ均保留。Mac新增[R7六宫图](../data/analysis/X2_FGP_TRT_ACCEL_20261113/analysis_r7/six_panel.png)、CSV/summary已目视，R4/R5图也保存，无新模型调用。

当前保留R7为下一步候选，原衣服实时入口未自动切换；探针仍有32输出上限。还需要真实墙钟30Hz长序列/队列年龄/掉帧与Morph CPU 50Hz同机验证，然后再讨论接入第一终端。保持CANO GPU＋Morph CPU优先分工，GP GPU方案暂不采用；这不改变第二第三终端。180ms人体预览与FGP执行加速分别记录。未接BLE/网页/机器人执行，无真人GT；硬件命令、新反馈订阅、训练均0，原Q1暂停保持。

简短结论与待办见[给用户的结果](README_X2_FGP_S1_ACCEL_RESULT_20261113.md)，完整工件大小SHA以`x2_fgp_trt_r7_completed_20261113.json`及后续publication为准。小文档三端同步在publication核实后才登记，网盘备份未验证。
