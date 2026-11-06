# 因果上肢预测的抗噪训练探索

2026-11-06，Q1_UPPER_NOISE_ROBUST_20261106已完成。固定规则先提交推送8b34bd043ea5a906fb28de73c05721cb3f79351f，独立远端及Mac镜像核同后，08:29:08才开始读取真实数据和拟合。新抗噪模型在独立噪声下改善，但牺牲部分干净参考精度，也没有全面胜过旧模型。未部署策略，未据此宣称自然遥操或端到端延迟改善。

## 数据与实验身份

沿用[上一轮](README_Q1_UPPER_CAUSAL_FORECAST_20261106.md)393片已交付reference肩/肘轨迹与167/98/128来源划分，训练8192、验证2048、测试2048窗口。786份CSV/case工件42220058B的发布身份和读取前后SHA均通过；actual机器人角度不入训练或标签，TEST_ONLY真人小包未使用。窗口身份与上一轮一致。

测试集已在上一轮被查看。虽然本轮仍先保存validation选择再读取test标签，全部test结果均明确是探索性复比，不是新确认性留出，也不是全身SMPL或原策略陌生片泛化。

预登记文件script/supermimic_teleop/upper_noise_robust_plan_20261106.json，SHA 4f012a13964b45b333a91a99845063407a908b20fff8a89370a09689b48b32f1。所有候选和压力条件在真实数据前固定，未根据本轮test改参数。

## 训练与选型

归一化仍只拟合干净train输入。新目标为干净预测均方误差，加.01rad独立输入噪声造成的精确期望误差，再加原ridge罚项。若预测为y=xJ+b，则噪声风险增加sigma²乘J各元素平方和。这里J包括相对历史特征、标准化和最新姿态输出锚点；忽略锚点会得到不同目标。

固定lambda=.0001/.01/1，共3次真实闭式拟合。旧validation所选ridge .01仅读取原模型，不重新拟合。另保留hold/CV/damped CV；无滤波、限幅、时轴变换或测试噪声调参。

本轮validation标准是0/.01/.02rad三档独立噪声下，十时距RMSE的等权平均；噪声积分由仿射协方差精确计算，不向标签加噪。新noise_ridge_1.0以.056139rad胜出，对照阻尼.072116、旧模型.081950。选择文件先落盘，随后才读test。

## 探索性复比

下表单位rad，口径均为十个20-200ms时距RMSE的等权平均；各方法使用完全相同的压力历史。独立噪声seed=2026110602，与上一轮压力样本不同。

| 输入 | hold | 阻尼CV | 旧干净选型 | 新抗噪选型 |
| --- | ---: | ---: | ---: | ---: |
| 干净 | .065982 | .059140 | .039339 | .046069 |
| 独立std=.01 | .066988 | .063617 | .068280 | .049308 |
| 独立std=.02 | .069770 | .074706 | .116768 | .057351 |
| 时间相关AR1 rho=.8/std=.01 | .066973 | .062178 | .047989 | .048180 |
| 整个窗口固定偏差std=.01 | .066987 | .060451 | .041192 | .047528 |
| 最后5帧冻结为更早的第19帧 | .111344 | .111344 | .102957 | .103631 |

新模型在干净/独立.01/独立.02下分别120/126/128片优于阻尼，三固定走片在.01噪声下均改善约22.8%-27.2%。相对旧模型，干净均值退步约17.1%，.01独立噪声均值改善约27.8%；AR1略退，固定偏差和冻结也未胜过旧模型。不能称全部压力通过。冻结是历史数值的人工压力，模型没有独立可用性输入，不代表真实丢包安全控制已经验证。

80/100/200ms的噪声RMS增益由旧模型5.409/5.982/6.745降至新模型1.615/1.701/1.814，说明目标确实抑制了原始历史噪声放大。该结果同时显示：降噪和预测偏差存在取舍，时间相关误差与独立噪声不能混作同一训练分布。

## 对短缓冲路线的意义

H4/H5保留原20ms时间槽，分别需要从最新可见锚点预测余下120/100ms。对本轮.01独立噪声，前六/五个未知时距RMSE均值分别为：新模型.034725/.030580，阻尼.042154/.036495，旧模型.052914/.047887。这只汇总已保存预测误差，不包含可见前缀、全身姿态、策略敏感性或机器人反馈；不是H4/H5闭环实验，也不把80/100ms预览预算称为实测延迟。

保留“短缓冲+抗噪预测”作为可测试路线有依据。是否优于直接短掩码，要由准备好的同human接口对照决定。零预览和因果学生仍值得独立训练研究，不能用预测器准确性代替它们。

## 验证、资源与工件

7项检查全过，-W error、无跳过；覆盖原始历史Jacobian、解析噪声积分、闭式最优梯度、只允许train拟合、固定压力的可复现/不改输入、选型落盘先于test、已完成根拒绝覆盖。

实际Mac单线程全流程1.214567秒，峰RSS163528704B，低于预登记90秒/1GiB。0服务器命令、机器人策略推理、物理积分、CUDA或原训练修改。用户GPU优先用于后续安全训练/媒体；本分支是既有小矩阵的独立本地计算，不是服务器CPU训练替代。

完整工件保留Mac主根data/supermimic_for_q1/Q1_UPPER_NOISE_ROBUST_20261106/：

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| result.json | 3342646 | 48a728eb84e35d9d3aceeb9c3bf542f02daa0ceed6b76253e9ad442fce9cfd5f |
| selected_new_model.json | 524659 | ad57ed43d33b659c34ac5dbf26c5f2bd1c6dcff4597985d927e15d07cad7b971 |
| validation_selection.json | 536 | 0fad4eef219f827a7b3d1c373b685c51a0a2a3a72cb6470d412abaac8299612e |

[实际审计](../logs/session_records/q1_upper_noise_robust_20261106.json)、[预登记审计](../logs/session_records/q1_upper_noise_robust_preregistration_20261106.json)。完整结果/模型不入Git，服务器记录镜像待可用原Windows桥接。原V04实时UNKNOWN，原final/完整五阶段/采纳null保持。goal继续active，最低今日10:00，不能以本轮上肢预测完成结算。

## 复现

~~~sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -W error -m unittest discover -s script/supermimic_teleop -p 'test_upper_noise_robust.py'
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -W error script/supermimic_teleop/upper_noise_robust.py \
  --local-cpu-study --project-root /Users/yu/Documents/ChatGPT/元启 \
  --plan data/manifests/q1_teleop_forecast_split_20261106.json \
  --admission data/manifests/primebot_q1_admission_frozen_20261103.json \
  --sources data/manifests/morphmimic_faithful_common_sources_results_20261101.json \
  --review-root /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/PRIMEBOT_Q1_FINAL_DELIVERY_20261104/review \
  --delivery-manifest data/manifests/primebot_q1_final_delivery_manifest_20261104.json \
  --preregistration script/supermimic_teleop/upper_noise_robust_plan_20261106.json \
  --prior-model /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_UPPER_CAUSAL_FORECAST_20261106/selected_ridge_model.json \
  --output /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_UPPER_NOISE_ROBUST_20261106
~~~

已完成根不可覆盖，不重复拟合。本轮没有生成或采纳可直接上线的全身策略。
