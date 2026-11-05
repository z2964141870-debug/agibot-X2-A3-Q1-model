# 已保存上肢参考的因果预测实测

2026-11-06北京时间。实验Q1_UPPER_CAUSAL_FORECAST_20261106已正常完成。这是利用已有交付文件进行的真实小型模型拟合与留出比较：学习型预测在干净参考上有收益，但验证集选出的模型在固定噪声压力下退步。不能据此部署预测尾部或宣称遥操、步态合格。

## 数据与预登记

读取已有Primebot最终全库交付的393份upper_timing.csv，只取8个reference肩/肘通道，actual机器人角度不进训练或评价标签。相应393份case结果均核完整、帧数和原参考SHA；CSV与case JSON共786份42220058B，逐项匹配已发布交付清单，读取前后SHA相同。没有读取服务器SMPL侧车、权重或TEST_ONLY真人录制。

来源沿用已冻结的37组划分：训练167片、验证98片、测试128片。额外核完整Q1参考内容身份，未发现同一完整参考跨集。每片按来源均衡、片内等距取窗口，训练8192个、验证2048个、测试2048个；不是全112227窗口评测。每个输入为25帧可见历史，标签为接下来10个20ms时间槽，无片尾padding或跨片窗口。

运行前固定的[预登记配置](../script/supermimic_teleop/upper_forecast_plan_20261106.json) SHA为9353ceabfeed9fac11d6179a2790c74b1d4ebfc24fb9596a515cdc86353d634b。主比较只用干净validation选型；固定test噪声为独立高斯std=.01rad/seed20261106，不参与选型。实际先写validation_selection.json和所选学习模型，再读test标签。

## 模型与资源

单个输入为200维：过去24帧相对最新姿态，加最新8维姿态；目标为未来十帧相对最新姿态的80维增量。标准化只拟合训练输入。线性ridge使用平均平方损失、未惩罚偏置，预登记lambda为.0001/.01/1，共真实拟合3个模型；无梯度优化器。对照为保持、5点锚定常速度和tau=.12s阻尼常速度，均不加滤波、限幅或时轴重对齐。

这是明确的Mac本地轻量CPU分支，单BLAS线程、90秒协作式时限、1GiB RSS界限。实际全流程0.990秒、峰RSS155123712B，未接触服务器、V04锁或CUDA。它不替代已准备的全身SE(3) GRU/GPU训练；后者仍待可用桥接、真实侧车与资源门。

## 结果

validation选择ridge_0.01，十个时距RMSE均值.049886rad。测试集的比较口径为8关节误差的每时距RMSE，再对十个时距等权平均；下表单位均为rad。

| 测试输入 | 方法 | 十时距均值 | 80ms预测 | 100ms预测 | 200ms预测 |
| --- | --- | ---: | ---: | ---: | ---: |
| 干净 | 保持 | .065982 | .052594 | .063812 | .108739 |
| 干净 | 常速度 | .072533 | .047754 | .063119 | .142035 |
| 干净 | 阻尼常速度 | .059140 | .042810 | .054876 | .106315 |
| 干净 | validation选出的ridge .01 | .039339 | .028049 | .036949 | .070456 |
| .01rad噪声 | 保持 | .067076 | .053568 | .064633 | .109250 |
| .01rad噪声 | 常速度 | .079442 | .053950 | .069483 | .150113 |
| .01rad噪声 | 阻尼常速度 | .063950 | .047667 | .059448 | .110112 |
| .01rad噪声 | 原选出的ridge .01 | .068267 | .061004 | .070440 | .097575 |

干净测试下，所选模型平均误差比阻尼低约33.5%，128片中126片改善；两片非常慢的停走/弧线起走略退约1.7%-2.0%。加入预先固定噪声后，平均反而比阻尼高约6.8%，仅40/128片改善；最差片约为阻尼的1.97倍。三段固定前走/起走/停走在噪声下分别退约39.7%/17.7%/29.9%。不能只保留干净结果。

完整结果另保留所有ridge候选。lambda=1在噪声test上更好是事后观察，未替换validation选择、未改预登记、未将测试最优当作可采纳模型。下一次抗噪方案需事先固定训练噪声、验证压力与选择规则，并明确同一测试集已被查看，不能冒充全新确认性留出。

## 含义与边界

这支持进一步研究“历史中存在可学习的短期结构”，也直接暴露干净参考训练对观测噪声的敏感性。它不证明全身SMPL未来可可靠恢复：未覆盖根、腿、接触、骨长、急停意图、执行力矩或闭环稳定性；原始动作和重定向仍是离线数据，实时预处理因果性未认证。原机器人策略本身见过393片，不能称整套系统的陌生片泛化。

80/100/200ms在本表是预测时距，不是机器人端到端延迟。预测误差减小不表示自然性通过；保留原final及完整五阶段门、用户采纳null。当前不把本模型接入策略或根据test改参数。

## 验证与保存

新增9项边界检查全部通过、0跳过，使用-W error；覆盖发布身份、完整参考/时轴、实际角度排除、训练归一化、不可见未来、先落盘选择后读test和已有输出不覆盖。最初两项因本机Accelerate的matmul数值警告失败，发生于夹具、尚未读真实数据；改为等价显式einsum且保持警告为error后9项通过，未抑制警告。

原始工件位于Mac主根data/supermimic_for_q1/Q1_UPPER_CAUSAL_FORECAST_20261106/：

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| result.json | 1266629 | 36c391a64ccc28fc5c5eae45d866f94ebe339a82c9d2f8f925a38ecef0f16864 |
| selected_ridge_model.json | 517134 | be36ef81c0c9564d3b9e0b249d6e5031637dcf9ad86c2310ec4a500dc1e988fd |
| validation_selection.json | 383 | b9f8d01c03118584a834f3d8f162cf432349af544f17cc7e0292af46cf9097a6 |

模型与完整结果保留在被忽略的data，不入Git。[小型审计](../logs/session_records/q1_upper_causal_forecast_20261106.json)包含结果摘要、输入/输出身份和边界。实际发布/镜像以后续收据为准，服务器镜像待原桥接。

## 复现

在权威Git根运行，下方Mac主根沿用用户已有目录：

~~~sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -W error -m unittest discover -s script/supermimic_teleop -p 'test_upper_forecast.py'
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -W error script/supermimic_teleop/upper_forecast.py \
  --local-cpu-study \
  --project-root /Users/yu/Documents/ChatGPT/元启 \
  --plan data/manifests/q1_teleop_forecast_split_20261106.json \
  --admission data/manifests/primebot_q1_admission_frozen_20261103.json \
  --sources data/manifests/morphmimic_faithful_common_sources_results_20261101.json \
  --review-root /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/PRIMEBOT_Q1_FINAL_DELIVERY_20261104/review \
  --delivery-manifest data/manifests/primebot_q1_final_delivery_manifest_20261104.json \
  --preregistration script/supermimic_teleop/upper_forecast_plan_20261106.json \
  --output /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_UPPER_CAUSAL_FORECAST_20261106
~~~

已完成根会拒绝覆盖，不为等待而重复训练。下一步分析已保存actual与reference的执行相位/幅度失真，继续等待原桥接完成真正human短预览物理比较。长期goal保持active，最早2026-11-06 10:00才可考虑结算，必要验证继续延长。
