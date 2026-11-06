# 已选因果预测器的噪声放大

2026-11-06北京时间。Q1_UPPER_FORECAST_SENSITIVITY_20261106已完成。只读取上一轮已选线性模型和结果，解析传播输入噪声；没有新拟合、原轨迹读取、机器人推理、物理积分或CUDA工作。

## 方法与结果

将上一轮25帧×8关节历史到10槽×8关节未来的预测写为原始历史的仿射映射。输入标准化、相对最新姿态的特征和输出锚定均还原到同一个Jacobian；每个输出的独立同方差输入噪声增益为该列系数平方和的平方根。下表是每时距8关节的RMS增益。

| 方法 | 20ms | 80ms | 100ms | 200ms |
| --- | ---: | ---: | ---: | ---: |
| 保持 | 1.000 | 1.000 | 1.000 | 1.000 |
| 常速度 | 1.346 | 2.445 | 2.819 | 4.702 |
| 阻尼常速度 | 1.318 | 2.044 | 2.219 | 2.769 |
| 原validation所选ridge .01 | 2.179 | 5.409 | 5.982 | 6.745 |

例如80ms时，.01rad独立输入噪声给所选ridge带来约.0541rad的噪声误差分量，而阻尼常速度约.0204rad。这解释了[上一轮](README_Q1_UPPER_CAUSAL_FORECAST_20261106.md)干净预测改善、固定观测噪声下退步的现象；不能据干净误差直接部署预测尾部。

原干净test的误差平方加上解析噪声方差，可给出上一轮固定噪声test的期望RMSE。ridge十个时距的预测与已保存实际结果最大相对差约.192%，四方法总体最大.656%。这里只解释已发生结果，没有据test重选模型或修改原选择。

该解析式严格适用于此仿射模型与独立、零均值、同方差的角度噪声。真实动捕存在时间和关节相关误差、遮挡、偏差与突变；本轮没有估计传感器噪声分布，也没有处理全身SO(3)、根或接触。预测时距不等于端到端延迟。

## 验证与完整性

4项边界检查全过，使用-W error，无跳过：原始历史仿射映射对直接预测和有限差分一致、三个固定基线一致、独立噪声协方差、零权重等于hold、错误来源和非有限归一化拒绝。原模型517134B/SHA be36ef81c0c9564d3b9e0b249d6e5031637dcf9ad86c2310ec4a500dc1e988fd，原结果1266629B/SHA 36c391a64ccc28fc5c5eae45d866f94ebe339a82c9d2f8f925a38ecef0f16864，读取前后核同。

实际Mac单线程分析.009013083秒、峰RSS30687232B，沿用90秒/1GiB本地轻量界限。结果在Mac主根data/supermimic_for_q1/Q1_UPPER_FORECAST_SENSITIVITY_20261106/result.json，19234B/SHA 78a5a84326f1da32618bce9de9f85641762b0bd6a36a1c4d38b085ee780bbabf。[小型审计](../logs/session_records/q1_upper_forecast_sensitivity_20261106.json)。大结果和模型不入Git，服务器镜像待原桥接。

## 下一步

新抗噪研究需先固定训练噪声模型、候选和验证压力选择规则，再拟合；原test已被查看，只能作为探索性重复比较，不能称新的确认性留出。可使用解析期望噪声损失约束原始历史Jacobian，避免仅对归一化权重加罚而忽略物理噪声增益；是否有收益由实际验证和完整压力结果决定。

真实human/H10-H1与H4/H5尾部物理对照仍待可用原Windows桥接；不能用本地上肢系数分析代替。原V04/final/完整五阶段和采纳null保持，goal继续active，最低执行至今日10:00。

~~~sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -W error -m unittest discover -s script/supermimic_teleop -p 'test_upper_forecast_sensitivity.py'
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -W error script/supermimic_teleop/upper_forecast_sensitivity.py \
  --local-cpu-study --project-root /Users/yu/Documents/ChatGPT/元启 \
  --model /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_UPPER_CAUSAL_FORECAST_20261106/selected_ridge_model.json \
  --result /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_UPPER_CAUSAL_FORECAST_20261106/result.json \
  --output /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_UPPER_FORECAST_SENSITIVITY_20261106
~~~

已完成根拒绝覆盖，不重复运行。
