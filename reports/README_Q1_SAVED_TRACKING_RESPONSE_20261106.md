# 保存轨迹的表观相位与幅度诊断

2026-11-06北京时间。独立实验Q1_SAVED_TRACKING_RESPONSE_20261106已完成，只读取已交付Primebot最终10000轮motion基线的393片保存轨迹，分析8个肩/肘通道，共3144条信号。不是当前MorphMimic/V04的human回放，也没有新机器人策略、物理积分、训练或CUDA工作。

## 方法与可辨性

每条信号在-300到+300ms、20ms网格上比较原reference与actual。正号表示较晚的actual与较早reference对应，即表观滞后；负号表示表观提前。每个候选lag都用同一组reference索引15到N-16，避免不同裁剪长度制造更低误差。原始全程未平移RMSE始终单列保留，平移后的数字只作诊断。

另报告标准差幅度比、平均偏置和在相关性最佳lag处的描述性仿射增益/残差。仿射拟合不用于修正交付指标。参考/actual低激励、周期多峰、搜索边界、弱相关，以及相关性与原始误差最优错位不一致均显式标记。

配置在读取actual前固定并写入输出根：std阈值.01rad、相关性阈值.7、离峰差.01内的候选跨度不超过60ms，以及相关性和原误差lag差不超过40ms。这些是描述性启发式，既不是统计置信区间，也不构成策略质量或训练准入新门。完整搜索曲线只保存三固定片，全库保留逐关节摘要。

## 实际结果

| 项目 | 结果 |
| --- | ---: |
| 片段 / 关节信号 | 393 / 3144 |
| 窄峰且高相关的候选 | 1404 |
| 弱波形相关标记 | 1580 |
| 宽峰或多峰 | 313 |
| 搜索边界峰 | 248 |
| 相关与原误差最优错位不一致 | 346 |
| 低激励 | 38 |

标记可重叠，不能相加当作片数。1404个窄候选中表观正lag441、零lag266、负lag697；候选lag的p05/中位/p95为-100/0/80ms。候选的actual/reference标准差比p05/中位/p95约为.646/1.219/2.155。不能把中位0ms称为零端到端延迟，负lag也不是负物理延迟：策略可能预览未来，幅度/偏置和形状差异也影响匹配。

三固定片未平移8关节整体RMSE为前走.154571rad、起走.126943rad、停走.312272rad，与原保存诊断口径一致。前走左肩pitch出现80ms表观lag且相关.895，但左肘最佳相关为负且幅度比约3.25，不能称左肘“只是迟了300ms”。停走左肩yaw的幅度比约6.75、相关仅.517；平移不能解释这类跟随失真。

这项证据支持把相位、幅度和形状分开检查；不支持把整套问题归为一个可统一抵消的200ms常数。当前V04共享策略必须用自己的同human接口H10/H5/H4/H2/H1对照，原Primebot结果不可替代。

## 时间边界

这里只有仿真参考时轴与保存的实际关节状态；没有硬件capture、网络往返、在线重定向耗时、真实q_target/qd_target链路和执行器采样时钟。因此不能隔离PD本身相位，也不能把表观lag与预览预算相加当作实测遥操总延迟。

本轮另核作者[ExtremControl v4，2026-10-15更新](https://arxiv.org/html/2602.11321v4)：其输入预览与低层控制相位分开处理，并讨论速度前馈、噪声放大和离散控制稳定性。论文G1系统的181ms与约48ms等数字是作者特定设置，不用于Q1调参；旧研究页引用的v1为当时核验版本。下一次独立执行端实验需真实命令/速度/子步日志与新动力学验证，不能直接修改当前V04的PD。

## 完整性、资源和工件

共786份CSV/case JSON逐大小SHA匹配已发布交付，原始参考身份/全长和输入前后SHA核同。5项新增检查使用-W error全过，无跳过，覆盖已知正负lag、统一支持、幅度/偏置、低激励、短片、多峰、边界和缺值不变零。

实际Mac单线程分析2.297秒、峰RSS45219840B；协作式时限90秒/RSS1GiB，未访问服务器或争GPU锁。结果路径为Mac主根data/supermimic_for_q1/Q1_SAVED_TRACKING_RESPONSE_20261106/：

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| result.json | 4121293 | 99ca45956c92e88d9213c5c2ef1ae9482f943094dca1263b41130853850a59e2 |
| configuration.json | 1073 | 928fc2f4c8a65224786103fc5641a306258f4d3831a3c1c9cff7c86cf57f5eeb |
| status.json | 798 | f1ae30397ab6427ee01bf1704d033be9c9db560a5fe3b5b2238928b3839194d2 |

[小型审计](../logs/session_records/q1_saved_tracking_response_20261106.json)。完整结果留data，不入Git；发布和Mac镜像以后续收据为准，服务器新记录镜像待原桥接。此前上肢预测ca3dddc93c4e67463d88274eef5b76dece686b55已独立远端及Mac十文件628102B核同，[发布收据](../logs/session_records/q1_upper_causal_forecast_publication_20261106.json)。

## 复现与下一步

在权威Git根运行：

~~~sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -W error -m unittest discover -s script/supermimic_teleop -p 'test_tracking_response.py'
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -W error script/supermimic_teleop/tracking_response.py \
  --local-cpu-study --project-root /Users/yu/Documents/ChatGPT/元启 \
  --plan data/manifests/q1_teleop_forecast_split_20261106.json \
  --admission data/manifests/primebot_q1_admission_frozen_20261103.json \
  --sources data/manifests/morphmimic_faithful_common_sources_results_20261101.json \
  --review-root /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/PRIMEBOT_Q1_FINAL_DELIVERY_20261104/review \
  --delivery-manifest data/manifests/primebot_q1_final_delivery_manifest_20261104.json \
  --output /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SAVED_TRACKING_RESPONSE_20261106
~~~

已完成根拒绝覆盖，不重跑已完成分析或物理。下一步解释所选预测器的噪声放大，并继续真正human短预览对照的服务器准备；不凭噪声test选新的参数。goal保持active，至少执行到2026-11-06 10:00，原V04/final/完整五阶段和采纳null保持。
