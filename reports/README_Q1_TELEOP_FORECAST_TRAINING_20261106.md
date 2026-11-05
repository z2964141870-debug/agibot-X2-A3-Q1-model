# 因果预测器特征、模型和有界训练入口

2026-11-06北京时间。已完成仅使用可见历史的几何特征、物理单位标签解码、独立训练入口和逐时距物理单位误差评估入口；尚未读取服务器393份human侧车，尚未启动模型训练、策略推理、机器人物理或GPU作业。长期goal仍active，至少执行至今日10:00，必要训练/验证继续延长。

## 表示

`script/supermimic_teleop/forecast_geometry.py` 将最后可见帧作为锚点：

- 根平移使用同一世界偏航坐标；其余23个身体点使用相对根位置。
- 每个关节的姿态用相对于最后可见姿态的SO(3)旋转向量表示；四元数按最短反号处理。
- 历史特征是每帧216维，标签是未来每帧144维（根3、23个相对位置69、24个旋转向量72）。历史固定25帧，目标固定10帧，对应20至200ms。
- 解码后同时给出世界线速度和角速度，二者都用最后可见帧到当前目标的20ms后向差分，避免速度与姿态错位。

特征对共同世界平移和偏航不变；未来数组被修改不会改变历史编码。该表示没有骨长、脚接触、碰撞、力矩或动力学约束，不能单独证明自然步态。

## 模型和训练边界

`forecast_model.py` 定义单层、单向GRU（默认隐藏128）和10槽输出头。输出头初始为零，因此未训练模型严格等价于保持最后可见姿态；训练标签不减均值，只按训练集尺度标准化，保证零输出仍是物理上的hold基线。归一化只接受 `split=train` 的窗口，验证和测试会拒绝拟合或加载。

`train_forecast.py` 是独立实验入口：

- 输出必须位于正式V04目录之外的新根；不读取或修改V04 checkpoint、训练输入或保存频率。
- 启动前要求cgroup总量低于22GiB、限额高于30GiB、OOM计数为0、GPU低于18GiB/80°C且无计算进程，并以非阻塞方式取得Q1 GPU锁；原V04持锁时直接拒绝。
- 默认 `--device auto` 只在安全CUDA可用时运行；没有GPU时必须显式 `--allow-cpu`，避免无意把长训练落到CPU。
- 运行中总量达到24GiB、GPU达到18GiB或80°C、OOM变化、磁盘余量低于100GiB时退出本实验；默认最大1800秒/2000步，验证只读取validation，最终test不参与模型选择。
- 保存独立 `normalization.json`、`forecast_gru.pt`、`status.json` 和 `result.json`。每步记录loss和资源，但loss只用于训练监视，不作为遥操自然性结论。

07:06补审修正：1800秒协作式时限现覆盖归一化、优化、验证及保存，各阶段检查设备boot、实际cgroup/OOM和GPU/磁盘，资源快照最多间隔2秒。它在数值调用间检查，不是对阻塞CUDA调用的外部硬超时。验证样本按来源片段均衡、片内等距选取，不再只取清单最前256个窗口；不完整末批按实际元素数加权。CPU保存前核参数有限性和split身份，checkpoint记录完整schema、实际optimizer计数及来源计划身份；异常写FAILED，既有实验根在拒绝时保持原样。

当前本地环境没有Torch/服务器资产，因此没有调用入口。服务器侧车审计和真实资源门通过后才能运行，训练结果还必须接入H10/H5/H4/H2/H1同权重、hold/CV/damped尾部及原速物理媒体比较。

## 独立评估入口

`forecast_eval.py` 只用每个窗口的25帧可见历史生成 hold、常速度和阻尼常速度基线；如果传入独立 `forecast_gru.pt`，再加入GRU结果。它把预测解码回世界位置和四元数，在20至200ms每个目标槽分别报告位置、根位置和姿态测地线RMSE，并保留逐片结果。位置采用欧氏距离RMS，与 `forecast_geometry.error_by_horizon` 同口径；已知三维偏差(1,2,2)cm的结果应为3cm，不除以坐标数。

默认每个split最多2048窗口、上限8192；按来源均衡并在整片时段等距取样，所有方法使用完全同一组窗口及SHA，不能冒充全窗口评估。所有方法/参数预先固定，先由validation选型并落盘 `validation_selection.json`，再读取test标签作报告。测试集不参与选择；离线插值依赖及没有实测机器人延迟的边界保留。

CLI要求独立新输出根及boot/GPU/cgroup/磁盘资源门，CPU数值评估可在原GPU计算存在时通过资源门运行，GPU推理必须非阻塞取得同一Q1锁且无其他计算进程，否则拒绝。GRU先在CPU以受限权重读取方式加载，检查完整schema、冻结split身份、非零更新、仅train且覆盖全部训练源的归一化、张量有限性及读取前后SHA，再移到获准设备。没有绕过原V04锁、原训练资源线或final门。

```sh
python3 script/supermimic_teleop/forecast_eval.py \
  --plan data/manifests/q1_teleop_forecast_split_20261106.json \
  --admission data/manifests/primebot_q1_admission_frozen_20261103.json \
  --sources data/manifests/morphmimic_faithful_common_sources_results_20261101.json \
  --human-root /opt/data/private/ZHY/data/supermimic_for_q1/MORPHMIMIC_Q1_SOMA_FULL1H_V01_20261102/converted/npz50/human50 \
  --project-root /opt/data/private/ZHY \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/Q1_CAUSAL_FORECAST_EVAL_20261106/result.json \
  --checkpoint /opt/data/private/ZHY/data/supermimic_for_q1/Q1_CAUSAL_FORECAST_GRU_20261106/forecast_gru.pt
```

命令默认CPU数值评估；GPU空闲并获准时可显式使用 `--device cuda`。省略 `--checkpoint` 可只比较三种固定基线。GRU预测质量对照须先于GRU尾部接入，已准备的原生短预览/固定尾部物理campaign可以独立执行；预测误差不能替代脚底离地、交替、承重滑移、侧倾或急停行为。

## 验证

完整teleop测试现共96项，93通过、3项受限于本机缺Torch或固定服务器运行时而跳过；`-W error`、compileall和diff检查通过。新评估9项、训练门5项覆盖非零距离口径、来源均衡取样、验证/test相反偏好、schema/split拒绝、资源门先于载入和实际文件锁冲突。原Torch检查已扩展CPU保存加载往返，但仍待服务器执行；不能将skip当作Torch验证完成。没有TEST_ONLY真人录制、测试集、reward/loss或物理回放被用于训练选择。

已知限制：当前393片原策略本身见过全部数据，预测器测试集不能证明整套策略对陌生动作的泛化；来源后缀没有被试身份证明；120→50Hz侧车上游动捕滤波和硬件capture因果性仍未知；预测输出还没有接触约束和置信度回退。

## 复现

```sh
python3 -W error -m unittest discover -s script/supermimic_teleop -p 'test_*.py'
python3 script/supermimic_teleop/train_forecast.py \
  --plan data/manifests/q1_teleop_forecast_split_20261106.json \
  --admission data/manifests/primebot_q1_admission_frozen_20261103.json \
  --sources data/manifests/morphmimic_faithful_common_sources_results_20261101.json \
  --human-root /opt/data/private/ZHY/data/supermimic_for_q1/MORPHMIMIC_Q1_SOMA_FULL1H_V01_20261102/converted/npz50/human50 \
  --project-root /opt/data/private/ZHY \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/Q1_CAUSAL_FORECAST_GRU_20261106
```

该命令尚未在服务器执行；它是一次独立有界实验，不是对当前V04训练的续训，也不自动重试。

v2补审代码 `931be10582f43847896e5b289967ea1e9f5db023` 已显式push并独立40位远端核同，13份661958B与Mac主镜像逐大小/SHA相同。完整197项源码身份清单及独立部署包已准备：16成员201863B，每成员与Git原字节同；独立解包34项33过1缺Torch跳过。包只应解到新的独立代码快照目录，不覆盖原V04输入。[发布收据](../logs/session_records/q1_teleop_forecast_eval_publication_20261106.json)与[新审计](../logs/session_records/q1_teleop_forecast_eval_guard_audit_20261106.json)。

评估初版 `dc1093f6a980a783556cf22bf034d29d52a3f10a` 和此前训练准备 `49fd7781acfbbf418f78242ed0c91847f4e40c57` 都没有在服务器运行。服务器桥接和原训练实时状态仍UNKNOWN，新包未上传或派发；不能用本地夹具通过替代真实侧车、GPU或遥操闭环结果。原final及完整五阶段门、采纳null和原训练保护保持。[冻结数据阶段](README_Q1_TELEOP_FORECAST_DATA_20261106.md)
