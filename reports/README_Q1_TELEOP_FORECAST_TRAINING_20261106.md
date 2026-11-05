# 因果预测器特征、模型和有界训练入口

2026-11-06北京时间。已完成仅使用可见历史的几何特征、物理单位标签解码和独立训练入口；尚未读取服务器393份human侧车，尚未启动模型训练、策略推理、机器人物理或GPU作业。长期goal仍active，至少执行至今日10:00，必要训练/验证继续延长。

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

当前本地环境没有Torch/服务器资产，因此没有调用入口。服务器侧车审计和真实资源门通过后才能运行，训练结果还必须接入H10/H5/H4/H2/H1同权重、hold/CV/damped尾部及原速物理媒体比较。

## 验证

新增8项几何检查、5项模型/归一化检查、2项训练入口边界检查。完整teleop测试共84项，81通过、3项受限于本机缺Torch或固定服务器运行时而跳过；`-W error`、compileall和diff检查通过。没有TEST_ONLY真人录制、测试集、reward/loss或物理回放被用于训练选择。

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

本阶段代码和报告待本轮发布；服务器桥接和原训练实时状态仍UNKNOWN，不能用本地夹具通过替代真实侧车、GPU或遥操闭环结果。原final及完整五阶段门、采纳null和原训练保护保持。[冻结数据阶段](README_Q1_TELEOP_FORECAST_DATA_20261106.md)
