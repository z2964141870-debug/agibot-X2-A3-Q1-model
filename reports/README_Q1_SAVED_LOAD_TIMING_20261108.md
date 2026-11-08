# Q1保存分叉的承重标签时序

2026-11-08，北京时间。新只读诊断已经完成，不是新增仿真、策略推理、训练或硬件测试。原16分叉、62个来源SHA前后相同；八依赖锚点来自四条已探索train/validation动作，不能当独立留出恢复试验。原共享训练保持暂停，用户采纳`null`。

done：

1. 新增[固定计划](../script/q1_checkpoint_review/saved_load_timing_plan_20261108.json)、[时序分析](../script/q1_checkpoint_review/saved_load_timing.py)及[八项合同](../script/q1_checkpoint_review/test_saved_load_timing.py)。默认CPU合同/.011秒通过，实际绘图环境同八项/.012秒通过；两AST/CLI通过。分别验证当前阈值、因果确认不回填、未来不影响既有标签、未确认尾部保留、候选中断重置、缺时间不当零延迟、回顾性A-B-A不当在线输入和无效来源拒绝。
2. 实际分析与绘图1.402938秒、峰RSS147832832B，842条模式切换记录及4800条逐帧设置记录，0新策略/物理/Adam/GPU秒。三阈值1/5/10N、即时与两样本确认均已输出，缺支持及终止转换没有补齐。
3. 六宫PNG/PDF已目视，无文字覆盖；两CSV、原数值、README共六成员ZIP，CRC、成员大小/SHA和5个相对链接全过。原速度诊断、旧九张恢复图、已有视频及84事件不重做。

## 新证据与决定

| 动作/锚点 | 共同有效步 | 5N原始切换 S/T | 5N两样本确认 S/T | 首次原始标签不同(ms) |
| --- | ---: | ---: | ---: | ---: |
| train A095/64 | 80 | 16/7 | 10/6 | 160 |
| train A095/96 | 51 | 9/10 | 5/6 | 40 |
| train A075/64 | 47 | 13/8 | 9/5 | 60 |
| train A075/96 | 15 | 6/4 | 5/4 | 80 |
| validation A267/64 | 80 | 22/16 | 9/6 | 220 |
| validation A267/96 | 80 | 25/18 | 13/13 | 40 |
| validation A033/64 | 37 | 13/8 | 7/4 | 20 |
| validation A033/96 | 2 | 1/1 | 0/1 | 20，支持过短 |

S为f0学生，T为特权H10教师。至少80ms支持的七锚点中，教师六处原始模式切换更少，一处更多；1至10N阈值范围内，八锚点切换次数差的正负排序均不变，只有A267/64的教师计数16变18。但A095/96教师比学生切换更多，原完整有效步却是80对51，不能将“少切换”直接设为质量或优化目标。A267/96原始25对18的差异经过两样本确认后变成13对13，也不能把滤波后的计数变化归因于策略学习。

本数据中两样本确认把每个锚点首次可见标签分歧后移20ms：例如A095/96从40ms到60ms。50Hz下，从首次不同样本到第二个连续样本相隔20ms；代码只在第二个样本实际到达后发出新标签，未把候选起点回填为在线生效时间。这是已知滤波行为和保存数据的观察，**不是策略响应、步态相位或硬件端到端延迟**。若以后用于控制，其额外等待必须单列，不能被隐藏在“因果”名称下。

5N两样本确认有五个学生尾部候选尚未确认、教师为零；这些尾部停在双方共同区间边界，按右删失保留，不能补用后续或终止帧。单样本A-B-A是需两邻居的回顾性模式，可重叠，不能作为当前帧可见特征。只有40ms支持的A033/96不用于恢复或相位判断。

**决定：保留接触时序/接触模式辅助预测作为候选，否定仅凭这些计数直接增加“切换惩罚”的依据。** 下一步仍先做已发布的多步FP32 WM校准，再决定时序损失；不跳过模型可靠性门或追加同MSE拟合。教师末子步承重标签是已有行为的观测，既不是计划参考接触真值，也未证明真实物理接触梯度可用。

## 工件

[证据ZIP](../data/deliveries/Q1_TEACHER_RECOVERY_LOAD_TIMING_R3_20261108.zip)，268505B，SHA-256：`8d13073bdfafc213cacf9f8f1317b737b75f20732ff7d86b6443c9cb3a393b14`。含一张六宫PNG、一PDF、两CSV、summary及README，无新增MP4。服务器镜像仍PENDING。

![承重时序六宫图](../data/analysis/Q1_TEACHER_RECOVERY_LOAD_TIMING_R3_20261108/load_timing_six_panel.png)

图中近白尾部表示没有共同支持；色条灰色模式0表示末子步两脚都未达所选阈值，不能等同无真实接触。模式1/2/3分别是左/右/双脚达到阈值，没有接触力预测或计划真值。

本机`/usr/bin/python3`与工具所报bundled Python均缺matplotlib，两次出图失败前已产生两CSV，原失败根`Q1_TEACHER_RECOVERY_LOAD_TIMING_20261108`及`..._R2_20261108`均保留。定位原有`data/environments/q1-teleop-mps-20261106/bin/python`，numpy1.26.4/matplotlib3.9.4后，使用新R3输出完成。三次产生的两CSV大小SHA逐位相同，不安装新依赖或改数值规则。R3是本地出图恢复编号，非新增GPU训练stage r3。

可复现命令，保留原件时应换未存在的输出与ZIP路径：

```sh
data/environments/q1-teleop-mps-20261106/bin/python data/repositories/q1-rf12-20261028/script/q1_checkpoint_review/saved_load_timing.py \
  --packet data/analysis/Q1_TEACHER_RECOVERY_NUMERIC_20261108 \
  --prior-summary data/analysis/Q1_TEACHER_RECOVERY_FIGURES_20261108/summary.json \
  --plan data/repositories/q1-rf12-20261028/script/q1_checkpoint_review/saved_load_timing_plan_20261108.json \
  --output /absolute/fresh/output --archive /absolute/fresh/evidence.zip
```

从项目主根执行，输出和ZIP要求绝对路径。实际完成与来源记录见[收据](../logs/session_records/q1_saved_load_timing_completed_20261108.json)；源码发布与独立远端、Mac同步以后续publication为准。

todo：

1. 12:05:43原45014仍实回Windows日期；本轮有界原生第二跳再次明确Connection refused，未到认证，第一跳不等于失联。Linux资源、人体条件CPU审计及曲线仍未核，不循环登录、不代用户重启服务或训练。
2. 恢复后先读已提交CPU审计/原曲线并补双训练六宫图，不重派已成6000更新；再按已发布五源码/原manifest做多步WM部署、原运行时、来源资源准入及唯一派发。
3. 以真实多步结果决定接触/速度/时序机制。新留出事件、可获得真实人体流、完整软件及硬件延迟、成本质量结论仍缺；本诊断不结算整体自然遥操目标。成本账本仍5036/14400秒、无新增预留，原共享暂停/final门与采纳null保持。
