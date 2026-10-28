# Q1全身折中UR03

记录日期：2026-10-28，北京时间。状态：原始方法研究、五档真实闭环三重复、四项测试及本地小工件交付完成；Git发布身份由历史/同步收据提供。用户要求寻找上肢/下肢trade-off；RF12及旧训练已经结束，本阶段0optimizer，不新增训练、恢复监督或启用agent。

## 本次结论

当前冻结策略存在可测折中；75%参考混合在本次网格内值得保留，但**不是根部/上肢同时改善**。它使上肢误差降低约83%、骨盆heading相对腕误差降低约57%，原30验证三次均0提前终止、固定两走/抬手均完整；代价是共享前缀根XY误差增加约54%、腿部约5%、承重滑速约11%、PD目标振荡约25%。没有达到RF12原约定的根误差最多增加10%条件，不据此重开训练或追加五小时。

| 参考混合alpha | 上肢RMSE rad | 相对腕RMSE m | 根XY RMSE m | 腿RMSE rad | 验证提前终止/30（三次） |
| --- | ---: | ---: | ---: | ---: | --- |
| 0，原RF11 | 0.94412 | 0.12339 | 0.12368 | 0.26479 | 0 / 0 / 0 |
| 0.25 | 0.56517 | 0.08904 | 0.15561 | 0.28257 | 0 / 1 / 1 |
| 0.50 | 0.32027 | 0.06582 | 0.18905 | 0.28526 | 1 / 1 / 0 |
| 0.75 | 0.16071 | 0.05345 | 0.19057 | 0.27837 | 0 / 0 / 0 |
| 1，直接参考 | 0.06987 | 0.05448 | 0.22941 | 0.28630 | 2 / 2 / 2 |

误差列是同一重复中五档共享前缀、每片RMSE先等权平均再跨三次平均；不是所有存活时刻混算。共享前缀平均5.1793/5.1793/5.1940秒，包含各条件最早终止帧，不包含终止后重置状态。逐候选对RF11的配对前缀也单独保存，其数值不能与共享前缀列混用。三次是同片重复，不是90条独立验证；现有验证集用于扫档，不能声称独立泛化或部署安全。

alpha0.25的根损失较小，但仍增加约26%，并非自动最稳；alpha0.50与0.75根代价接近，后者上肢/腕更准且本次无验证失败。alpha1关节更准，但相对腕误差反而略大于0.75且出现早终止。这说明整个响应曲线不是单纯的“越少改手越稳”。记录的九项目标下五档均未被严格支配，不能把0.75称为全局最优。

固定抬手的271目标步：alpha0/0.25/0.50/0.75三次均完整；alpha1为197/271/271步，失败保留。alpha0.75双腕骨盆相对峰高约0.476/0.532m，参考0.534/0.560m；原RF11约0.253/0.129m。峰高绝对误差均值0.3562→0.0431m，但左腕峰时1.50/1.82/1.82秒、右腕1.24秒，参考1.42/1.72秒，幅度改善不代表时序准确。所有档位固定普通走均320/210步完成，未换片。

## 目标与预定协议

UR01直接参考手臂显著降低上肢/腕误差，但根部与稳定性变差；UR02同时测到神经输入敏感性和物理耦合。开环回放本身失稳，故本次始终保留下肢真实闭环反馈。固定RF11教师及归一化、原PD/动作尺度/噪声关闭的评测协议，不改reward或参考数据。

五档预定alpha=0、0.25、0.5、0.75、1；每档3重复，原30验证+固定普通前向走两片/双手抬起一片，共33真实MuJoCo Warp环境。只将8维手臂物理PD目标替换为`(1-alpha)*q_policy + alpha*q_reference[f+1]`；14维腿/腰动作仍按实际历史实时计算。alpha0动作必须逐值等于原策略，alpha1必须等于UR01直接参考端点，非均匀动作尺度/非零上下文和默认角换算均测试。最终仍使用原动作限幅，不能把alpha当实际抬手幅度比例。

每个repeat固定按alpha升序运行，沿用reset-to-reference协议，不宣称底层求解器隐藏状态或各重复独立同分布。保存完整JSON/NPZ、验证失败与完成率、固定片逐步数及腕峰高/时序。同时提供逐条件对RF11配对共同前缀、五档共享前缀，后者每片长度完全相同但会被最早终止条件缩短；两者均须搭配完整失败/完成记录。报告候选的非支配关系，不将验证扫档称为独立测试、全局最优或部署保证。

独立输出`data/supermimic_for_q1/Q1_UR03_UPPER_TRADEOFF_20261028`；0optimizer、最多25分钟诊断，保留30GiB共享cgroup/OOM0、100GiB盘余、12GiB实验、86°C和900秒无进展保护。运行前后核1123项RF12冻结来源和RF11权重，不覆盖旧实验来源。

## 方法研究

以下为作者论文/官方项目页核验，不是Q1复现结果；首次本阶段阅读2026-10-28。

- HUGWBC（RSS2025）：论文IV-E/F通过训练时替换上肢动作为平滑外部干预，并根据步态跟踪表现增减混合系数；介入时屏蔽冲突的上肢正则项。可迁移假设是让下肢在渐增手臂干预下适应；原文主要是速度/步态命令，不等同本项目逐帧全身轨迹。来源`https://hugwbc.github.io/resources/HugWBC.pdf`、`https://hugwbc.github.io/`。
- HOMIE（arXiv2502.13013v1，2025-02-18）：III-B2用上肢姿态课程训练下肢适应连续上肢姿态，结合高度跟踪与对称性；不能仅从外骨骼演示推断任意Q1动作均可行。来源`https://arxiv.org/html/2502.13013v1`。
- ConstrainedMimic（arXiv2606.00374，初稿2026-05-29，核验v2为2026-10-14）：在输入运动和策略输出两侧加入考虑接触约束的运动学/动力学过滤，含质心稳定性限制。作者明确说明仅修改几何参考不足以处理惯性及实际跟踪误差。对Q1需适配接触与力矩模型，不能把静态支撑多边形或当前混合器叫作该过滤器。来源`https://stanfordasl.github.io/constrainedmimic/`、`https://github.com/StanfordASL/constrainedmimic`、`https://arxiv.org/abs/2606.00374`。
- KungfuBot（arXiv2506.12851v3，2026-07-01）：3.2根据当前跟踪误差自适应指数奖励容差；是优化容差课程，不是运行时稳定性保证。可供未来分项尺度调整参考，尚未照搬其G1数据过滤、奖励或PD。来源`https://arxiv.org/html/2506.12851v3`。

这些原文提供的是方法依据，尚未移植/复现到Q1；没有引入新第三方安装、改奖励或复制其机器人数值。X/论坛转述未作为证据。

## 具体方案选择

当前精度预算可明确选择：根跟踪优先时继续保留RF11作为基线；允许约6.7厘米额外根XY RMSE、优先手势时，可用alpha0.75作为仿真候选对照。现有网格里没有非零alpha同时满足“上肢改善、根误差增加不超过10%”；不把保住存活等同保住根轨迹，也不推断更小未测alpha一定满足约束。

后续最值得验证的结构沿用UR02冻结教师与14维下肢/腰残差，同时把上肢参考干预作为明确条件。训练课程从alpha0逐步增加到本次候选区间，普通走/低干预状态继续保留，使腿部学习真实上肢变化产生的惯性与接触补偿。记录根/腿/腕分项物理指标、教师偏移和KL，不能只提高全局上肢reward后整体改写原网络。KL保持可能妨碍必要的补偿，系数及可接受偏移仍须实测，未声称已有正确预算或学习器。

运行时可进一步研究平衡优先的可回退alpha，输入实际根误差、躯干倾角/角速度和原生接触/承重滑速，使用变化率限制与滞回；何时回退、是否有足够时间纠偏均未标定。全局XY漂移本身不等于即将跌倒，单个接触快照亦不足以给出支撑安全保证。若简单残差仍不能守住根约束，再评估ConstrainedMimic式动力学过滤，届时需Q1力矩/接触模型及独立守约验证。上述为有文献依据的待验证候选，未自动实现动态门控或开始训练。

## 版本、产物与恢复

权威Git为Mac持久`data/repositories/q1-rf12-20261028`，Q1分支；起点`9184622f9f749562ec2eda7989f23bbb540773d0`。本阶段新增`q1_upper_blend.py`、端点/物理语义测试及独立`diagnose_q1_upper_tradeoff.py`，不改旧PPO/HP/runtime。实际代码身份由本次`plan.json`记录，不改变旧冻结来源提交。

仅25386，根`/opt/data/private/ZHY`。连接资源核查13:20:47：原boot/GPU一致，GPU15MiB/26°C无计算进程、共享7700127744字节/OOM0、余盘1652079550464字节。Python仍经activate。复现命令见下，输出不得覆盖已有目录。

四项单元测试在Mac/25386均通过：alpha0精确保留/不修改输入，alpha1等于原直接参考端点，中间档正确处理非零base/非均匀scale且下肢逐值不变，拒绝非法alpha/形状；另通过语法检查。测试不等同物理效果证据。

实际2026-10-28 13:30:25至13:35:31北京时间；controller88185/start_ticks7906923，worker88195/start_ticks7907023，boot不变。三状态COMPLETED，guard WORKER_COMPLETED/worker_reaped=true/return0。终后原进程均不存在，GPU15MiB/26°C无计算进程、共享约7.82GB/OOM0。1123旧冻结来源286568437字节前后逐大小SHA相同，冻结教师/归一化指纹不变。没有优化器或新训练计时。

```sh
# 独立0optimizer诊断；output必须为新目录，不能覆盖本次证据。
data/environments/supermimic_q1_hp_20261026/bin/python \
  script/supermimic_q1_hp/activate.py script/project_tools/diagnose_q1_upper_tradeoff.py \
  --runtime /opt/data/private/ZHY/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027 \
  --library /opt/data/private/ZHY/data/supermimic_for_q1/AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json \
  --checkpoint /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF11_SOMA300_SAMPLERONLY_E4096_MAX01H_20261027/run/final.pt \
  --frozen-plan /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF12_PPO_SOMA_AP03_E4096_MAX08H_20261028/plan.json \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/NEW_NONTRAINING_UR03_DIRECTORY
```

日志`logs/Q1_UR03_UPPER_TRADEOFF_20261028.log`。同目录`results.json`、15个分条件JSON/NPZ、`tradeoff_summary.json`保存原值与失败，`run/status.json`/`guard.json`/`status.json`提供终态。`deliver_q1_upper_tradeoff.py`只读保存结果生成六格`upper_tradeoff.png/pdf`和长表CSV，无新增积分；这是诊断曲线，不是训练500iteration滑均图。PNG已目视检查，本阶段不重做历史54秒训练MP4。

Mac目录`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_UR03_UPPER_TRADEOFF_20261028`，26个JSON/PNG/PDF/CSV共2953599字节与服务器逐大小SHA全部相同；权重/大NPZ只留服务器SERVER_LOCAL_ONLY，不宣称云备份。量值/来源与工件清单见`data/manifests/supermimic_q1_UR03_upper_tradeoff_20261028.json`。阶段代码/记录按payload audit、显式push origin Q1、独立核40位远端后镜像三端，最终发布身份以Git历史和`data/manifests/q1_UR03_sync_20261028.json`为准，不改旧冻结训练提交。RF12监督保持暂停，其他自动任务未改。
