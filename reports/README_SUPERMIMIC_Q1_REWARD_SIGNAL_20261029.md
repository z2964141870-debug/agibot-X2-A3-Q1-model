# Q1底座与PPO的抬脚奖励信号

记录日期：2026-10-29，北京时间。阶段：SOLE02。状态：保存轨迹CPU分析完成，36组/9,612状态；6项测试通过，0跳过。没有新策略推理、物理积分或optimizer更新，未证明新步态收益。

## 目标与范围

用户最新要求“继续探索研究”，延续SuperMimic底座和PPO残差两条线。此次读取原RF16的初始、约301/601/1201实训秒、三次重复、原两段普通走和固定抬手，定位贴地摆脚与学习信号的关系。没有使用过期夜间预算、恢复旧训练或第三次H32物理运行；PPO16/4096仍未测，用户采纳null，原监督及其他任务不变。

本轮是保存状态的解析反事实：固定其余身体误差，仅把脚底Z误差设为0，测量位置奖励能增加多少。它不保证存在可实现的动作，更不包含未来critic值、姿态/速度项或闭环物理影响。轨迹的实际动作代价是绝对代价，不能当作纠正抬脚需要额外支付的边际代价。

## 主要结果

下表为两个固定普通走、四阶段、三重复的24组最小至最大值；不同阶段不是配对干预。

| 指标 | 保存数据结果 | 含义 |
| --- | --- | --- |
| 参考声明摆脚时，原生足力仍>5N的足部帧 | 60.4%–69.4% | 要摆的脚仍经常承重 |
| 同一声明摆脚窗口，脚底Z<=5mm的足部帧 | 82.3%–91.5% | 贴地并非只出现在个别截图 |
| 底座：理想脚底Z修正的位置奖励增量 | 每20ms约1.89e-6–2.96e-6 | 原14-body均值、0.3m尺度对厘米级脚高不敏感 |
| 同轨迹原生动作变化代价 | 每20ms约9.24e-4–1.94e-3 | 仅是量级背景，不是纠正动作边际成本 |
| 上述脚高增量等价的12腿目标同时变化 | 中位数约0.0022–0.0034rad | 仅与原动作变化项同一步等值 |
| PPO：理想脚底Z修正的高度增分，仅在摆脚帧平均 | 约0.00053–0.00074 | 高度项对现有误差区分有限 |
| 同轨迹额外PPO平滑代价 | 每20ms约0.00085–0.00201 | 没有据此直接减小平滑权重 |
| 8上肢占原生动作变化代价 | 4.89%–9.51% | 大部分观测代价来自14腿腰 |
| 8上肢占额外PPO平滑代价 | 2.72%–5.68% | 不支持主要被无直接残差输出的上肢代价淹没 |

声明相位来自原重定向接触意图，并非仿真原生承重真值；保存足力是最后5ms子步读数，不是整个20ms的冲量。此处统计双脚全部声明摆脚足部帧，与先前“右脚/参考高于20mm窗口内94%–100%贴地”口径不同。

底座原位置项为`dt*exp(-mean14body_position_error/0.3**2)`。脚底定义替换解决测量对象问题，但两个Z分量仍被14个body平均与0.3m尺度稀释。此结果支持优先校准足部信号，不能证明完整actor目标或世界模型梯度方向错误。

PPO高度项原尺度0.04m：20mm误差仍保留`exp(-0.25)=77.9%`的高度分支分数，XY分支及其他奖励另计。仅把踝Z改为脚底Z，在同一条未改善的走路轨迹上，平均额外分数就会上升约0.00249–0.00316；评分上涨不能作为步态进步。

残差直接输出14维腿腰，8维上肢目标仍受未来状态间接影响。代价分项保留原总维数归一化、各关节action scale和完整总量截断后再分摊，36组重新合计与先前总成本全部一致；没有把上肢视为完全不可控或据此断言PPO有偏。

## 实现与实际检查

- `script/supermimic_q1_sole/reward_signal.py`读取330库顺序、原接触metadata、原生MDP/FK及保存状态，输出分项与逐帧数据。正式CPU分析于17:44:26完成。
- `target_cost_attribution.py`读取同批保存targets，将原生动作变化项和额外平滑项分为14腿腰/8上肢；17:58:52完成36组重建。首轮读JSON字符串路径触发`AttributeError: 'str' object has no attribute 'read_bytes'`，未创建产物；已显式转Path，并补临时JSON/NPZ入口及源文件变更拒绝测试，不是训练重试。
- `test_reward_signal.py`为6/6通过、0跳过，覆盖替换保留其他误差、非法输入、reset后缀与帧对齐、总截断/不同action scale、JSON路径及源轨迹变化保护。
- 三文件编译与CLI帮助通过。图例/坐标范围调整后仅从CSV重绘，未重算状态或改写分析收据；最终PNG已目视，图例不遮挡数据。

第一转移没有保存前一目标，两类动作代价均从第二帧起算。固定抬手没有声明摆脚，条件均值为null。阶段NPZ核历史SHA和本次前后身份；初始NPZ只有原路径收据、帧时轴及本次前后SHA，没有补称初始历史SHA。

## 两条线的下一项

| 线路 | 下一项独立候选 | 必须检验的证据 |
| --- | --- | --- |
| SuperMimic底座 | 保留全身项，另校准相位内脚底Z分辨率；先按真实摆脚幅度比较尺度与梯度 | 足部信号是否足够，支撑脚/姿态是否受损；H32方向仍需有效同起点对照 |
| PPO残差 | 固定同底座、动作权限与原平滑代价，单独校准摆脚高度项 | 同相位卸载/离地、连续换脚、承重滑移、全程稳定；不能以新奖励涨分晋级 |
| 起点核查 | 分开核策略输入/输出一致性与后端动力学重复性 | 保留严格状态/历史/warmstart保护；不自动追加第三次物理检查 |

现有数据不支持先删上肢平滑惩罚，未隔离奖励、探索、critic、残差权限或模型的单一失败因。两候选未组合验证，没有可直接采纳的新检查点。本轮只量Q1，不能证明G1/Q1形态差异就是主因。

安装版本实测为Torch2.9.1+cu128、MuJoCo3.11.0、MuJoCo-Warp3.11.0、Warp1.16.0；安装包`types.py`没有`DeterminismType`定义。这只是诊断接口边界，未据此判后端为起点差异原因，也未升级软件。已读策略/归一化/重置源码仍不足以把seed、BF16或WM定为根因。本轮外部3.11文档再次获取遇TLS错误，未用未取得的网页支撑结论。

## 版本、命令与证据

权威Git为`/Users/yu/Documents/ChatGPT/元启/data/repositories/q1-rf12-20261028`的Q1分支，开始HEAD为`5dcb60ad46592b23bcdf9c97cc3e84415781e24f`。RF16训练冻结仍为`2c83b944059d4a007f0ca74fa54c0076fda3a357`，不因本轮分析改变。

以下为唯一25386容器已执行的CPU命令。原输出已存在、工具拒绝覆盖；如需复现，须另指定全新输出目录，不能重新执行原路径。保存文件内部标签`initial_static075`为沿用旧命名，RF16实际初始是静态alpha.5，不是.75。

```sh
env CUDA_VISIBLE_DEVICES= timeout -k 10s 240s /opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python /opt/data/private/ZHY/script/supermimic_q1_hp/activate.py /opt/data/private/ZHY/script/supermimic_q1_sole/reward_signal.py \
  --runtime /opt/data/private/ZHY/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027 \
  --scene /opt/data/private/ZHY/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027/q1_assets/scene.xml \
  --trial /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029 \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/Q1_SOLE02_SAVED_REWARD_SIGNAL_20261029
env CUDA_VISIBLE_DEVICES= timeout -k 10s 180s /opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python /opt/data/private/ZHY/script/supermimic_q1_hp/activate.py /opt/data/private/ZHY/script/supermimic_q1_sole/target_cost_attribution.py \
  --review /opt/data/private/ZHY/data/supermimic_for_q1/Q1_SOLE02_SAVED_REWARD_SIGNAL_20261029
python3 -m unittest discover -s script/supermimic_q1_sole -p test_reward_signal.py -v
```

保存状态脚本SHA为`b555044f0045cf5be4e7a56fcfad86555f35bc4f39015838402de7450928dc8d`。分项运算时SHA为`aa75b911c9d44fd551cf9fcb730ae8c3f22fe175c1ffa85cd984d11bca6bb3c1`；随后仅改绘图的最终SHA为`cc95c3916ac57293fbb6b28c01e159b00677d21623e7ab59875795761bb7d531`，原运算收据保留旧SHA。原始来源身份在summary/attribution收据内，终端摘要见`logs/session_records/sole02_reward_signal_20261029.log`。

## 保存与恢复

7个JSON/CSV/PNG/PDF小工件共1,896,597字节，与三份最终代码合计10项1,925,254字节，权威工作树/Mac主根/25386逐大小SHA全同。原权重/大NPZ仍SERVER_LOCAL_ONLY，未称网盘备份。[完整清单](../data/manifests/supermimic_q1_sole02_reward_signal_20261029.json)。

- [本地四格图](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SOLE02_SAVED_REWARD_SIGNAL_20261029/reward_signal_review.png)
- [本地PDF](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SOLE02_SAVED_REWARD_SIGNAL_20261029/reward_signal_review.pdf)
- [分项CSV](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SOLE02_SAVED_REWARD_SIGNAL_20261029/fixed_reward_signal.csv)
- [关节代价CSV](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SOLE02_SAVED_REWARD_SIGNAL_20261029/fixed_target_cost_attribution.csv)

本阶段记录按payload audit、显式push Q1、独立40位远端核验后镜像，实际发布以随后publication收据为准，不预称本地文档已经发布。整体步态目标未完成，没有修改监督、其他任务、agents或goals。下一步先按上表分别形成奖励尺度候选与起点证据，再确定独立物理验证范围。
