# RF02核心训练链路只读审阅

记录日期：2026-10-25，北京时间。阶段：RF02-M01。状态：审阅完成，RF02训练仍在进行；本报告不是训练完成或策略效果证明。

## 目标与范围

用户要求优先复刻SuperMimic，最多使用一个子agent。本阶段由唯一子agent只读审阅RF02的replay时序、8步世界模型、H32想象BPTT、critic目标及回合边界；主代理独立核对具体采样问题、固定上游身份和实际动作长度。未改训练源码、未启动新训练/评测、未新增数据门槛。

当前学习核心来自第三方`CuboiLeo/SuperMimic-Reproduced@d9be385b3eeb13a638fb33fb200d3bfd9be9839f`，不是论文作者官方训练源码。来源与许可见[THIRD_PARTY](../script/supermimic_for_q1/THIRD_PARTY.md)。本次对照该固定第三方实现和已登记的方法，不声称与官方源码逐行一致。

## 审阅结果

未发现足以确认RF02核心训练失效、要求立即中止本轮的实现错误。以下为源码审阅结论，不替代最终动力学评测：

| 环节 | 实际核对 |
| --- | --- |
| 世界模型8步 | 从记录状态开始，用执行的带噪动作自回归拟合，没有每步重置为真值或断梯度。 |
| replay时轴 | 当前行存`terminated(s_next)`，取当前行done与本地记录语义一致；不能机械改回上游的下一行索引。 |
| replay边界 | 环境、episode ID、terminal和环形写指针约束采样窗口，未见跨回合拼接。 |
| H32 BPTT | 冻结WM/critic参数同时保留状态/动作梯度，存活权重和末端critic bootstrap参与目标。 |
| critic | GAE递推、固定目标、detach输入及5轮×4个打乱小批实际存在。 |
| 动作域 | 真实和想象均clip20；WM拟合执行动作，actor想象使用确定性动作，语义一致。 |
| 模型槽 | 主代理确认每波按各环境上下文重新应用摩擦与COM；未确认模型槽之间的DR串扰。 |

## 已确认的采样偏差

[q1_sim.py:108](../script/supermimic_for_q1/supermimic/q1_sim.py#L108)的`_update_returns`对所有结束回合都从`ret=0`反算采样回报；[第144行](../script/supermimic_for_q1/supermimic/q1_sim.py#L144)除真实终止和动作结束外，还在500步人工截断。机器人若尚未倒地、参考尚未结束，这会把未执行的后续价值记为零，可能错误提高这些时间点的困难采样权重。

固定第三方上游`src/supermimic/envs/mdp/commands.py:222–233`对人工截断用critic bootstrap；bootstrap不可用时，排除紧邻截断的回报样本。主代理已实际读取这段源码，服务器上游Git HEAD与固定提交完全一致。

主代理独立解析PF02原样准入收据，1024段中86段超过500帧，最长1044帧，说明触发条件存在；尚未逐次统计RF02真实触发次数或量化效果影响。该问题位于reset采样回报估计，没有直接改写critic GAE或世界模型的物理终止标签，不能据此宣告本轮方法基线无效。

另登记两项实现差异：本地`bin_memory.add_(soft)`为累计记忆，上游使用EMA；初始bin回报也有差异。论文未明确这些更新细节，本阶段仅登记为实现差异，不直接判定算法错误。PD50/2、25%力矩上限、Q1形态、CPU MuJoCo、64环境和小批量继续按已登记工程适配处理。

## 决定与证据

RF02按既定12000采集轮继续，保持冻结代码和现有终态记录。采样截断修正留给本轮实际评测后的独立下一轮，当前不热改源码、不重启训练，也不根据审阅增加训练准入门。

实际训练源码提交`d66ac1cd21de1682ca6f069227c070ebf98da02f`；本轮三网络从随机初始化开始。按当前循环，完整12000轮预期为768000条真实转移、WM4749组、actor4699次，critic Adam93980步；终态仍须读取实际保存和独立核对，不能把预期数写成已完成。

[取回的原样进度快照](../data/manifests/supermimic_q1_BONES_RF02_review_status_20261025.json)为RUNNING/1625轮/104000真实转移，WM599组/actor549次，训练计时2850.56秒，GPU40摄氏度、容器内存27105280000字节、OOM14。它只代表取回时的观察；恢复时重新读服务器状态与进程身份。

动作长度复算命令：

```bash
python3 -c 'import json; from pathlib import Path; rows=json.loads(Path("data/manifests/supermimic_q1_BONES_PF02_1024_admission_20261025.json").read_text())["clips"]; print(len(rows), sum(r["frames"] > 500 for r in rows), max(r["frames"] for r in rows))'
```

服务器上游目录为`/opt/data/private/ZHY/data/yuanqi_ops/references/supermimic_reproduced_20261022`；执行`git -C <该目录> rev-parse HEAD`及`sed -n '195,245p' <该目录>/src/supermimic/envs/mdp/commands.py`可复核上述固定版本和截断处理。未运行新的动力积分、optimizer更新或CUDA评测。

## 保存与下一步

报告、小型原样快照随本阶段保存到Q1分支；同次提交号以Git历史为准，推送后独立核对远端。大型模型仍只在服务器，LOCAL_ONLY。本次唯一子agent已结束，无子agent继续运行。

继续原有RF02流程：完成12000轮、最终保存核对、同一固定8段train的初始/最终MuJoCo比较与回放。恢复入口为[当前状态](README_CURRENT_STATE.md)和[BONES基线阶段](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)，不要因SSH或观察进程超时重启训练。
