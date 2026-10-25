# Q1每轮训练的六宫格曲线

记录日期：2026-10-25，北京时间。阶段：RF02-V01。状态：标准工具、未来回合记录入口和真实预览已完成；本轮结束回调正在等待，RF02训练及最终评测仍进行中。

## 目标与用户要求

用户要求后续每轮训练结束都生成与示例相似的2×3训练曲线。采用细线原始记录、粗线500个训练iteration的滑动平均，支持多实验叠加；标注真实环境数、actor/WM批量、已完成与计划轮数。输出PNG、PDF、CSV和来源/缺项/平滑方法收据，不套用示例的1024环境/1024批量预算。

此要求已增补至权威副本、用户镜像和服务器各自的AGENTS，保留三份原有差异。本次新增源码只写`script/project_tools/`；当前RF02训练相关68份冻结Python源码逐SHA保持，没有热改训练、重启worker或新增数据门。

## 指标定义与实际可用性

| 面板 | 数据与含义 | 当前RF02 |
| --- | --- | --- |
| Episode normalized return | 对每个已完成回合计算`sum(reward)/(实际长度×max_step_reward)`，再对回合求均值；无折扣 | 历史未记录，明确留缺 |
| Hard terminations / completed episodes | `硬终止回合数/(硬终止或timeout回合数)`；二者同时为真仍计硬终止 | 历史未记录，明确留缺 |
| Completed episode length | 已完成回合的实际控制步数均值；未完成回合不算短回合 | 历史未记录，明确留缺 |
| Simulator step reward / maximum | `reward_mean/max_step_reward`，本轮每25轮采样一次 | 已有 |
| World model error at step 8 | `wm/nse_kstep`，绑定WM窗口8；自回归第8步的归一化状态平方误差，再平均训练小批 | 已有 |
| Imagined survival | `imag_survival_H`，H32想象轨迹末端模型存活权重的均值 | 已有 |

本轮`termination_fraction`只是每25轮采样时该步64个环境的硬终止比例，不能恢复完整回合数、累计长度或回报，不能替代第二项。没有原始回合数据，不从模型或曲线推算前三项。

单步奖励最大值来自`TrackingRewardCfg.max_step_reward`，源码为`dt × 六项正奖励权重之和 = 0.02 × 5 = 0.1`；源文件SHA`c23a3336865b75e098e67eb58af4e7e25b1a679ace9dc0c67882af6eb6149fd3`。本轮绘图显式传入0.1和原源码路径；未来入口记录实际`heads.reward_cfg`及运行时最大值，不静默套固定常数。

完整回合指标针对随机起点和训练DR，不能代替固定起点整条参考的完成率/归一化回报。世界模型误差取训练小批，不是独立测试误差；想象存活率是模型预测，不是真实动力学存活率。

## 实现与真实验证

新增工具：

- `script/project_tools/plot_training_dashboard.py`：只读日志，不加载权重，不创建CUDA上下文；导出六宫格、数值CSV及来源收据。缺项显示`Not recorded`，不补零或插值。
- `script/project_tools/train_q1_with_dashboard.py`：供下一轮训练使用，在独立进程中包装现有Q1Sim入口，逐控制步只读奖励/terminated/timeout，另写`episodes.jsonl`；现有训练循环、采样、损失与optimizer保持原方法，结束或有日志的异常退出后调用绘图。
- `script/project_tools/test_training_dashboard.py`：计数、跨日志窗口累积、双重终止标记、按iteration平滑以及可选真实MuJoCo不变性控制。

滑动窗口为`(当前iteration - 500, 当前iteration]`，不是500行日志。前三项用窗口内已完成回合数加权；其他三项对窗口内已有日志点求均值。起始不足500轮时使用已有部分，原始点同时保留。没有完成回合的窗口不伪造均值。

Mac四项纯计数控制通过；服务器五项全部通过，耗时4.863秒。真实控制使用同一Q1资产/同一冻结参考、CPU MuJoCo、2环境×5步、episode_steps=2，分别运行原sim与记录包装：逐步state、reward、frames、terminated、timeout逐项完全相同，最终torch RNG相同，实际完成回合可正确记录。两次合计20条额外物理控制转移、0策略训练/0optimizer更新；它不计入RF02训练预算。控制关闭DR/观测噪声，验证范围是记录包装的读操作，尚未以新入口启动另一轮完整训练。

服务器运行测试：

```bash
env PYTHONDONTWRITEBYTECODE=1 YUANQI_DASHBOARD_TEST_ROOT=/opt/data/private/ZHY \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/data/private/ZHY/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python \
  -m unittest discover -s /opt/data/private/ZHY/script/project_tools \
  -p test_training_dashboard.py -v
```

## 本轮预览与结束回调

16:20:45已真实生成截止4325/12000轮的预览。图面已目视检查：2×3布局、细/粗曲线、图例、预算文字及缺项均清楚，没有遮挡。它不是最终训练图，也不能据此宣称固定参考跟随改善。

服务器：`data/supermimic_for_q1/BONES_RF02_dashboard_preview_20261025/`。原样[预览收据](../data/manifests/supermimic_q1_BONES_RF02_dashboard_preview_20261025.json)登记PNG304070字节、PDF33060字节、CSV48808字节；PNG SHA`b7c909f6e9a2449308681ffba274592033f849bbaf3253dcc7f40a58dc4dc9f8`。Mac仅取304KB预览图，小图不进入Git。

16:25:54已启动本轮唯一的结束绘图回调：[原样launch](../data/manifests/supermimic_q1_BONES_RF02_dashboard_launch_20261025.json)，PID868333/start_ticks3620875959。它绑定现有worker862408/start_ticks3620018050及boot，每30秒读取一次状态，等待本轮terminal receipt，届时自动出图；24小时上限、身份变化报错退出，无启动/重启训练能力。日志`logs/BONES_RF02_training_dashboard_20261025.log`，最终图目录`data/supermimic_for_q1/BONES_RF02_training_dashboard_20261025/`。16:27实际读到回调处于`WAITING_FOR_ONE_TRAINING_RUN`，不是仅凭Popen成功宣称已运行。

本轮回调的精确command保存在launch，不要重复启动。未来需要手动终后绘图时用新的output；叠加多实验可给`--runs <run1> <run2>`和同数量`--labels`，各轮预算分别显示。下一轮训练在原资源guard内使用以下入口，并传递与`train_reproduction.py`相同的其他训练参数：

```bash
<project-python> /opt/data/private/ZHY/script/project_tools/train_q1_with_dashboard.py \
  --root /opt/data/private/ZHY --output <独立新实验/run> \
  --dashboard-label <实验名> <原训练参数>
```

新入口默认将完整图保存至`<独立新实验/run_dashboard>/`，与模型分开但仍在同一实验内。它不能绕过原磁盘/内存保护；下一轮编排应将它作为原guard的worker入口。当前已运行的冻结控制器不改。

## 运行状态与版本证据

16:27:13[原样观测](../data/manifests/supermimic_q1_BONES_RF02_dashboard_attachment_20261025.json)：RF02为4525/12000轮、289600真实转移，WM1759组/actor1709次，pipeline/guard/training均RUNNING。四进程boot/start_ticks均匹配；GPU40摄氏度、共享内存25297174528字节/OOM14，训练目录分配893853696字节，盘余2077460029440字节。旧4000自动保存身份亦在该收据登记，没有新增CPU张量审计。

| 新工具 | SHA-256 |
| --- | --- |
| plot_training_dashboard.py | `70f03b41529832c4b582550424195bc9df3477b79db58791ee58e5da45a19e7b` |
| train_q1_with_dashboard.py | `dcfbb728c6da9d0c6206712d7b0e174a744f1ad7615f2b139769888e3a3563e8` |
| test_training_dashboard.py | `9481cda9164b80e998e46814f8ee854cc57320e81bc9ccfcb9b36433b68af8f7` |

源码在Mac与服务器逐SHA对应；原RF02训练版本仍为`d66ac1cd21de1682ca6f069227c070ebf98da02f`。本阶段工具/报告/小收据随Q1阶段提交保存，精确提交号由Git历史给出；发布后独立比对远端Q1。权重和完整图留服务器，LOCAL_ONLY；没有声称网盘备份。

## 下一步与恢复

先确认当前训练身份和`run/status.json`，继续原定12000轮，不因SSH/观察连接结束重启。终后检查上述callback日志、最终图receipt和图面，再完成既有固定8片初始/最终动力学比较、回放与Git记录。图表要求是后续交付的一部分，没有新训练准入/成功阈值。本轮前三项仍因历史未记录而缺失，后续新实验使用回合记录入口补齐。
