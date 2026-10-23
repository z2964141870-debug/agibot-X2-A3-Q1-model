# Q1追加预算在线趋势诊断

记录日期：2026-10-24，北京时间。阶段：SM-Q1-D06。状态：只读日志诊断与独立均值重算完成；R03继续原固定预算，尚无最终评测结果。

## 目标与范围

检查加长预算后，已有在线日志是否仍停留在R02B末期水平，为R03最终评测准备诊断依据。只读取R02B与R03已生成的日志，不读取checkpoint张量、不调用策略或动力学、不使用GPU，不修改冻结的两个训练源码目录、参考动作、奖励或预算。D04/D05已完成诊断不重跑，D05B候选仍不接入训练。

## 完成内容与结果

02:35:10读取R02B前3000轮与R03前5750轮的完整日志前缀，分别120/230条，25轮间隔完整，所有原始数值有限，已记录的OOM计数均为14。每个比较窗口有30条单步采样，每条是16个环境在当时控制步的均值；不是对750轮全部转移求均值，也不是整段固定动作评测。

| 来源/本阶段采集轮窗口 | 在线奖励均值 | 在线关节RMSE均值 rad | WM k8归一化误差均值 |
| --- | --- | --- | --- |
| R02B 2251–3000 | 0.072456 | 0.390883 | 0.003860 |
| R03 751–1500 | 0.070683 | 0.376099 | 0.002191 |
| R03 2251–3000 | 0.071879 | 0.382911 | 0.002002 |
| R03 3751–4500 | 0.080184 | 0.305853 | 0.002413 |
| R03 5001–5750 | 0.083835 | 0.250112 | 0.001691 |

R03早期751–1500与后期5001–5750窗口相比，在线关节误差均值下降33.50%，在线奖励均值增加18.61%，WM k8误差下降22.83%。变化不是全程单调，2251–3000的关节误差、3751–4500的模型误差曾回升；后期仍有非零策略梯度。

这些是在变化的参考相位、状态访问和策略分布上的描述性趋势。R03还使用了新replay/episode与seed，因此与R02B末期的差异不能作为匹配因果效应。较低在线关节误差不能证明完整走路、水平路径或全身跟随通过，0.15rad完整评测要求保持原口径。当前没有新位移或最终成功率证据。

## 实际验证与身份

远端聚合使用Python标准库math.fsum；随后使用JavaScript逐条增量均值独立核对150条选定记录、45个均值，最大绝对差1.1102230246251565e-16。再由独立只读SSH命令重新读取指定字节前缀，两份大小/SHA均一致。

| 日志前缀 | 字节数 | SHA-256 |
| --- | --- | --- |
| R02B metrics.jsonl前3000轮 | 156312 | 19071f014085778582b6813cc3c5f00a28cf485fa86feae920465001c4b32585 |
| R03 metrics.jsonl前5750轮 | 345930 | bacf3f14c4b6134c1effa0b1918162aef3a90b0dcb8104099a4e017dfffc5beb |

02:35:58核对原controller762428/start3606081278、worker762431/start3606081380，同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，均非僵尸且身份匹配。R03追加5900/12000轮、94400新交互，累计8900轮；2309组WM/2259次actor更新。冻结源码、输入、运行时核验通过。GPU40摄氏度、空闲16671MiB，容器31071297536字节、OOM14，无终态收据。快照不代表恢复时仍然相同，继续工作须重新核对原作业。

## 版本、命令与产物

02:38:46再次核验同一原作业仍在训练：5975/12000追加轮、95600新交互、累计8975轮，2339组WM/2289次actor，GPU40摄氏度、容器31070232576字节、OOM14；冻结条件通过，无终态收据。此快照及前一次5900轮观察一并保存在小证据中。

分析前Q1提交642fd39141624797cbac8c63618cbf90b524e531；训练冻结发布d5f9476125f2c3c7ed2d7c1e6ac898e1f69d6429不变。服务器为huanyu，根目录`/opt/data/private/YuanQi_Q1`，Python为`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。

原始日志：

- `data/supermimic_for_q1/R02_cachefixed_pair_20261023/R02B_batch128/metrics.jsonl`。
- `data/supermimic_for_q1/R03_B128_extend12000_20261023/run/metrics.jsonl`。
- 实时训练stdout仍在`logs/supermimic_q1_R03_B128_extend12000_20261023/train.log`。

小证据为[完整窗口与独立重算](../data/manifests/supermimic_q1_D06_online_trend_20261024.json)，包含列名、150条选定标量记录、两份原始前缀SHA、独立重算及原训练进程快照。精确远端Python源码、SSH参数和独立JavaScript复算源码保存在[执行命令](../logs/session_records/supermimic_q1_D06_commands_20261024.json)，原生退出均为0，不保存认证数据。

需要复算时，在Q1副本中执行下面的只读命令；SSH采集要求既有授权隧道可用，不因观察失败启动训练。

```bash
node -e 'const fs=require("node:fs"),cp=require("node:child_process");const r=JSON.parse(fs.readFileSync(process.argv[1],"utf8"));cp.execFileSync(r.ssh_executable,[...r.ssh_arguments,r.remote_command],{stdio:"inherit"});' logs/session_records/supermimic_q1_D06_commands_20261024.json
node -e 'const fs=require("node:fs");const c=JSON.parse(fs.readFileSync(process.argv[1],"utf8"));const e=JSON.parse(fs.readFileSync(process.argv[2],"utf8"));console.log(JSON.stringify(Function("snapshot",c.independent_js_source)(e.snapshot),null,2));' logs/session_records/supermimic_q1_D06_commands_20261024.json data/manifests/supermimic_q1_D06_online_trend_20261024.json
```

## 关键决定与下一步

保持R03原追加12000轮、累计15000轮预算及保护。在线指标已有改善迹象，仍由原流程完成保存审计、step0/final完整动力学评测与独立轨迹重算。然后执行V02实际渲染，对比实际移动、腿部和全身误差及画面；不能用在线窗口替代此流程或据此宣告可用。

本诊断不新增物理积分、策略推理、NN优化器更新或GPU渲染。大型模型/轨迹仅服务器LOCAL_ONLY；Mac/Git只有小证据和报告。本阶段记录按Q1分支审查、payload audit后提交推送，并独立比对远端；完整提交号由Git历史提供，尚未完成的推送不能记作通过。
