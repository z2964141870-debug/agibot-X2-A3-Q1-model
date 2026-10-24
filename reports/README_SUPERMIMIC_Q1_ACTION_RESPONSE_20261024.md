# Q1 R03世界模型动作响应诊断

记录日期：2026-10-24，北京时间。阶段：SM-Q1-D08。状态：两端2项控制、真实2176控制步/43520物理子步、192项响应与独立数组审计完成；支持登记H8/H32匹配试验，移动目标未达。

## 10:22至10:24实际结果

源码发布`e087cd5c2ff056dfcfe6d03ad879b33b5175a393`已推送并独立核对Q1。新增入口与测试及D07小记录共11文件大小/SHA逐项一致；两端各2项控制原生退出0、无跳过。真实prepare于10:22:10完成，计划29095字节/SHA `d67b6f54c92594f4d6c471cf048be41818748ec829db280101d8f655afcaa8ad`；四状态为走路405/1053与浅蹲281/731帧。

10:22:55至10:24:00真实execute退出0，动力学/模型诊断耗时64.526秒。4个基线与原轨迹逐元素完全一致，16组float64 JVP/有限差分及float32/64对照全通过；前者最大绝对差2.269e-9。全部192项响应有效、未替换或重试。独立数组审计通过，统计重算最大绝对差2.221e-16，4份数组各544控制步合计2176。

以下根位置/关节响应是两个动作幅度下首步扰动的局部导数，并非策略成功率；同一方向两个幅度也不是独立随机样本。各行共32项（4状态×4方向×2幅度）：

| 时域/部位 | 正方向项数 | 余弦中位数 | 模型/物理幅度中位数 | 相对L2误差中位数 |
| --- | ---: | ---: | ---: | ---: |
| H1 根位置 | 32/32 | 0.9588 | 0.8705 | 0.4060 |
| H1 关节 | 32/32 | 0.9952 | 0.9984 | 0.0991 |
| H8 根位置 | 32/32 | 0.9813 | 0.9533 | 0.3278 |
| H8 关节 | 32/32 | 0.9487 | 0.9419 | 0.3325 |
| H32 根位置 | 27/32 | 0.9818 | 0.9729 | 0.4265 |
| H32 关节 | 30/32 | 0.9401 | 0.9327 | 0.4214 |

分片看走路，根位置H8为16/16正向、相对误差中位0.4412；H32为11/16正向、相对误差0.7595。走路关节H8为16/16正向、H32为14/16；浅蹲全部时域两部位均正向。H32根位置的总余弦中位数仍高，不能概括为模型处处反向或完全失效；方向/幅度误差和D07长时根预测缺口共同支持试验短时域。

这仍是4个已访问状态的首步动作开环诊断，未检查闭环actor反馈、全奖励梯度或更大状态分布；不能推断H8必然产生移动。下一登记独立R04：从同一R03 final、相同fresh replay种子与等交互预算比较H32/H8，仅更改policy imagination horizon，保留奖励、WM k8、网络/优化器与输入。必须各自完整评测/数组重算后判断，不能只把H8追加训练与原R03不同预算直接对比。

receipt142297字节/SHA `164f2ddc7f0956b63adc881b711a4da7b12612d696605922b6da3be0825caaac`；audit3834字节/SHA `bd922c81bb083547a8098900dab103b73bece6c508fa65e2178e5cf77279c9ea`。计划、admission、receipt、audit、completion及三日志共8小文件大小/SHA与服务器一致，清单`data/manifests/supermimic_q1_D08_R03_execution_20261024.json`；Git小结果前缀`supermimic_q1_D08_R03_*`，日志`logs/session_records/supermimic_q1_D08_R03_*`。4份数组仅留服务器，其大小/SHA已在receipt与audit登记，不取回Mac。

原科学冻结条件执行前后通过。10:26:21退出后实核GPU38摄氏度/空闲17788MiB、容器25434927104字节/OOM14；诊断结束前采样容器27198783488字节，均无新增OOM。0策略推理/optimizer。新增为已声明物理积分和世界模型前向/JVP。大原件LOCAL_ONLY；本阶段结果先发布Q1，再进入R04准备。

## 以下为执行前登记

## 目标与范围

继续用户授权的SuperMimic for Q1世界模型复现。R03累计15000轮后，走路全身误差改善但最大水平离起点仍只有0.029120m、参考1.126480m。D07显示最终模型走路H32根坐标预测RMSE41.371mm、恒速基线9.665mm；预测误差不能证明动作梯度方向错误。本阶段直接比较同一已访问状态下，真实动力学有限差分与最终世界模型JVP/有限差分的方向和幅度。

只新增局部动力学响应和既有模型前向/自动微分，无策略推理、optimizer或训练。冻结R03、原参考与两个科学源码目录不改写；A3保持停止。不是公共状态下父/子模型因果对照，不声称完整actor梯度正确或已经学会行走。

## 预先固定的预算与判读

- 只用最终匹配评测的两个training片段：走路、浅蹲；各取25%/65%位置，共4个初态，不按结果替换。
- 每状态4个固定随机方向，种子20261023；归一化动作幅度0.05/0.10及正负方向，只扰动第一个动作，后31步固定原评测动作。
- 每状态1个未扰动基线和16个扰动，共最多2176控制步、43520物理子步。先复现全部4基线，再运行扰动。
- 读出H1/H8/H32，即0.02/0.16/0.64秒；根位置与22关节位置分别报告方向余弦、正方向比例、幅度比和相对L2误差，低于既有阈值的弱响应保留并标注。
- 自动微分另用float64有限差分与float32/64对照；保存全部4份数组，独立从数组重算192条有效性/响应及6项汇总。实际有效数以结果为准。
- 复用原协议：基线执行门为逐元素既有容差；独立审计进一步要求与原保存轨迹完全一致。若任一门失败，保留失败终态，不放宽门槛或自动重试。

局部开环结果仅用于选择后续单变量试验，正方向比例不是策略成功率。根响应可能很弱，不能只看余弦忽略幅度。H8对H32当前只是候选，不在本阶段准备或启动R04。

## 实现与已完成控制

新增`script/supermimic_q1_analysis/probe_completed_response.py`，复用冻结的`probe_action_response.py`/`audit_action_response.py`，仅在本进程绑定`ContactCorrectedAssets`，finally恢复构造器。读取已完成R03、匹配评测审计、checkpoint、reference、trace与新接触admission；先后核验原计划冻结条件，绑定新增入口及两个analysis依赖SHA。

输出使用独立目录，拒绝覆盖/重复admission。prepare与execute前后检查OOM计数不增、容器30GiB线至少896MiB余量、GPU空闲至少2GiB及温度低于86摄氏度。实际资源以本次执行为准，不沿用D07快照。

本地执行原生退出0、2项通过：构造器成功恢复、抛错时同样恢复。该控制不加载真实模型，不证明动作响应科学结果。服务器同控制与真实执行尚未进行。

## 版本、命令与产物

Mac权威副本`/tmp/yuanqi-supermimic-q1-20261023`、分支Q1；服务器`huanyu:/opt/data/private/YuanQi_Q1`。输入科学冻结发布`d5f9476125f2c3c7ed2d7c1e6ac898e1f69d6429`，D07结果已发布并核对`16d2c97d4b66ea857671fe945d768006cfd0ae3c`。新增代码版本由本阶段Git历史和执行SHA标识。

Python为`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`；从服务器项目根运行：

```bash
python3 -m unittest discover -s script/supermimic_q1_analysis -p test_probe_completed_response.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$PY" -u script/supermimic_q1_analysis/probe_completed_response.py prepare --root "$PWD" --experiment data/supermimic_for_q1/R03_B128_extend12000_20261023 --output data/supermimic_for_q1/D08_R03_action_response_20261024 --device cuda:0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$PY" -u script/supermimic_q1_analysis/probe_completed_response.py execute --root "$PWD" --experiment data/supermimic_for_q1/R03_B128_extend12000_20261023 --output data/supermimic_for_q1/D08_R03_action_response_20261024 --device cuda:0
```

计划原件`data/supermimic_for_q1/D08_R03_action_response_20261024/plan.json`；实际执行应新增`admission.json`、`sample_00..03.npz`、`receipt.json`、`summary_audit.json`及`completion.json`。日志`logs/supermimic_q1_D08_R03_action_response_20261024.log`。执行前未创建这些科学产物，不能将本地控制称为诊断完成。

## 保存与下一步

本次先审核diff/Git payload、提交并显式推送Q1、独立核对远端，再部署新入口和此前D07小记录，核对大小/SHA，执行服务器控制及prepare。检查真实计划后仅execute一次；采集原生退出、健康和完整/失败证据，立即更新本报告及两个入口并同步Q1。大型checkpoint/轨迹/响应数组留服务器LOCAL_ONLY，Mac只取小证据，不声称网盘备份。
