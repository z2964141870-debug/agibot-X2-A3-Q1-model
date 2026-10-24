# Q1 R03世界模型动作响应诊断

记录日期：2026-10-24，北京时间。阶段：SM-Q1-D08。状态：入口与本地两项构造器控制通过；真实计划登记、服务器控制及动力学执行尚未进行。

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
