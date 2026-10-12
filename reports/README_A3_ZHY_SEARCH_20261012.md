# A3 新服务器的微调方向对照

记录日期：2026-10-12。独立实验 A3_ZHY_SEARCH_20261012_E02。状态：64完整回放及3冒烟完成，独立核验通过；候选效果门槛全部失败。

## 目标与范围

用户要求迁移及首轮训练后，继续之前未完成的A3探索与训练微调。迁移和R11两次更新已经完成。本阶段在新服务器 `/opt/data/private/hechengzhu/ZHY` 补齐E21参考适配器闭环，并评测新机R11效果。原hp3090不运行训练，旧R01-R10/E10/E20/E21账本保留，Q1与每日研究不变；没有真机操作或驱动/功率/BIOS变更。

## 固定对照与决定

- E21角度MSE与量化前latent MSE两组已各完成200次监督更新、44保存点核验。本轮读取原模型，不重复训练。官方控制器仍冻结。
- 新机匹配基线：CV两档噪声0/0.01rad，各4完整诊断动作；两组adapter相同条件，共16完整动作。另做CV与两组零adapter各10步输入一致性检查。
- 官方035与R11 step2各20完整官方样例，oracle未来参考、30Hz采样/50Hz策略、stride4、相同初始化与零动作延迟。这个对照衡量普通PPO短微调，不与因果adapter条件混为同一排名。
- 共64完整回放和3个10步检查。旧机已有3个adapter动作结果保留为历史；在新硬件上重测是为了同主机的公平对照，避免把平台数值差异当作收益。
- 沿用预定门槛：无新增跌倒，每动作joint RMSE不增加，整体joint至少改善1%；每动作root位置/旋转及躯干/腕/腿世界和anchor-local误差最多增加5%，action delta RMS最多增加5%。adapter必须两档噪声均通过。不随结果放宽门槛。
- 四诊断动作已用于方向筛选，20官方动作也不是新鲜最终测试。这里只能选出当前条件下有证据的候选，不能宣称全局最优、真实衣服输入或机器人效果。

## 版本、执行与保护

入口 `script/a3/zhy_search.py`；独立输入与指标复算 `script/a3/verify_zhy_search.py`。复用固定MuJoCo与官方A3策略，不改vendor。运行前登记模型/数据/代码SHA和真实boot；不允许重置、重复启动或重启自动恢复。

保留CPU90/GPU85摄氏度连续两次5秒采样保护、容器可用内存估计保留4GiB、OOM与新增关键内核异常停止。容器实际16GiB，不能使用宿主可见251GiB作为预算；每动作最多900秒、全阶段最多5400秒。缺可靠遥测停止。所有计算串行，首故障后不继续升档或重复。

```bash
cd /opt/data/private/hechengzhu/ZHY
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m unittest script.a3.test_zhy_search -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_search register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_search run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_search verify
data/environments/a3-sonic/bin/python -m script.a3.zhy_search report
```

产物 `data/experiments/a3_zhy_search_20261012_E02/`；日志 `logs/a3_zhy_search_20261012_E02/`；小型清单 `data/manifests/a3_zhy_search_20261012_E02.json`。模型、原始输入与回放日志LOCAL_ONLY，只提交代码、报告与小型清单。

## 结果、保存与下一步

首次登记在日志目录已含源码备份时触发FileExistsError，尚未生成job或启动GPU评测。空的输出目录保留为 `_registration_incomplete_01`；登记允许既有日志目录，仍拒绝重用已登记实验。此失败是目录处理问题，不是计算、主机或策略故障。

新机3项选择/覆盖回归测试通过：帧数加权、缺失动作拒绝验收、关节改善不能掩盖身体误差退化。实际运行从北京时间12:21:26至12:52:55；独立核验随后通过。64完整回放、3个10步输入冒烟均完成，无跌倒；独立复算实际因果参考token、源动作SHA、逐帧覆盖及joint/root/body指标，零adapter与CV匹配。没有独立重建本体感知，也没有验收初始参考的因果性或真实输入链路。

| 条件 | 匹配基线joint RMSE/rad | 候选joint RMSE/rad | 相对变化 | 结果 |
| --- | --- | --- | --- | --- |
| 角度MSE，noise0 | 0.064949 | 0.071600 | +10.24% | 失败 |
| 角度MSE，noise0.01 | 0.058558 | 0.067390 | +15.08% | 失败 |
| prequant latent，noise0 | 0.064949 | 0.069820 | +7.50% | 失败 |
| prequant latent，noise0.01 | 0.058558 | 0.069308 | +18.36% | 失败 |
| R11 step2，oracle20 | 0.059060 | 0.059810 | +1.27% | 失败 |

四动作每条件5775策略步；oracle20每条件24304步。均有逐动作关节或身体误差退化，完整失败项保存在result.json和小型清单，门槛未放宽。不同参考条件不能混合排名；噪声条件误差较低也不能单独作为加噪改进证据。当前保留官方控制器及对应CV基线，不升级adapter或R11，不宣称最优方向已经找到。

435个主运行健康样本，同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`；CPU最高49/GPU36摄氏度，GPU功率采样最高115.04W，容器可用估计最低8.05GiB，OOM计数0，无新增关键内核错误。该窗口运行通过不证明长期稳定或旧机器故障根因。

运行代码提交 `33834248787614a0ff67010a2b3576703e65d4dc`；job SHA `aab90ecd9c678b1e3fac91ed5dbcd17e9d0a9bf0bb63c2d625a9d8bb41585856`，完整结果SHA `47795145564ec4864f5d4774f18460c80cc910b01412a5dd7568ac6f327a3685`。清单见 `data/manifests/a3_zhy_search_20261012_E02.json`；大型原始数据留新机LOCAL_ONLY。

下一步先做独立E03零参数更新诊断，补齐旧E10未能采集的同runtime证据：采样输入与存储是否相同、实际训练首minibatch是否复现采样分布、精度/模式/批处理是否有影响。概率合同未可靠前不扩展PPO；adapter失败后也不盲目追加相同监督训练。重要阶段保存README/索引/current state、scoped审计提交推送main并核验远端。
