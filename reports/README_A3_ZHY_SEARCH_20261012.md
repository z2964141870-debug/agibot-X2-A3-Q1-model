# A3 新服务器的微调方向对照

记录日期：2026-10-12。独立实验 A3_ZHY_SEARCH_20261012_E02。状态：评测预登记准备，尚未运行。

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

新机3项选择/覆盖回归测试通过：帧数加权、缺失动作拒绝验收、关节改善不能掩盖身体误差退化。实际评测尚未执行，不把代码准备记为效果通过。完成实际运行后独立检查完整帧覆盖、真实参考token及joint/root/body指标，保存包括失败项的结果，再决定下一轮单变量微调。若adapter失败，优先检查训练标签与实际机器人anchor的分布差异以及官方行为保持约束，不盲目追加PPO长训。重要阶段保存README/索引/current state、scoped审计提交推送main并核验远端。
