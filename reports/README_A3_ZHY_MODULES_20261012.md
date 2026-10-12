# A3 保存策略模块隔离与量化变化

记录日期：2026-10-12。独立实验 A3_ZHY_MODULES_20261012_E12。状态：3项CPU检查、6条件离线前向与独立CPU复算通过；encoder与decoder都能改变同输入动作，仍未证明完整闭环退化的唯一原因。

## 目标与范围

E11确认修正FP32概率仍一致，主PPO推动Actor且R12的encoder/decoder都被更新。用户继续排查的本轮问题是：保存策略的哪一部分会在同一输入上放大动作漂移，量化token是否参与？

复用E11已核验的384帧输入、E09运行时模型元数据和R12初始/更新1/更新2保存点。只做6种离线前向：初始、更新1、更新2、只换更新2encoder、只换更新2decoder、只换更新2std。无新回放、optimizer或仿真，不生成新策略供部署。原hp3090不计算，固定vendor只读。

## 实现与决定

`script/a3/zhy_module_probe.py`通过现有Actor构造函数和严格state加载，每次按E11相同四批前向。记录实际拼接的量化token和动作mean，不以训练时可微近似替代推理结果。

先用初始Actor重建E11采样mean/logprob，维持mean RMSE<=1e-5、ratio偏离<=0.001；失败就不允许归因。std只换不得改变确定性动作mean，decoder只换不得改变encoder token；所有前向参数SHA必须不变。CPU独立复算动作漂移和token变化条件统计。

CPU90/GPU85连续两次5秒、4GiB余量、同boot、遥测/OOM/内核异常停止沿用，无自动恢复或重复启动。新机仅串行运行。

原始机制依据：[FSQ论文](https://arxiv.org/abs/2309.15505)说明连续表征映射到有限标量集合；[PPO论文](https://arxiv.org/abs/1707.06347)描述采样后优化surrogate。论文机制不能直接证明本地退化，本实验通过真实Actor权重交换检验。已核对固定源码的encoder→FSQ→decoder与`_last_full_latent_flat`。

## 版本、命令与产物

基于main `9742f77f739e95924e4234e9916290e3f421208a`；vendor `fe6868ba37034f89b912f0fb851bce19f120266d`，登记固定代码/输入SHA。

```bash
cd /opt/data/private/hechengzhu/ZHY
PYTHONPATH=.:script/vendor/sonic_for_a3 OMP_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m unittest script.a3.test_zhy_module_probe -v
data/environments/a3-sonic/bin/python -m script.a3.zhy_module_probe register
PYTHONPATH=.:script/vendor/sonic_for_a3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_module_probe run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_module_probe verify
```

实验`data/experiments/a3_zhy_modules_20261012_E12/`；日志`logs/a3_zhy_modules_20261012_E12/`；清单`data/manifests/a3_zhy_modules_20261012_E12.json`。实际运行main `ff4eb0c5105f111536fceb1c8289bc7fa5d27933`，启动前推送及远端核验一致。原始输出新机LOCAL_ONLY，Mac只留小型代码/报告/清单。

## 结果、边界与下一步

3项CPU检查通过（0.028秒），覆盖模块交换范围、缺失模块/不匹配state拒绝、变化token帧的条件统计。北京时间18:33:33–18:33:54完成六条件离线前向，0optimizer、0新回放。初始重建通过原mean/probability门槛，各条件前向参数SHA不变；独立CPU重算384帧全覆盖、动作/token变化和初始概率，state为VERIFIED。

| 保存策略或交换条件 | 原始动作mean RMSE（相对初始） | 最大绝对动作差 | 改变token的帧数 | 改变token坐标 |
| --- | --- | --- | --- | --- |
| 初始 | 0 | 0 | 0/384 | 0/24576 |
| 更新1 | 0.0247774 | 0.219634 | 346/384 | 829/24576 |
| 更新2 | 0.0380346 | 0.295930 | 378/384 | 1436/24576 |
| 只换更新2 encoder | 0.0193054 | 0.153825 | 378/384 | 1436/24576 |
| 只换更新2 decoder | 0.0277995 | 0.273355 | 0/384 | 0/24576 |
| 只换更新2 std | 0 | 0 | 0/384 | 0/24576 |

这是策略原始动作输出，不能直接作为关节rad或闭环关节RMSE。update2与encoder-only的token变化位置完全一致，378/384即98.4375%帧至少一维token改变，全部坐标只改变5.8431%。encoder-only时6个token未变帧动作mean差为0；这批固定输入下，encoder变化经离散token影响动作。decoder-only时token保持精确不变但384帧动作都漂移，动作RMSE0.02780；说明动力学decoder变化也有独立影响，不能只归因FSQ。

额外CPU精确等价检查通过：encoder-only token精确等于完整update2、decoder-only token精确等于初始、std-only mean精确等于初始，保存`data/experiments/a3_zhy_modules_20261012_E12/module_swap_contracts.json`。上述“相同token变化”由完整张量等价支持，不只依赖数量相等。

update2完整动作RMSE大于单独换encoder/decoder；两部分非线性组合，不能把上述RMSE相加或当作各自闭环退化贡献百分比。std-only不改变确定性mean是预期结果，但随机训练时探索std仍影响采样，不能据此认定std完全无关。

6健康样本同boot，CPU最高41/GPU31摄氏度、采样功率最高111.99W、可用估计最低7.92GiB、OOM0，无新关键内核错误/遥测失败。样本18:33:01–18:33:54，最长31.50秒间隔在登记至启动前；不宣称全时连续覆盖。任务结束，无训练或仿真进程继续运行。

动作mean漂移不等于闭环误差；替换模块证明的是固定输入下的影响，不能宣称找到所有训练退化的唯一凶手。六种条件同一批输入且来自已使用诊断，不能当作泛化/真机验收。

下一项优先保留官方行为的单变量约束/冻结对照，而非继续仅缩小LR；encoder和decoder都独立产生漂移，不能未经对照就承诺只冻一个即可修好。仍需区分小样本优势噪声、训练目标与评测不一致、合法参数更新造成基础技能损失，按实际缺陷选择一个主要变量。旧R12不升级，不重复完整训练。

## 保存与恢复

启动前保存README/索引/current state并审计scoped代码；结果已保存小型清单，按main检查diff、payload审计、scoped提交推送并核验远端。原始输出新机LOCAL_ONLY，未验证网盘备份。恢复先读本文件和E12 state/log，状态VERIFIED，已启动目录不重置。
