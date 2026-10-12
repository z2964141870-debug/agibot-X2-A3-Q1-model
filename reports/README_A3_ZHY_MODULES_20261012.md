# A3 保存策略模块隔离与量化变化

记录日期：2026-10-12。独立实验 A3_ZHY_MODULES_20261012_E12。状态：已实现，待CPU检查和实际运行；尚无模块归因结果。

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

实验`data/experiments/a3_zhy_modules_20261012_E12/`；日志`logs/a3_zhy_modules_20261012_E12/`；清单`data/manifests/a3_zhy_modules_20261012_E12.json`，目前尚未生成。原始输出新机LOCAL_ONLY，Mac只留小型代码/报告/清单。

## 结果、边界与下一步

待核验。动作mean漂移不等于闭环误差；替换模块证明的是固定输入下的影响，不能宣称找到所有训练退化的唯一凶手。六种条件同一批输入且来自已使用诊断，不能当作泛化/真机验收。

下一项由实测决定冻结哪部分或加入参考约束的单变量对照；不把少更新参数或较小动作漂移当成策略效果通过。旧R12不升级，不重复完整训练。

## 保存与恢复

启动前保存README/索引/current state并审计scoped代码；实际状态与结果随后记录。GitHub同步以远端提交核验为准，未执行不标成通过。恢复先读本文件和E12 state/log，已启动目录不重置。
