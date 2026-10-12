# A3 真实首训练入口零更新诊断

记录日期：2026-10-12。独立实验 A3_ZHY_ENTRY_20261012_E08。状态：实现与3项CPU验收完成，真实运行待执行；尚未修复概率偏差。

## 目标与依据

用户要求通过代码修正采样/训练不一致。E05实际首训练小批次相对采样mean的RMSE为0.010159，初始ratio仍明显偏离1；E06/E07离线重建、直接Actor、原PPO包装器和Accelerate均能复现采样，不能归因于包装器、梯度开关或未经验证的权重提前改变。本轮捕获真实首Actor调用，以确定差异发生在输入、Actor计算还是分布包装阶段。

## 实现与固定合同

入口 `script/a3/zhy_entry_probe.py`，诊断trainer `script/a3/zhy_entry_trainer.py`。沿用官方035、16环境、24采样步、4小批次、seed0和E05的一致float32/TF32关闭；实际首batch进入Actor时记录输入值、shape/stride/dtype、调用选项、梯度/自动混合精度、随机状态、参数/buffer摘要和各模块train/eval状态，以及实际启用的编码器/解码器配置。

保存原始Actor输出和包装器mean/std/logprobs，并在同一个Actor即时比较同输入同选项、存储输入同选项、存储输入去episode mask三组。各组恢复首调用随机状态和精度上下文。首loss入口立即保存并退出，禁止backward和optimizer更新；只接受初始化时既有std钳位，不把Critic归一化buffer变化误判为Actor优化。

仅在新机 `/opt/data/private/hechengzhu/ZHY` 执行；原hp3090不计算。需E07同boot核验、官方模型SHA与固定代码一致、GPU空闲且遥测可靠。CPU90/GPU85摄氏度连续2次5秒、4GiB内存保留、900秒上限、OOM/新boot/关键内核错误/遥测缺失停止；不重置、复跑或开机恢复。原概率门槛不变：mean RMSE不大于1e-5，ratio最大偏离1不大于0.001。

CPU测试检查完全一致时通过、输入改动即使输出相等仍拒绝、权重稳定不能掩盖输出偏差；CPU独立重载复算捕获与采样存储身份。诊断增加同步CPU拷贝和即时回放，会改变时序，不能当成未经诊断修改的完整训练验收。

## 版本、命令与产物

固定vendor `fe6868ba37034f89b912f0fb851bce19f120266d`，不修改vendor。基于工程main `283457a1da727c4d5b8d62f8e33c15d4f63312b4`；执行版本由注册文件和Git历史登记。

```bash
cd /opt/data/private/hechengzhu/ZHY
PYTHONPATH=.:script/vendor/sonic_for_a3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m unittest script.a3.test_zhy_entry_probe -v
data/environments/a3-sonic/bin/python -m script.a3.zhy_entry_probe register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_entry_probe run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_entry_probe verify
```

输出 `data/experiments/a3_zhy_entry_20261012_E08/`，日志 `logs/a3_zhy_entry_20261012_E08/`，小型清单 `data/manifests/a3_zhy_entry_20261012_E08.json`。实际输入、随机状态和张量只在新机LOCAL_ONLY；Mac/Git仅小型代码、报告和清单，凭据不落盘。

## 实测与边界

Mac静态编译通过，新机3项CPU证据测试通过（4.181秒）。新机只读核验仍为boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`，GPU26摄氏度/约33W/0%利用率，无计算进程。E08注册/执行/独立核验尚未完成，不将代码存在写成修复成功。

让同输入同权重在两条路径上的概率一致，是优化前的计算验收；这不保证PPO每轮效果上升，也未证明历史退化的唯一原因，更不解释旧主机重启。

## 保存与下一步

本README与索引/current state先保存诊断合同，执行前scoped审计提交并推送main、核验远端。下一步执行CPU检查和独立E08，依据实际入口证据实现针对性修正，再做同门槛验证与有界训练效果对照。
