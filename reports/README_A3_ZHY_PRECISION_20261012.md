# A3 采样与训练一致精度对照

记录日期：2026-10-12。独立实验 A3_ZHY_LIKELIHOOD_20261012_E05。状态：完成0更新对照并独立核验；实际小批次概率合同仍失败。

2026-10-12后续更正：E08证明真实训练入口仍启用BF16自动混合精度。E05只关闭了TF32，未关闭该入口的autocast，因此本轮不能称为采样/训练完整计算精度一致。原数值及失败账本保留，见[真实入口证据](README_A3_ZHY_ENTRY_20261012.md)。

## 目标与依据

E04已在同一运行Actor、相同输入、0更新条件下复现初始概率不一致；TF32开启时原采样形状逐帧精确复现，整批/首minibatch输出不同。输入错位与保存分布错误本次已排除，train+aux/eval在同精度整批输出相同。见[有效诊断](README_A3_ZHY_LIKELIHOOD_20261012.md)。

本轮单一主要变量是采样开始前同时禁用matmul/cudnn TF32，并保持到首实际训练前向。不是只在旧rollout回放时改精度，不改变奖励、权重、数据、seed或优化器目标；仍禁止任何optimizer.step。当前没有具体FSQ层归因或微调收益证据。

## 固定合同与实现

新机 `/opt/data/private/hechengzhu/ZHY`，官方035、16环境、24步、4minibatches、seed0。加载权重后显式设置两个TF32开关false，保存覆盖前后状态；sampling24次与首实际前向的标志必须均false。固定代码/模型SHA，需E04 VERIFIED且概率合同失败、输入一致才允许登记。

在首真实minibatch之外，用训练器原 `_get_mb_rollout_data` / `_forward_model` 补算其余12环境，分3个不重叠的4环境minibatch；这是补充覆盖，**不是原训练loop后续的随机排列**。四批覆盖每个环境恰好一次，保存每批mean/std/indices并独立CPU重载复算。原完整GPU/CPU/模式/逐帧/整批变体也保留，跨精度回放不替代实际合同。

门槛不变：输入及采样mean与存储精确一致，四批动作mean RMSE≤1e-5、ratio最大偏离1≤1e-3、0optimizer；Actor唯一允许的初始变化为官方std限幅。运行900秒上限，无重置、重复、开机恢复；温度CPU90/GPU85连续2次5秒、内存保留4GiB、OOM/新boot/内核错误/遥测丢失停止；所有计算串行。

入口 `script/a3/zhy_likelihood.py --experiment E05`，新trainer `script/a3/zhy_precision_trainer.py`，四批覆盖CPU检查 `test_zhy_precision.py`。固定vendor只读，旧E03/E04/E10账本保留。

```bash
cd /opt/data/private/hechengzhu/ZHY
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m unittest script.a3.test_zhy_precision -v
data/environments/a3-sonic/bin/python -m script.a3.zhy_likelihood register --experiment E05
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_likelihood run --experiment E05
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_likelihood verify --experiment E05
```

## 产物、结果与下一步

输出 `data/experiments/a3_zhy_likelihood_20261012_E05/`，日志 `logs/a3_zhy_likelihood_20261012_E05/`，小型清单 `data/manifests/a3_zhy_likelihood_20261012_E05.json`。模型/采样张量/原始日志留新机LOCAL_ONLY，Mac与Git只保留代码、报告、清单。

新机2项CPU测试通过：16环境完整覆盖时独立概率复算、重复环境拒绝验收。实际13:30:50–13:31:42完成，随后CPU独立重载/全部12概率行/4小批次/输入identity核验通过，state=VERIFIED，optimizer=0。

采样与存储输入/mean精确一致，24次采样与实际首训练前向均两项TF32=false。采样保存、实际首minibatch及补充三批捕获SHA均核验；唯一初始参数变化仍为官方std限幅。

| 对照 | mean RMSE | ratio范围/最大偏离 | 结论 |
| --- | --- | --- | --- |
| 首实际4×24 minibatch | 0.010159 | 0.534225–1.300595 / 0.465775 | 失败 |
| 补充batch1 | 0.010681 | 0.726297–1.435033 | 失败 |
| 补充batch2 | 0.009253 | 0.657941–1.941035 | 失败 |
| 补充batch3 | 0.010686 | 0.663328–1.306490 | 失败 |
| 同Actor CUDA完整精度16×24直接前向 | 4.04e-7 | 最大偏离1.91e-5 | 这个回放变体通过 |
| 同Actor CUDA完整精度16×1逐帧 | 0 | 最大偏离2.86e-6 | 原采样形状复现 |
| 同Actor CPU完整精度16×24 | 5.34e-7 | 最大偏离3.05e-5 | 这个回放变体通过 |

完整精度使16环境整批与逐帧接近一致，但**没有修复4环境实际训练前向**，不能将TF32关闭写成修复成功。合同按实际4批判定失败，没有放宽门槛；尚不能区分4环境计算本身和训练调用路径/输入处理的影响。

12个去重健康样本同boot，CPU最高40/GPU34摄氏度，可用估计最低5.56GiB、OOM0、无新增关键内核事件。运行版本 `3aeb3a3854227eabf547419da4c817f5171cd222`；捕获SHA `b8c14e4238c7f647fe8d8553613f559c7abd108d9957984d37f999c079cb379a`；完整结果在E05小型清单，原始张量LOCAL_ONLY。

下一步只重放已有捕获，使用保存的runtime配置重建Actor且先要求能复现16环境基准，再对相同4环境做直接前向、逐帧及训练结果对照。若重建不能复现，标记证据不足，不能重复E09发布配置误配后的根因误判。训练仍暂停，不登记更长PPO。

每重要阶段立即写README、索引/current state与小型SHA清单，scoped审计提交推送main并核验远端；原机不计算，不涉及Q1、真机、驱动/功率/BIOS操作。
