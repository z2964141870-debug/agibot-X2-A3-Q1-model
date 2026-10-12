# A3 显式完整精度修正与验收

记录日期：2026-10-12。独立实验 A3_ZHY_CONSISTENCY_20261012_E09。状态：针对性修正已实现，8项CPU回归通过；真实零更新验收待执行。

## 目标与原因证据

修正同输入、同Actor权重、零优化器更新时采样与首训练概率不一致。E08确认输入/权重/buffer稳定，但真实训练Actor为autocast=true/bfloat16、Accelerator=bf16；E07离线为no，见[入口证据](README_A3_ZHY_ENTRY_20261012.md)。

新机已安装TRL `experimental/ppo/ppo_config.py:129–135,301–304` 中bf16默认None，未指定时转为 `not fp16`，即通常true。Transformers `training_args.py:1872–1878`及2092–2099据此重写ACCELERATE_MIXED_PRECISION；vendor先解析PPOConfig再创建未显式指定精度的Accelerator。因此启动脚本 `--mixed_precision=no` 会被后续参数解析覆盖。此为本环境与入口的可复现原因，不将其推广为所有官方环境均有同一问题。

## 修正内容与固定合同

新增 `script/a3/consistent_precision.py`：复用既有VerifiedFineTuneTrainer、官方PPO/辅助loss、学习率与持久化保存。新入口ConsistentPrecisionTrainer要求algo.trl.bf16/fp16/tf32均false、Accelerator=no/native_amp=false，采样前禁用matmul/cudnn TF32；在真正Actor backbone入口检查autocast及观测dtype，异常立即失败。每次调用均检查，计数与每种模式首条精度记录固定大小，避免长训累计大量诊断数据。

E09通过 `script/a3/zhy_consistency.py` 与 `zhy_consistency_trainer.py` 验证：官方035、16环境/24步/4小批次、seed0，与E08仅改变实际AMP设置；明确设置ACCELERATE_MIXED_PRECISION=no及 `++algo.trl.bf16=false` / `++algo.trl.fp16=false` / `++algo.trl.tf32=false`。复用E08真实入口捕获；首实际4环境batch之外补算其余3组不重叠batch，覆盖所有16环境。补算不代表原训练循环的后续随机顺序。

仍禁止backward/optimizer；初始允许官方std钳位。验收要求输入与采样存储精确一致、Actor权重/buffer稳定、实际FP32、四batch mean RMSE≤1e-5且ratio最大偏离1≤0.001，不改旧logprobs来强行凑ratio。

新机ZHY串行执行、原机不计算，需同boot E08核验与GPU空闲/可靠遥测；CPU90/GPU85摄氏度连续2次5秒、4GiB内存保留、900秒上限、OOM/新boot/关键内核错误/遥测缺失停止。不重置E08或E09、不重复启动、无开机恢复，vendor和历史账本保留。

## 版本、命令与产物

工程基于 `e473c1679082c40b9cfaf8d1089bb308c40d8c35`，固定vendor `fe6868ba37034f89b912f0fb851bce19f120266d`；注册记录固定代码/模型SHA。模型/张量/原始日志只留新机LOCAL_ONLY。

```bash
cd /opt/data/private/hechengzhu/ZHY
PYTHONPATH=.:script/vendor/sonic_for_a3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m unittest script.a3.test_consistent_precision script.a3.test_zhy_precision script.a3.test_zhy_entry_probe -v
data/environments/a3-sonic/bin/python -m script.a3.zhy_consistency register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_consistency run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_consistency verify
```

输出 `data/experiments/a3_zhy_consistency_20261012_E09/`，日志 `logs/a3_zhy_consistency_20261012_E09/`，小型清单 `data/manifests/a3_zhy_consistency_20261012_E09.json`。全部概率和输入身份由CPU独立重载复算；Mac仅小型代码/README/清单。

## 实测、边界与下一步

8项CPU回归通过：实际配置复现默认BF16覆盖并验证显式false保留no、错误native AMP/实际autocast拒绝、输入与四batch覆盖。最初一次测试的PyTorch后端mock属性恢复失败，已改为显式try/finally恢复；失败发生在CPU测试，无GPU实验启动。

E09尚未注册或执行，不能写成修复通过。即使零更新概率门槛通过，也只说明优化前两条路径一致，不保证PPO每轮变好或所有历史退化均由此导致。

代码/合同与索引/current state在执行前scoped审计提交推送main并核验远端。下一步独立E09真实验收，通过后使用同入口做有界短训练并独立核验更新与效果，继续以官方策略作基线。
