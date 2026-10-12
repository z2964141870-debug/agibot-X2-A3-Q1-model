# A3 真实首训练入口零更新诊断

记录日期：2026-10-12。独立实验 A3_ZHY_ENTRY_20261012_E08。状态：真实入口捕获及CPU独立核验完成、0更新；确认真实入口启用BF16，与离线float32回放不同，修正待验证。

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

Mac静态编译通过，新机3项CPU证据测试通过（4.181秒）。实际北京时间16:06:16–16:07:07完成，0optimizer/0backward，独立CPU重载复算通过，state=VERIFIED；概率门槛仍失败。

真实Actor输入逐键与采样存储完全一致；Actor权重/buffer摘要从采样后至三次即时回放一致。原始Actor、包装器与loss入口mean完全相同，同Actor即时回放及去episode mask都精确复现真实训练输出，排除本次输入错位和包装后改写。实际首batch mean RMSE=0.010158833，ratio=0.534225–1.300595，最大偏离1=0.465775。

**关键新证据：真实Actor入口autocast=true、dtype=bfloat16，Accelerator mixed_precision=bf16；输入仍是float32，两个TF32开关false。** 因此E05“两个TF32均false”的事实成立，但不能推导为完整计算精度一致；E07显式构造mixed_precision=no的离线包装器也没有重建真实精度。这是对先前推断的更正，不改写E05/E07原始账本或数值。

固定vendor `gear_sonic/train_agent_trl.py` 先用HfArgumentParser生成PPOConfig，再创建未显式指定mixed_precision的Accelerator。新机已安装transformers `training_args.py:1872–1878`会根据training_args.bf16/fp16重写ACCELERATE_MIXED_PRECISION，因此启动脚本的 `--mixed_precision=no` 不是最终生效状态。下一轮显式关闭algo.trl.bf16/fp16、检查实际Actor精度后对照，尚未宣称这就是全部历史退化的唯一原因。

12健康样本同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`、CPU最高47/GPU35摄氏度、可用估计最低6.06GiB、OOM0，无新关键内核错误/遥测缺失。最大60.095秒间隔位于注册至启动之间，运行期仍按5秒采样。运行版本 `5f4bd2e0172e9d3093c52c8ade3c90976ce41601`，捕获SHA `b837e789303d5000201a2e9ff17f50a21828d1b705c1e3713e3803a8665111a5`，15522553字节；完整小型证据见[清单](../data/manifests/a3_zhy_entry_20261012_E08.json)。任务结束后GPU无计算进程。

让同输入同权重在两条路径上的概率一致，是优化前的计算验收；这不保证PPO每轮效果上升，也未证明历史退化的唯一原因，更不解释旧主机重启。

## 保存与下一步

本README与索引/current state及清单保存真实入口证据，scoped审计提交并推送main、核验远端。下一步独立E09显式关闭训练参数BF16/FP16，增加真实入口精度保护，沿用原门槛验收；通过后再做有界训练，不重置E08或只复跑离线变体。
