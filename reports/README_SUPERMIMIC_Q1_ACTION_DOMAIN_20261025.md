# Q1真实与想象动作域一致性

记录日期：2026-10-25，北京时间。阶段S02。状态：接口缺口确认，服务器五项控制、保存1500审计15门及真实replay H32/BPTT探针7门通过；尚无修正后训练或跟随改善结果。

## 问题与目标

用户要求BONES交师兄，继续现有资源。R02C于04:07:44新增1810轮非有限数值失败，详见[原终态](README_SUPERMIMIC_Q1_R02_PARTIAL_CONTINUATION_20261025.md)。原FAILED与496冻结文件保持，不在失败目录续跑、不放宽非有限检查。本阶段先纠正实际仿真与想象训练的动作域差异，保持SuperMimic世界模型、H32、BPTT与critic路线；不依赖新增数据或更换算法。

`supermimic/q1_sim.py:step`将clean和带噪执行动作裁至[-20,20]，真实奖励、历史与replay使用此合同；原`imagination/rollout.py`直接使用无界MLP输出，作为世界模型输入、奖励和下一历史动作。因此不只是极端动作数值很大，两个分支实际动作域不同。原22维、dt0.02、action-rate权重-0.1下，若连续动作均在[-20,20]，每步该项下界是`-0.1 * 0.02 * 22 * 40^2 = -70.4`；R02C第1800轮想象均值为-1693.5538，不能来自这一有界合同。

此证据确认接口不一致并支持其作为崩溃候选机制；失败现场replay和1810权重未保存，不能宣称已精确重放崩溃、证明它是唯一根因或保证修正后学会移动。浮脚/滑动及训练规模差距仍按原记录处理，不把动作修正与改参考、增力矩、改H同时混在一轮。

## 实现与检查

新增独立`script/supermimic_q1_stability/bounded_policy.py`，继承原Policy且保持网络参数名、形状与state_dict不变，只对forward结果应用与真实仿真相同的clip20。真实分支此前也会clip，闭环clean动作合同一致；想象中世界模型、奖励、历史和previous action共同接收有界动作。域内梯度不变、饱和区梯度为0，原始NaN/Inf在裁剪前明确拒绝，不将Inf伪装成合法20。

同文件提供三个Adam的step前非有限梯度拒绝hook，由独立trainer子类绑定；正常有限Adam更新和状态不改。原trainer已有grad-norm/value剪裁和非有限日志保护继续保留。该hook在坏梯度写入Adam前停止，不是给坏样本填零或跳过更新后继续训练，也不声称穷尽所有浮点异常。

`audit_failed_checkpoint.py`独立CPU/mmap检查R02C最后step1500保存：原FAILED、496冻结、保存字节、网络/Adam/bin有限性、父5000至累计6500和各Adam真实增量、同admission、完整有限日志前缀。它不将1810失败进度当权重，也不改写原终态或执行推理/优化器。

`probe_bounded_imagination.py`只在该保存审计全部通过后，从原340训练片中按固定SHA/seed选2个位移候选+2个其他参考，16环境/4槽实际采128轮/2048控制转移。用同一真实replay起点和六个固定噪声种子比较原/有界H32前向，再对有界策略执行一次含原critic的BPTT，检查参数梯度有限且非零、世界模型和critic参数无梯度、全部输入SHA不变。0optimizer，非精确崩溃重放/策略效果评测，不使用dev/test选参数。

本地三个实现及测试语法通过。五项PyTorch控制为域内前向/保存布局不变；极值裁剪和导数；NaN/Inf不被掩盖；action-rate有限下界；有限Adam相同而坏梯度写入前拒绝。Mac缺torch，五项均明确skip，不能计作通过；须使用原服务器环境实际运行。

## 版本与复现

权威工作副本`/tmp/yuanqi-supermimic-q1-20261023`、分支Q1；服务器`huanyu:/opt/data/private/YuanQi_Q1`。所有新文件位于原496冻结集合外，原CORE/expanded/PHUMA/partial/extension源码不改。先审核发布核远端，再逐SHA部署并执行：

```bash
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" \
  -m unittest discover -s script/supermimic_q1_stability -p test_bounded_policy.py -v
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" \
  script/supermimic_q1_stability/audit_failed_checkpoint.py --root "$PWD" \
  --expected-oom-kill 14 \
  --output data/supermimic_for_q1/PHUMA_S02_R02C_saved1500_audit_20261025.json
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" \
  script/supermimic_q1_stability/probe_bounded_imagination.py --root "$PWD" \
  --checkpoint-audit data/supermimic_for_q1/PHUMA_S02_R02C_saved1500_audit_20261025.json \
  --expected-oom-kill 14 \
  --output data/supermimic_for_q1/PHUMA_S02_bounded_H32_probe_20261025.json
```

这些是真实运行前命令，不重复覆盖已有输出。OOM14须现场重核；保留28.75GiB探针内存线、至少100GiB盘余、原12GiB实验预算。探针/审计stdout保存`logs/PHUMA_S02_{controls,saved1500_audit,bounded_H32_probe}_20261025.log`，实际执行后补充原件与结果。无需安装依赖、下载数据或新增大型Mac资产。

## 边界、保存与下一步

先完成服务器五项控制、保存审计与真实replay/BPTT检查，再决定独立有界动作训练。原R02C已失败，其7阶段后续不自动补为完成；若安排新匹配实验，仍从已审计原5000父保存、原340片、相同seed和7000固定预算开始，仅修正动作域并增加拒绝坏梯度保护。新612片P08B入口[P09](README_SUPERMIMIC_Q1_SOURCE_VIEW_REGISTRATION_20261025.md)已真实通过，可供后续独立数据实验，本阶段不混入以免归因混杂。

新代码、阶段报告、索引/当前状态和原样小收据按Q1提交推送并独立核远端；大权重只留服务器LOCAL_ONLY，未宣称新增网盘备份。当前无有效移动策略改善证明，主目标仍未达。

## 04:36至04:37服务器实际验证

修正代码`aa3696de02b7bf487fac05c0d0aa76b900c094cc`已独立核远端Q1。首次SSH代理推送连接被关闭，改用直连SSH成功推送并独立比对；未将失败推送记成同步。四份新文件逐SHA部署，均不在原496冻结范围。

服务器五项PyTorch控制全部通过、0跳过，包含有限Adam与原更新一致、NaN梯度拒绝时参数和状态不变。控制中的两个合成小Adam成功步不是项目训练；小收据`new_optimizer_updates=0`仅指无项目训练更新，其note保留此边界。[原样控制收据](../data/manifests/supermimic_q1_S02_controls_20261025.json)SHA`6b83855518a6269b773050185b55e0387c091bb098bae755926298af60067690`，原stdout在服务器`logs/PHUMA_S02_controls_20261025.log`。

04:36:54保存1500独立CPU审计原生退出0、15门全通过：网络/Adam/bin有限，WM549组/actor499次，三Adam19490→24980、1899→2398、37980→47960；3网络真实变化且完整前缀、原admission、全部496冻结及文件前后身份保持。原FAILED不改、无final，确认仅累计6500保存可读取，不把6810当已保存。[审计原件](../data/manifests/supermimic_q1_S02_saved1500_audit_20261025.json)6035字节/SHA`b69c57f29e4ef3723eae864f38984c9dfaa7ab5a1548b21bca2f8a7dcbfd9e7c`。

04:37:24真实探针原生退出0，2048控制转移、6对同起点H32前向及1次有界actor backward完成，7门全过；actor梯度范数0.25272274、世界模型/critic参数无梯度，0项目optimizer。原/有界动作峰值均6.538至6.634、0越界，5对状态精确相同，另一对最大差7.63e-6；因本批原动作没有超过20，不把这些正常样本当作已重现失败机制。有界reward/action均有限，输入前后身份无变。[探针原件](../data/manifests/supermimic_q1_S02_bounded_H32_probe_20261025.json)SHA`e79a98dc40b0aa0c51df0cb4943a1298f7dc7c2ddd3d991d717b1dc69a8eef0c`。实际OOM14、共享内存12.94→15.59GB、盘余1.196TB；共享挂载其他写入下降不能归于本探针。

本阶段验证完成，下一独立R02D使用原5000父、相同340训练/seed20261025/7000新轮，仅显式修正动作域并增加写入前拒绝坏梯度；不同时接入612片新库。先登记、保存新协议、发布核验，再启动并立刻独立检查step0继承，不能复用旧FAILED目录或将旧通过结论当新机运行证据。
