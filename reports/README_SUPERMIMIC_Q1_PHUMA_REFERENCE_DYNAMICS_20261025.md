# PHUMA合格参考的接触与逆动力学诊断

记录日期：2026-10-25，北京时间。阶段PHUMA-D04，状态：8段真实诊断及736项独立统计/差分重算完成。

## 目标与决定

用户要求继续现有资源，BONES由师兄处理。原R02第5000轮开发对照仍未学会移动，因此除继续原剩余有限预算和PHUMA扩库，再核实已通过静态19门的训练参考是否存在明显滑脚、接触或所需力矩缺口。该诊断不证明唯一失败原因，不改变SuperMimic世界模型/BPTT/critic方法。

新增`script/supermimic_q1_sources/inspect_phuma_reference_dynamics.py`，只从R02C注册的340训练参考中选8段`locomotion`，固定seed2026102508的SHA顺序，与策略结果无关；不选择development/test。此标签只表示源位移阈值，不能当步行语义认证。转换manifest、独立reference_check与注册split完整连接，拒绝资格、19门、来源、字节或划分不一致。

复用已验证的`inspect_reference_dynamics.py`之模型配置、几何接触、三/五点qvel差分、inverse与RNE+armature分解；不修改原helper、参考或训练冻结文件。按相同Q1模型、控制cap、摩擦1.0和足接触优先级2，报告静态轨迹脚底间隙/切向速度、接触比例、根残余外力和关节力矩相对cap。0积分、策略推理和optimizer；逆动力学不是优化接触力分配，也不能据此证明近似跟随不可能。

## 版本与验证

工作副本`/tmp/yuanqi-supermimic-q1-20261023`、分支Q1、阶段前发布`8a55bad14d01207a79c203e67990a7313e5fdcbf`。新producer SHA`7afeb5e99cb5649e09eae1ecf303a3abe692d8180285e7dab3dc85aba0a32e88`，测试SHA`7d5cf73f25bc97533b0b0eb0a503d1349e2974c0350185a1378abdef6dc198ff`。复用helper SHA`ee36f6f9c89663b61c2655d06d58db7be44c356658cad440f1cbbe80f2c9e6ba`及独立统计auditor SHA`e07c0a597500cb64e8b78279e4d6e39bd21272bad34b4d641e53d803641bedf6`已核服务器一致。

Mac三项控制通过：固定选择不受输入顺序影响、只取train且不改输入；拒绝门/划分/原字节/来源/重复身份变化；拒绝交叉split或训练候选不足。语法检查通过。这些是选择控制，尚非真实MuJoCo结果。03:36核新文件均不在R02C的496冻结内，R02C900新轮/累计5900、WM309组/actor259次，内存19.00GB/OOM14/GPU40；B019转换352段后独立检查进行中。

## 命令与产物

发布并独立核远端Q1后，将新两文件按SHA部署，再运行同三项控制和真实只读选择预检。实际执行使用原Python，无新安装：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  script/supermimic_q1_sources/inspect_phuma_reference_dynamics.py \
  --root /opt/data/private/YuanQi_Q1 \
  --experiment /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_R02C_partial5000_extend7000_20261025 \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_D04_reference_dynamics_20261025 \
  --expected-oom-kill 14
```

guard沿用28.75GiB/OOM14提前保护、100GiB盘余、输出512MiB、nice10/单线程/无CUDA、900秒无进展终止并回收自身worker。状态为新目录guard.json，原始数组与receipt在diagnosis/，stdout为`logs/PHUMA_D04_reference_dynamics_20261025.log`。完成后使用原`script/supermimic_q1_analysis/audit_reference_dynamics.py --root ROOT --receipt OUTPUT/diagnosis/receipt.json --output OUTPUT/summary_audit.json`独立重算保存统计和差分；不把同helper自比当独立重算。

## 边界与下一步

原R02 FAILED及R02C固定340输入/7000预算保留。静态19门仍是现有准入，诊断不增加新通过门或自动淘汰/改写参考，不把描述统计当可控性证明。先记录真实诊断及独立审计，再依据具体结果选择后续改进；当前0新效果证据。B019完成后先保存该批记录，再完整组库并重查新增HumanML别名；不在当前训练中热换参考。

本阶段代码/准备记录待审核提交推送并独立核远端；完整数组和大型工件仍服务器LOCAL_ONLY。

## 真实执行与独立审计

准备发布`d09a4cacbcff28adcbd34fff751e998fbccf17f0`已推送并独立核远端Q1。两份新代码SHA一致，服务器三项控制及真实train选择预检通过；03:40:24至28原生退出0，guard836214/worker836215已不存在，worker_reaped=true/return_code0。八片共1441分析帧/2882次inverse，40份输入前后SHA不变、无Torch导入、0积分/推理/optimizer。进出共享内存17435496448→18099179520字节/OOM14，峰值RSS728988KiB，输出分配1232896字节。

03:41原独立auditor退出0，736项统计及保存qvel差分重算全过，最大差`1.1102230246251565e-16`。它重算保存统计，没有再次独立调用inverse；producer同时逐帧检查inverse与RNE+armature分解，最大差`8.53e-14`。原样小证据已取回，两端SHA一致：[诊断](../data/manifests/supermimic_q1_phuma_D04_reference_dynamics_20261025.json)66935字节/SHA`f9b04eaaea4c872e808f73c2849eb933ecc95d4c9624bf054d498a37941a2562`，[guard](../data/manifests/supermimic_q1_phuma_D04_guard_20261025.json)1953字节/SHA`6e5089caaa7f99c48dba1d345dd939a7832fa443e6dd2af3ede739827afd2bc3`，[独立重算](../data/manifests/supermimic_q1_phuma_D04_summary_audit_20261025.json)34601字节/SHA`596a1f804b5490dfcdb9363589c9ed93404c62bde040e95f4938d158180165b5`。

下表间隙为每帧较低脚底的中位数；近地滑动指脚底10mm内样本中切向速度超过0.05m/s的比例，是描述指标而非新增质量门。力矩超cap来自中央三点差分；五点结果及全部原始数值在诊断收据中。

| 固定训练来源 | 脚底中位间隙 | 实际脚地接触帧 | 近地滑动 | 关节超cap帧 |
| --- | --- | --- | --- | --- |
| humanml/000962 | 6.98mm | 0.51% | 37.95% | 0.51% |
| idea400/Running_in_place_and_walking_at_the_same_time | 10.81mm | 0 | 86.67% | 0 |
| idea400/Simultaneously_walking_and_Shake_hands_and_say_goodbye | 6.92mm | 0 | 57.05% | 0 |
| kungfu/Tai_Chi...wild_horse_parts_mane | 7.68mm | 0.51% | 53.08% | 0 |
| humanml/011361 | 7.15mm | 0 | 35.67% | 0 |
| humanml/006220 | 7.10mm | 0 | 68.98% | 1.54% |
| idea400/Simultaneously_Gestures_in_the_Air_and_walking | 11.66mm | 0 | 70.18% | 0 |
| haa500/dj | 11.96mm | 0 | 42.86% | 0 |

八片精确轨迹所需根残余外力中位数约0.999至1.007倍体重，与缺乏真实脚地接触相符。六片关节从未超当前cap，不能泛称“力矩不足”，另外两片少数超cap峰值仍需局部接触/加速度归因。八片也不能代表全部62训练位移候选或证明唯一策略失败原因。

## 结论与后续决定

现有19门能够筛出近地、限位及导出合规参考，但15mm支撑容差不等于真实接触，仍会接纳浮脚/近地滑动。此为PHUMA当前合格输入的实际物理一致性缺口；它支持后续独立小样本接触/轨迹一致性改进，并不证明单独降低根高就能解决。旧A3来源D05B逐帧根高投影曾恶化动力学，保留该失败依据，不直接把同类修补混入新库。

当前R02C继续原剩余有限预算及配对评测。下一先完成P08扩库和来源审计，再以训练样本独立核对平滑接触约束/参考速度的一致性；开发或测试成绩不用于选修补参数，不直接增加力矩上限。原参考、496冻结及旧终态保持，真实移动策略目标仍未达。
