# PHUMA合格参考的接触与逆动力学诊断

记录日期：2026-10-25，北京时间。阶段PHUMA-D04，状态：新入口及三项本地选择控制通过，真实服务器诊断待执行。

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
