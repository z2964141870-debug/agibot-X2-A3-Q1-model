# A3 非线性候选积分几何验收 E21

日期：2026-10-12。实验：A3_ZHY_SMPL_CHECKED_20261012_E21。状态：VERIFIED_QUALITY_REJECTION，脚几何7/7通过但合格pair0，未训练。

## 目标、固定变量与代价

E20有符号恢复使最大穿地6.32→1.95cm，仍超5mm预算。E21保持相同QP/目标/关节限位速度/每帧步数和dt，在每次完整候选积分后核验全部脚mesh/box顶点，最低Z若小于地板0则还原原qpos并按1/2缩步，最多12次；所有候选均不安全则保持该步原qpos。初始实际脚几何必须已在地板上方，禁止事后直接抬高root掩盖穿地。记录完整候选、接受后高度、接受比例、尝试数和原最大位移。

缩步会损失该帧跟随进度，不能当作免费增加时间；仍120初始化步骤、后续8×1/240s，全部旧姿态/root/接触/速度门槛保留。合成旋转盒角测试要求完整步穿地、缩步后实际最低角安全，另测安全完整步保留及拒绝初始穿透。真实仍固定7段各300帧，禁止重置/重复/自动恢复。

## 版本与执行

仅ZHY `/opt/data/private/hechengzhu/ZHY`既有Python、单线程CPU，无CUDA/模型前向/optimizer/策略回放。原hp3090仅Git与传输。新增`script/a3/smpl_checked_integration.py`、`script/a3/smpl_checked_probe.py`、`script/a3/test_smpl_checked_integration.py`，注册固定代码/安装库/父工件SHA，原E14–E20和vendor不改。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.test_smpl_checked_integration
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_checked_probe register
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_checked_probe run
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_checked_probe verify
```

90/85°C连续两次温度保护、4GiB内存保留、5秒采样、900秒上限、首故障停止。工件`data/experiments/a3_zhy_smpl_checked_20261012_E21/`，日志`logs/a3_zhy_smpl_checked_20261012_E21/`，小清单`data/manifests/a3_zhy_smpl_checked_20261012_E21.json`。每段新增`integration_checks.npz`，数值数据/原日志服务器LOCAL_ONLY，Mac仅小型代码/报告/清单。

独立验收从保存qpos重算FK、CSV、原目标、实际脚底与全部质量门，并检查逐步积分记录覆盖/有限性/接受比例/非负脚底。合格对仅支持下一有上限离线学生蒸馏，不能证明动力学平衡、真实SMPL精度或超过官方。完成即保存README、索引/current state，审计scoped提交推送核验，恢复先读[E20](README_A3_SMPL_RECOVERY_20261012.md)。

## 实际结果与拒绝理由

3项CPU合成检查0.103秒通过。2100真实帧于北京时间23:12:20–23:13:23完成、23:13:33独立验收。全部7段真实脚底非负、0穿地，限位/关节速度/接触贴地/滑动均通过；但全部7段位置P95/最大失败，P95 5.44–24.90cm，最大31.12–67.53cm，6段root跟踪失败，5段root速度超4m/s。因此合格pair0、optimizer0、策略回放0，不能把不穿地单项通过称为可靠标签。

每段2512次积分均保存。各段缩步/保持次数分别922/656、450/348、810/482、1338/923、943/779、1243/763、1046/615；原完整候选最低脚底-0.89至-2.62cm，接受后非负。证据显示只在求解后缩小整步会频繁停止进度，不能采用本版本训练。下一项把脚平面包络多个顶点约束直接放入QP，使求解器可提前改变移动方向，而保留非线性核验；包络的保守性和贴地误差必须另外计入旧门槛。

15健康样本UTC15:12:17.274–15:13:33.105，最大间隔10.18s；同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，CPU44/GPU26°C，可用最低9,299,935,232字节、OOM0，无关键内核/遥测错误，任务退出。基线HEAD `de58fb4a468d57b0b8675d7ff5d198335a7a9f66`；代码/输入/安装库/工件SHA见[清单](../data/manifests/a3_zhy_smpl_checked_20261012_E21.json)。原始工件LOCAL_ONLY，小记录按scoped提交推送核验。
