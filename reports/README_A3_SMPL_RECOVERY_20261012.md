# A3 脚地板不足间距恢复 E20

日期：2026-10-12。实验：A3_ZHY_SMPL_RECOVERY_20261012_E20。状态：VERIFIED_QUALITY_REJECTION，合格pair0/7，未训练。

## 目标与单变量

E19已确认碰撞正间距h单位错误，修正后最差真实帧仍有负碰撞距离和h=0。新的假设是：非线性积分/最低顶点切换产生微小穿透后，零截断没有恢复要求，误差可能累积。E20仅将不足间距的碰撞h改为`.85*(distance-.001)`有符号恢复，正间距沿用E19位移单位；G、目标、限位、速度、根轨迹、采样、全部质量门保留。安装库/vendor和E14–E19账本不改。

合成检查先把脚实际置于地板下1cm：旧h0允许保持穿透，新h必须产生向外位移；8个真实受限QP步骤应逐渐恢复至1mm间距。真实实验仍固定7段各300帧。失败结果同样保存，禁止重置/重复与自动恢复。

## 版本与命令

仅ZHY `/opt/data/private/hechengzhu/ZHY`既有Python，单线程CPU/无CUDA/无模型前向/0optimizer/0策略回放。原hp3090仅Git和文件传输。新代码`script/a3/smpl_collision_recovery.py`、`script/a3/smpl_recovery_probe.py`、`script/a3/test_smpl_collision_recovery.py`；代码、安装库、源文件与父工件SHA由注册固定。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.test_smpl_collision_recovery
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_recovery_probe register
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_recovery_probe run
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_recovery_probe verify
```

90/85°C连续两次温度保护、保留4GiB内存、5s采样、900s上限、首故障停止。工件`data/experiments/a3_zhy_smpl_recovery_20261012_E20/`，日志`logs/a3_zhy_smpl_recovery_20261012_E20/`，小清单`data/manifests/a3_zhy_smpl_recovery_20261012_E20.json`。数值/原始日志服务器LOCAL_ONLY，Mac仅小型文件。

完成后独立从已保存qpos重算FK、CSV、脚几何与全部质量门，保存README/索引/current state、审计scoped提交推送核验。合格对仅可支持下一有上限离线学生蒸馏，不代表真实SMPL准确度、动力学平衡或超过官方。恢复先读[E19结果](README_A3_SMPL_UNITS_20261012.md)，不凭旧进程记录重新启动。

## 实际结果与下一变量

4项CPU契约0.123秒通过，包括已穿透1cm的脚在8步真实QP中单调恢复至1mm间距。真实2100帧于北京时间23:03:41–23:04:07完成、23:04:18独立验收拒绝。7/7root位置/滑动/贴地/速度/限位均通过，6/7位置P95通过；但最低脚底仍依次-1.66/-1.88/-1.72/-1.95/-0.96/-1.73/-1.76cm，7/7超过5mm预算，squat位置P95 6.13cm另失败。最大穿地由E19 6.32cm降至1.95cm，恢复约束有作用但尚不充分。合格pair0、optimizer0、策略回放0，CUDA未初始化，任务已退出。

8健康样本覆盖UTC15:03:38.137–15:04:18.614，最大间隔10.88s；同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，CPU42/GPU26°C，可用最低9,307,340,800字节、OOM0，无关键内核/遥测错误。版本基线`a850749c1a97e62ae869a4c1708508fa16605979`；代码/输入/安装库/工件SHA见[清单](../data/manifests/a3_zhy_smpl_recovery_20261012_E20.json)。

下一单变量是在每次候选积分后核验全部实际脚几何，越过地板则回退并减小该步，记录原候选与接受结果；维持固定IK步数和时间预算，不额外加入求解时间或修改目标。该检查可直接检验线性化约束和非线性积分之间的差异，不能只通过最终脚底抬高投影掩盖速度或身体误差。
