# A3 碰撞约束位移单位 E19

日期：2026-10-12。实验：A3_ZHY_SMPL_UNITS_20261012_E19。状态：VERIFIED_QUALITY_REJECTION，合格pair0，未训练。

## 目标、依据和固定变量

E18不再直接赋值root但2100帧仍穿地。安装的Mink 1.2.0中，`solve_ik.py`明确用Delta q构建QP再除dt返回速度；`VelocityLimit`的h是`dt*vmax`，但`CollisionAvoidanceLimit`的h是`gain*(dist-min)/dt`。这项碰撞上限与求解变量的单位不一致。在1/240秒步长下，正距离接近预算放大240倍，负距离时h归零只能阻止继续接近而不主动恢复。

本阶段先以单滑动关节/球/平面检查不一致是否能允许实际穿地，再用项目内Limit适配器仅将碰撞h乘dt，保留G、gain .85、1mm最小距离、15cm检测距离、零relaxation与全部E18目标/配置/关节限速。vendor、安装库和E14–E18账本不修改。以新编号登记后运行，首故障停、无重复/自动恢复。

## 版本、命令与保存

ZHY `/opt/data/private/hechengzhu/ZHY`，既有`data/environments/a3-sonic/bin/python`，单线程CPU，无CUDA/模型前向/策略回放或optimizer。原hp3090仅Git与文件传输。代码`script/a3/smpl_collision_units.py`、`script/a3/smpl_units_probe.py`、`script/a3/test_smpl_collision_units.py`；注册时固定安装库三份源码SHA，不能套用到未经验证的其他版本。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.test_smpl_collision_units
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_units_probe register
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_units_probe run
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_units_probe verify
```

固定7段各300帧，保持全部位置/脚几何/接触/速度/root门槛。CPU90/GPU85°C连续两次、4GiB内存保留、5秒采样、900秒上限、首故障停止。路径`data/experiments/a3_zhy_smpl_units_20261012_E19/`，日志`logs/a3_zhy_smpl_units_20261012_E19/`，小清单`data/manifests/a3_zhy_smpl_units_20261012_E19.json`。

原始数值工件服务器LOCAL_ONLY，Mac仅小代码/清单/报告。完成后独立重算FK/CSV/目标/脚底与全部质量门，立即保存README、索引/current state，审计scoped提交推送核验。即使通过也只支持下一有上限离线蒸馏，不能证明动力学平衡、在线跟随或超过官方。

## 实际验收与剩余机制

最小反例3项CPU测试0.106秒通过：5cm正间距时原约束允许6cm下降并实际穿地；位移适配器拒绝该步骤，在1/30、1/120、1/240s下给出相同位移预算；真实QP求出的单步保留至少1mm间距。初版测试曾缺Q1 helper导入，随后简化模型的脚为world直接子体而被库parent-child规则过滤；在真实实验登记前修正测试模型，并加入恰好1个约束行断言。未重置已登记实验。

真实7段2100帧于北京时间22:54:29–22:54:50求解、22:55:00独立验收。限位/速度/root/接触滑动/贴地均通过，但最低脚底依次为-6.32/-2.94/-2.66/-5.23/-1.42/-5.68/-5.98cm，7/7几何仍失败；squat P95 6.19cm也失败。0合格pair/0optimizer/0策略回放，不能据最小反例通过宣布真实问题解决。

7健康样本UTC14:54:27.031–14:55:00.525，最大间隔9.98s；同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，CPU52/GPU26°C，可用最低9,294,163,968字节，OOM0，无新关键内核/遥测错误，任务已退出。

事后CPU只读重算每段最差已保存qpos：六个碰撞pair存在，碰撞距离与几何minZ一致，负距离行G_root_z=-1且h=0。例如zj walk第149帧左实际碰撞geom55距地板-0.0609373m、视觉最低-0.0632114m；fjw第89帧geom72为-0.0265301m。这排除仅视觉包络或漏掉pair的解释。库在间距不足时把h截为0，能阻止线性模型继续接近，却不要求恢复已有穿透；下一单变量为有符号间距恢复，继续检查非线性步进/最低顶点切换，不将机制假设写成已通过。

基线HEAD `6c85779f510df5b8dcb79ce7626306d412a01d91`，新代码/输入/安装库SHA及结果见[清单](../data/manifests/a3_zhy_smpl_units_20261012_E19.json)。小记录按本阶段scoped提交同步，提交身份由Git历史提供；原始工件LOCAL_ONLY。
