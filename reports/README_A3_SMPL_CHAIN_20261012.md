# A3 逐段骨长与姿态兼容重定向 E15

日期：2026-10-12。实验：A3_ZHY_SMPL_CHAIN_20261012_E15。状态：VERIFIED_QUALITY_REJECTION；形态映射残差明显下降，脚接触/根轨迹仍未通过，0训练。

## 问题与假设

E14原始标签7/7内部一致，但2100个新A3参考帧全部质量失败。只读取已保存目标/实际位置拆解后，行走腕部误差P95约0.64–0.70m，踝部多数约0.02–0.04m；首帧腕目标最低约0.22–0.42m，而机器人肩约1.2m，机器人手臂总长约0.35m。这些目标不满足臂长可达性，不是增加学生训练更新能修正的问题。

E14沿用了旧 `robot_rest + scale*(human_local-human_neutral)` 位置增量公式。人类canonical neutral是T姿态，机器人rest是垂臂/屈肘姿态；人体从T姿态放下手臂的位移又加到已经垂臂的机器人rest上，导致额外侧移/下降。旧首帧相对校准掩盖了此不兼容，不能直接拿它作为有正确绝对姿态的训练标签。

本阶段把主要变量限定为形态映射：骨段位置用人体对应段的方向和机器人固定段长构建；机器人髋/肩安装点使用已核验资产。给方向目标配套兼容link朝向，保留原参考相对旋转的扭转信息。原始片段、根运动/缩放、Mink/quadprog、4IK迭代、权重、限位/速度与质量门槛全部沿用E14。不增加地板补偿，不通过平移机器人掩盖穿地，未调宽质量门槛。

## 版本、运行与记录

父阶段已审计/提交推送并核验GitHub：`4b8d6130f3758307f6ae7b32a6d1bec908bce976`。vendor固定 `fe6868ba37034f89b912f0fb851bce19f120266d`。新代码 `script/a3/smpl_chain_retarget.py`、`script/a3/smpl_chain_probe.py`，登记代码与父工件SHA。合成契约检查单段方向/反向180度、零向量拒绝、机器人rest位置/旋转恒等与全部肢段长度。

所有计算只在ZHY CPU单线程，0模型前向/optimizer/策略回放/无CUDA，原机仅代码/记录同步。沿用90/85°C、4GiB保留、5秒采样和900秒上限；无自动恢复、不得重置E14/E15，首故障停止。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.test_smpl_chain_retarget
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_chain_probe register
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_chain_probe run
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_chain_probe verify
```

输出 `data/experiments/a3_zhy_smpl_chain_20261012_E15/`，日志 `logs/a3_zhy_smpl_chain_20261012_E15/`，小清单 `data/manifests/a3_zhy_smpl_chain_20261012_E15.json`。大型数值数据留服务器LOCAL_ONLY，Mac只保存小代码/README/清单；没有网盘验证。

## 验收与下一步

4项CPU契约通过（1.811秒）。新机20:46:34–20:46:50完成2100帧对照，20:46:59从全部保存qpos独立FK/足几何、重构人体目标、比对CSV、复算质量通过；不是重新跑IK。主要残差降低支持原形态映射不兼容的定位，但不支持数据已经可训练。

| 片段 | E14位置P95 | E15位置P95 | E15位置最大 | E15最低脚几何Z |
| --- | ---: | ---: | ---: | ---: |
| zj walk | 0.63164m | 0.02527m | 0.03259m | -0.03787m |
| zj warmUp | 0.40290m | 0.04364m | 0.11363m | -0.03889m |
| fjw walk | 0.62530m | 0.02302m | 0.02947m | -0.05032m |
| zls walk | 0.64187m | 0.02205m | 0.02805m | -0.03034m |
| yy squat | 0.66058m | 0.04530m | 0.21882m | 0.04040m |
| yy walk | 0.66006m | 0.02390m | 0.03209m | -0.01649m |
| zzx walk | 0.64690m | 0.02418m | 0.03088m | -0.02943m |

7/7位置P95通过、6/7位置最大值通过（squat首帧进入姿态受速度限位，后续须单独分析），全部限位/速度通过。6/7仍穿地，5/7接触滑动失败；合格pair0、训练门不放行。squat虽不穿地，最低脚点仍在地上4cm；仅“没有负Z”不够验证支撑，因此后续增加接触期贴地检查，不放宽现有门槛。

6健康样本20:46:31–20:46:59，同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`，CPU最高40°C/GPU27°C、可用cgroup最低8,875,827,200字节，OOM0，无新关键内核/遥测错误；最大间隔9.38秒含worker结束到独立核验。无CUDA初始化、0模型前向/optimizer/策略回放。计算已退出。

证据：[E15独立回执](../data/manifests/a3_zhy_smpl_chain_20261012_E15.json)。E14输入与失败记录SHA不变；没有重新训练、没有抬高地板或改宽门槛。剩余误差集中为根轨迹/肢体比例与支撑脚的世界位置不一致；下一独立阶段核对接触期root/足目标一致性，明确量化root修正与接触贴地/滑动，不用仅平移机器人掩盖失败。闭环改善仍未验收。

七段已参与转换质量诊断，不是未见最终测试。当前来源已由用户确认，优先修转换而非换掉内部一致的原始人体标签；如果数据缺少必要信息，再核验外部来源。恢复先读本文件及[E14结果](README_A3_SMPL_CORPUS_20261012.md)；重要阶段结束后立即更新README/索引/current state并审计推送。
