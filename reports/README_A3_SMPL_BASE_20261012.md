# A3 连续求解floating base与关节 E18

日期：2026-10-12。路径实验：A3_ZHY_SMPL_BASE_20261012_E18。状态：VERIFIED_QUALITY_REJECTION，合格pair 0/7，未训练。

## 定位与更改

E17证明实际脚/floor geom pairs存在、地板Z0及负距离与负脚Z一致，但每帧在IK前直接改root qpos绕过了地板约束。E18不再每帧直接设置floating base，root和所有关节共同由Mink求解；root任务position/orientation cost固定100/20，替代每帧root直接设置及硬等式。初始化仍从固定机器人rest开始，首帧120个受限IK步骤，源帧8×1/240s。foot/floor约束、joint限位速度、contact目标和所有旧质量预算保留。

额外记录实际root对目标的位置P95/最大，预算0.05/0.15m；root任务变为软目标可能产生偏差，不能只看脚不穿地而忽略身体目标。所有body误差仍计入原全局P95/最大。原始/父失败账本不改。

## 执行、记录与边界

代码 `script/a3/smpl_base_retarget.py`、`script/a3/smpl_base_probe.py`；基于已保存E17脚地板/root/contact方案，固定vendor不修改。ZHY CPU单线程，0模型前向/训练/策略回放/无CUDA，原机仅代码/记录同步。沿用90/85°C、4GiB保留、5秒采样、900秒上限、首故障停止、无自动恢复；新的实验不重复或重置。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.test_smpl_base_retarget
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_base_probe register
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_base_probe run
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_base_probe verify
```

工件 `data/experiments/a3_zhy_smpl_base_20261012_E18/`，日志 `logs/a3_zhy_smpl_base_20261012_E18/`，小清单 `data/manifests/a3_zhy_smpl_base_20261012_E18.json`。原始/数值数据服务器LOCAL_ONLY，Mac仅小型代码/记录。独立从保存qpos复核全部body/脚几何、目标/root/contact及CSV；合格pair仅支持有上限离线监督，不等于动力学平衡、在线能力或超过官方。

## 实际结果与连接恢复

2项CPU合成测试通过（2.069秒），包括根目标下降10cm时支撑脚不穿地。真实7段共2100帧于北京时间21:43:35–21:43:56求解完成、21:44:05独立验收完成；7/7限位、速度、接触滑动、贴地、root速度及root位置跟踪通过，但7/7仍穿地1.56–6.11cm，squat位置P95 6.13cm亦超过5cm门槛。因此训练门关闭，0 optimizer、0策略回放、CUDA未初始化。

| 序列 | 位置P95 / 最大 cm | 最低脚底 cm | root位置P95 cm |
| --- | --- | --- | --- |
| zj walk | 3.04 / 10.44 | -6.09 | 0.31 |
| zj warmUp | 4.60 / 7.80 | -2.93 | 0.79 |
| fjw walk | 2.76 / 6.62 | -2.65 | 0.21 |
| zls walk | 3.12 / 9.89 | -5.44 | 0.27 |
| yy squat | 6.13 / 14.11 | -1.56 | 4.78 |
| yy walk | 2.90 / 9.63 | -6.11 | 0.29 |
| zzx walk | 3.02 / 10.66 | -6.07 | 0.31 |

7个健康样本覆盖UTC13:43:32.264–13:44:05.615，最大间隔9.77秒；同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`，CPU最高41°C/GPU27°C，容器可用最低8,858,140,672字节，OOM0，无关键内核/遥测错误。22:41恢复SSH后确认同boot、state为已独立验收拒绝、无活动E18进程；连接中断期间未观测到的状态不自行补写，不认定发生重启。

连续base求解没有消除穿地，因此E17的root赋值机制不是充分解释，不能标为已解决。当前安装Mink碰撞限制的h使用`gain*(dist-min)/dt`，而build_ik求解变量为位移Delta q；VelocityLimit明确使用`dt*vmax`。下一实验用独立最小反例验证单位合同，再仅修正碰撞限制到位移单位，不修改vendor/历史代码或放宽验收。

注册文件中的experiment字符串因模板替换遗漏保留为`A3_ZHY_SMPL_FLOOR_20261012_E18`；实际BASE路径、脚本SHA、主要变量及2100帧均正确。保留原注册和哈希，不修改已执行代码以掩盖该命名错误。

版本基线 `0ff135e7b6c1c0f11d26cc289dbd402615ab2a84`，本阶段三份新脚本SHA及工件SHA见[小清单](../data/manifests/a3_zhy_smpl_base_20261012_E18.json)。原始数据/日志仍服务器LOCAL_ONLY；小代码/记录按本阶段scoped提交同步，提交身份由Git历史提供。未经独立数据质量门不更新学生/官方模型。恢复先读本文件、清单及[E17机制](README_A3_SMPL_FLOOR_20261012.md)。
