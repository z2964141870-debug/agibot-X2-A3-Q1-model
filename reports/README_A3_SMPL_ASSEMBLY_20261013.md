# A3被动参考装配与冻结闭环E33

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_ASSEMBLY_20261013_E33；14:10:19已停止，装配验收通过，闭环入口回执目录错误、0策略步；旧E33不复跑。

## 目标与唯一变量

继续衣服完整跟随和同条件官方基线提升主线。E32发现原参考完整qpos保留rod球关节key pose，内部connect分量残差8.92–19.00cm，限制动力学分析；官方reset直接复制该qpos/qvel。E33只补全六根被动连杆的球关节姿态及其派生速度，严格保留root、29外部关节、六驱动电机、脚几何、CSV、策略权重、帧率及全部原评测条件。

本次包括装配验收及合格后的冻结官方035闭环六段对照；原E26作为同机/boot基线直接复用，不重跑旧条件。不是编码器训练或权重微调，也不能因初始化修正就宣称官方策略已改善。

## 方法、预算与门槛

使用编译模型connect的两端局部anchor，按当前位置将rod方向对准对应平台端点；SciPy单向量最短旋转在帧间沿用原扭转自由度，零偏置ball关节且只改球四元数。该扭转是确定性连续选择，不是观测到的真实连杆扭转。关节速度沿用官方向前mj_differentiatePos，含球/自由关节四元数。

装配门在执行前固定：六根rod、每帧连接点距离<=2mm，所有非rod qpos精确不变，root/外部关节/驱动qvel精确不变，可碰撞geometry位置/方向差<=1e-12，原脚最低点预算5mm。先运行旋转端点、静态速度、单位四元数三个合成对照；装配失败保存QUALITY_REJECTED且不启动策略，不放宽门槛。

通过后冻结035、seed0、30Hz原CSV→50Hz、无动作/参考延迟、每段499步，保留全状态/输入/接触采样；独立核验原跌倒门（根高<0.45m或roll/pitch绝对值>60度），原五段不倒须仍不倒、各段RMSE不得恶化超过2%，六段均不倒且等权平均RMSE更低才通过本开发对照。初始完整1570维策略输入必须与E26完全相同；后续闭环状态可因该干预而不同，不能要求整条输入相同。仍须未见会话和能力保留验收，开发通过不等于最终目标或部署许可。

仅ZHY既有环境，一次900秒、串行计算、CPU90/GPU85摄氏度连续两次5秒保护、4GiB保留、首异常停止；无开机恢复、无重复、0optimizer。原hp3090只Git/标准库传输，Mac只小代码/README/清单，禁止真机与驱动/功率/BIOS操作。

## 版本与命令

代码script/a3/smpl_closed_reference.py、smpl_closed_reference_entry.py、smpl_closed_reference_trial.py；登记固定E32/E26全部输入/源码/桥SHA及本次代码，生成装配NPZ另存SHA，实际入口再次核对，旧参考不覆盖。

```bash
cd /opt/data/private/hechengzhu/ZHY
data/environments/a3-sonic/bin/python -m script.a3.smpl_closed_reference_trial register
data/environments/a3-sonic/bin/python -m script.a3.smpl_closed_reference_trial run
```

实验data/experiments/a3_zhy_smpl_assembly_20261013_E33/，日志logs/a3_zhy_smpl_assembly_20261013_E33/，清单data/manifests/a3_zhy_smpl_assembly_20261013_E33.json。不重置或重用登记/运行入口。

## 当前结果与保存

登记14:09:44，执行14:09:48–14:10:19。3合成控制通过；六段2994帧/17964内部连接验收全部通过，非rod qpos、受控qvel、所有可碰撞geometry位置/方向差均0，脚最低点沿用E26，原5mm门未变。

| 开发序列 | 原连接点最大距离cm | 补全后最大距离m |
| --- | ---: | ---: |
| zj_walk | 16.584 | 2.242e-7 |
| zj_warmUp | 15.056 | 3.447e-7 |
| fjw_walk | 9.411 | 2.305e-7 |
| zls_walk | 20.038 | 1.632e-7 |
| yy_walk | 16.549 | 1.669e-7 |
| zzx_walk | 19.102 | 2.148e-7 |

该表为连接距离范数，与E32的最大坐标分量定义不同；不要把两个数字直接当方法间改进率。补全后全部<固定2mm门，最大全域3.447e-7m。只证明内部几何装配合格，不证明动力学跟随或跌倒改善。

冻结闭环首段在写reference入口回执时FileNotFoundError：OUT/entry不存在，尚未进入策略步。该失败不是数学/动力学发散；原E33 STOPPED、不重置或复跑。0完整回放、0策略步/optimizer；未单独统计权重加载，不能据0步宣称一定未读权重。独立失败归档data/manifests/a3_zhy_smpl_assembly_20261013_E33_failure.json保存装配结果/日志SHA，原STOPPED清单保持原样。

8健康样本同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，CPU40/GPU26摄氏度，最低可用9,457,995,776字节、OOM0，无新关键内核/遥测异常，最大间隔5.188秒，进程已结束。代码基线4c1a2b0269dc00712f2a679806edc7ba5e0d79da。

本README/索引/current state、原STOPPED小清单和独立失败归档已保存，待scoped audit/推送核验；原始装配NPZ/数据/模型/日志仍LOCAL_ONLY，未验证网盘备份。E32结果91f42fe已核对GitHub与源机/ZHY，生成清单scoped stash保留。

## 下一步

下一独立E34复用六份已验收装配NPZ（0装配复算），提前创建并实际测试回执目录后完成六段冻结闭环。它是封存的策略步前入口错误的新执行分支，不复跑E33或原E26条件、不更改三个旧模块及历史账本。原硬件/温度/OOM/计算保护保持。若内部装配未改善跟随，下一选择接触/根运动参考修正或有官方行为约束的小微调，不用checkpoint数证明提升。恢复[E34入口](README_A3_SMPL_ASSEMBLY_FROZEN_20261013.md)。
