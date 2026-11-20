# Q1 独立倒地起身分支

Q1 在 CPU MuJoCo 3.11 的固定 24 个倒地初态中成功 22 次。5 次行走踢倒后，均能起身、交回冻结的 Pure V2b，并再稳定行走 3 秒。训练和验证均在仿真中完成。这里的步宽指两足在航向坐标系下的平均横向距离。

## 验收结果

| 测试 | 条件 | 实测 |
| --- | --- | --- |
| 固定 E1 | 6 秒，末尾连续 50 个 50 Hz 控制步同时满足骨盆高度 ≥0.39 m、直立余弦 ≥0.85 | 俯卧 7/8、仰卧 8/8、侧卧 7/8，合计 91.67%。原始 Pure V2b 为 0/24 |
| 冻结后补测 | 种子 20261120，同一生成器和标准 | 俯卧、仰卧、侧卧各 8/8，合计 24/24 |
| 动态接力 | 向左右、前后、斜向施加约 2.7 m/s 的水平速度突变。确认身体触地、倒下、门控恢复和交还后的 150 个稳定步 | 部署入口 5/5，交还后平均步宽 14.48–20.66 cm，位移 1.03–1.91 m，二次跌倒 0 |
| 平地回归 | 原始 V2b 与并联门控在相同闭环中运行，核对原始动作、PD 安全目标和力矩的字节流与最终指标 | 直线 554 步，步宽 19.64 cm；8 字 1,339 步，闭合 22.80 cm。全程门控为零 |
| 落箱 | 复现上游 20 次原地面，用同一条件比较主干与门控。另测硬地 20 次和扩展 50 次 | 原地面 19/20；同组硬地 20/20；50 次硬地 49/50，6,700 步。每组两路逐字节一致。扩展 50 次包含前 20 个初态 |
| 安全与平滑 | 检查每步 22 关节估计 PD 力矩与限幅；动作变化量为相邻已发送归一化关节目标的平均绝对差 | 主 E1 和两套接力评测 PD 越界均为 0。主 E1 动作变化量总体均值 0.0948、最差单条均值 0.1560，直线/8 字为 0.1301/0.1356 |
| ONNX 导出 | 3 条实际姿态记录，重建速度和历史，连续比较 PyTorch/ORT，覆盖等待、翻身和起身状态 | 900 次调用中，目标最大绝对差 2.38×10⁻⁷ rad，递归状态差为 0 |

固定 E1 种子为 20261119，与原始任务一致。第 6 号俯卧和第 19 号侧卧都曾达到连续一秒，但在终点只分别连续保持 24 和 6 步，均计失败。成功样本首次完成一秒保持的中位时间为 4.94 秒。公共入口与固定 E1 逐试次 JSON 完全相同。固定测试在门控迭代期间已被查看，另列的补测仅覆盖同一初态分布。补测最大单条动作变化量为 0.1718。

接力同时验证了全身 FK 和部署入口的双腿 FK。全身 FK 的 5 条平均步宽为 17.73、20.08、17.01、20.71、19.79 cm。独立腿链的浮点运算顺序会改变接触轨迹，部署入口的相应结果为 14.48、20.66、15.68、20.46、19.73 cm。交还后的所有保存姿态重新计算碰撞，各条仅左右踝部的足底碰撞体接触地面，最低直立余弦 0.969，门控重新接管次数为零。部署入口五条接力的动作变化量总体均值为 0.1458，最差单条均值为 0.1607。

使用硬地接触参数 `priority=1, solref=[0.01,1], solimp=[0.99,0.999,0.001]`，每个静态初态松弛 20 个物理微步。轨迹记录的穿透深度取当步所有碰撞几何对的最大值。固定 E1 的 P95 和峰值分别为 4.75 和 26.58 mm，部署接力为 6.40 和 29.20 mm。视频附该数值。控制器仅使用 IMU、关节和已发送动作，骨盆真值与碰撞记录只用于打分和显示。

## 推理延迟与验收范围

在 yu-3090 的 i7-12700KF 上，将进程绑到一个性能核，ORT 的算子内和算子间线程数均为 1。五条原始接力重跑，计时完整部署入口，包含当拍实际调用的 FK、CGFA、参考重建、门控、ORT 和 PD；物理仿真在计时区间外。该次运行的评分与固定原始归档的复核逐字节一致。

| 原始接力实走路径 | 样本 | P99 | 最大 | ≥2.5 ms |
| --- | ---: | ---: | ---: | ---: |
| 完全起身接管 | 1,231 | 0.211 ms | 0.324 ms | 0 |
| 平地跟踪 | 1,551 | 4.177 ms | 4.755 ms | 860 |
| 余弦交还 | 145 | 4.225 ms | 4.227 ms | 138 |

另对四种已构建的输入流测量门控、推理与安全 PD，每路径热身 150 次、记录 1,000 次。该微基准排除参考特征准备和物理步。结果中的 P99 仅适用于相应机器与调度方式。

| 运行条件 | RobotPose 跟踪 P99 | 完全起身 P99 | RobotPose 交还 P99 |
| --- | ---: | ---: | ---: |
| i7 单核，连续调用 | 3.711 ms | 0.121 ms | 3.655 ms |
| i7 单核，50 Hz 休眠等待 | 3.931 ms | 0.841 ms | 4.298 ms |
| i7 单核，50 Hz 同线程忙等 | 3.382 ms | 0.398 ms | 3.420 ms |
| Apple M5 单线程，连续调用 | 0.517 ms | 0.080 ms | 0.597 ms |
| Apple M5 单线程，50 Hz 休眠等待 | 3.761 ms | 1.408 ms | 4.259 ms |
| Apple M5 单线程，50 Hz 同线程忙等 | 1.552 ms | 0.186 ms | 1.693 ms |

M5 忙等时四模态全部路径的 P99 均小于 2.5 ms，9,000 次中有一次 HumanSMPL 交还耗时 2.798 ms。忙等占用一条 CPU 线程，macOS 由系统调度核心。Linux 50 Hz 两组使用核 12、进程自身的利用率提示和 glibc 大页选项。Linux 不同计算任务并行的压力记录中，RobotPose 跟踪和交还 P99 达 32.49 和 35.24 ms；该次完全接管 1,000 拍中有 4 拍超过 2.5 ms。各路径的 P50、P95、P99、最大值与超限计数均在原始 JSON 中。

**已完成**：起身专家和门控在指定主机完成单核验收，静态起身、动态接力和逐位平地回归均有闭环原件。
**仍有限制**：yu-3090 上，包含冻结 V2b 的所有路径 P99 小于 2.5 ms 这一项未达标。单独测原始 RobotPose tokenizer 加主干头，默认 ORT 的 P99 已为 2.907 ms。关闭 ORT 权重预打包改变输出哈希，保留默认实现。所有分支的结论限于仿真和所列计时条件。

## 训练与模型

GPU mjwarp 的 1,024 个训练初态覆盖俯仰侧倒、屈膝和站立受推。独立 PPO 初版在高骨盆奖励下学出了倒立；增加足底承重和姿态约束后仍有蹲坐与弹跳。最终使用 CPU MuJoCo 搜索的翻身和起身关节轨迹，配合姿态反馈，再在 512 个 GPU 环境中训练神经残差。残差训练共 300 次更新，在开发集选择第 200 次更新的检查点，对应 3,276,800 个环境步。训练器仅优化专家、价值网络与探索方差。

发布模型为 `formal_03/q1_recovery_expert.onnx`（124,668 B），SHA-256 为 `0231328d60f07297b38a4079e01167cf9fec65fdb2c89302144ee65e0011eb92`。对应检查点 `ppo_residual_01/actor_00200.pt` 的 SHA-256 为 `655483c562b60429b6600c1bc74d693230090bf94bd54f6570d66e162feb1354`。早期 formal_01 的静态 23/24 和接力 4/5，以及 formal_02 的静态 12/24 和接力 4/5，保存在原目录供回溯。

## 接入

读取原始 V2b 归档 `q1_v2b_gantry_deploy_bundle.tar.gz`，SHA-256 为 `7b9a45346c873558db76480264ebeb2f8ba090ab92799911ea1b6a17d77ecbda`。项目中的独立规划器工作更新过共享展开目录的 PlannerCmd tokenizer。本项目保留单独展开的原始归档；全部四模态的最终复核以原始五张 ONNX 和原始部署配置为准。原始主干 PyTorch 检查点 SHA-256 为 `2604abbb2dd91f1f59b794755f971789e8a393a59f965e8157a750bd587128f7`。

将本目录、`dependencies` 和冻结 V2b 包的 `code` 加入 Python 搜索路径。运行时仅需 NumPy、ONNX Runtime、原始配置和机器人 XML。以当前测量姿态初始化 V2b，每个 episode 重置两个控制器。在 50 Hz 调用 `Q1RecoveryBranch.step`，传入机体到世界旋转矩阵、本体角速度、22 维关节位置/速度、上一拍安全目标换算的动作，以及下列两个回调。

- `base_step()` 读取当前时刻参考和 CGFA，运行冻结主干一次，返回安全目标、力矩和诊断字典。
- `reset_anchor()` 在交还开始时用测量姿态重建 CGFA 和因果参考窗口。完整实现见 `relay.py` 的 `reanchor`、`base`，以及 `reference.py` 的 `HandoffReference`。完全接管时冻结参考索引，先在新原点保持当前姿态，再平滑恢复原始参考的增量。

`step` 返回安全关节目标、估算 PD 力矩与 `g_rec`。下一拍的 `previous_sent_action` 为 `(target - default_q) / action_scale`。ONNX 输入为 `rec_obs_hist[1,10,74]` 与 `rec_state[1,26]`，输出为 `joint_target[1,22]` 与 `next_state[1,26]`。门控连续 3 拍满足严重倾斜或有效低高度时接管。回正、腿链高度及关节姿态连续合格 25 拍后交还，宽站姿需要 40 拍；门权重在接下来的 30 拍内按余弦退回零。

## 复现与原件

发布 ZIP 内的路径与服务器一致，含全部 24 条主 E1、5 条部署接力的轨迹、模型、代码和逐文件 SHA-256。复现物理评测需要原 SuperMimicHP 工作树的机器人与动作资产。

服务器目录为 `/home/yu/projects/SuperMimicHP/runs/h003_recovery_expert`。服务器证据直接位于 `formal_03`，Mac 副本位于 `evidence/formal_03`。下列命令从服务器项目目录运行，每个新评测输出目录需使用新名称，以保留原始结果。

```sh
py=../../.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export Q1_V2B_BUNDLE="$PWD/formal_03/pinned/q1_v2b_gantry_deploy_bundle"
expert=formal_03/q1_recovery_expert.onnx
"$py" check_base.py "$Q1_V2B_BUNDLE"
"$py" bench.py repeat_e1 --expert "$expert" --public-runtime
"$py" relay.py "$expert" repeat_relay --public-runtime --tracks formal_03/input_tracks
"$py" flat_audit.py repeat_flat --expert "$expert"
"$py" flat_audit.py repeat_box50 --expert "$expert" --hard-box-trials 50
"$py" bench_inference.py "$Q1_V2B_BUNDLE" ../h003_v2b_planner_cal/prebaked_clips_50hz/ho_lafan_walk2_subject3_50hz.npz "$expert" repeat_timing.json --cpu 8
```

逐次结果见 `formal_03/public_e1/trials.json`、`extra_e1/trials.json`、`public_relay/relay.json`、`flat/results.json` 和 `box50/results.json`。原始关节轨迹和遥测在各自的 NPZ 中。发布视频在 `formal_03/media_final`，50 Hz 控制每两帧取一帧，H.264、960×674、25 fps。三姿态各取该组第一个成功样本，原始基线使用完全相同的初态。三姿态对照为 18.00 秒，部署入口的接力为 13.92 秒。底部显示骨盆高度、倾斜余弦、门控权重、峰值力矩、接触穿透和门控时间曲线。

训练代码为 `pool.py`、`train.py`、`sequence_model.py`，轨迹搜索及反馈拟合代码为 `trajectory_search.cc`、`search_roll.py`、`search_teacher.py`、`fit_rise.py` 和 `fit_balance.py`。原始训练日志为 `ppo_residual_01/metrics.jsonl`。CPU/GPU 初态观测最大差 2.38×10⁻⁷，20 ms 后 qpos 最大分量差约 0.0477，最终闭环判定使用 CPU MuJoCo。


## Git 归档入口

[源码](../script/q1_h003_recovery_expert/) · [逐次验收数据与预览图](evidence/q1_h003_fall_recovery_20261120/formal_03/) · [完整工件路径和 SHA-256](../data/manifests/q1_h003_fall_recovery_20261120.json)
