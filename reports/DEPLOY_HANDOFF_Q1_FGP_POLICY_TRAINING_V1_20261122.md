# Q1 FGP 遥操策略训练交接说明

日期：2026-11-22

## 交接结论

Q1 的 FGP 衣服输入已经打通到低层 human_smpl policy。当前上肢跟随已达到可用水平，机器人能够跟随穿衣者迈步，但下肢步态仍有拖步、相位滞后和动作不自然现象。

当前主要优化对象是 **human_smpl encoder 与下肢 policy**。FGP 服务端通信和 A2 适配器已经完成主要链路修复，不建议先修改 FGP 服务端或重新调上肢。

## 已完成的部署链路

- A2 FGP V2 已合入隔离候选目录
  - `/home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122`
- 使用 `FGPSMPLStreamV2`、`FGPHttpSourceV2` 和 `Q1FGPFallbackGuard`
- FGP 采用 50 Hz paced FIFO 消费
- 短断流有 220 ms 宽限
- Planner / human_smpl 回退时自动重锚
- Planner 与 human_smpl 热切换保留 tracker 历史
- 保留同步记录器
- 低层退出保护已修复
  - 低层模式没有 `SetMcAction` 时，`q` 进入零速度 Planner 安全保持
  - `x` 才执行强制退出
  - 键盘进入 cbreak 前清空终端输入队列

候选桥接文件当前 SHA-256：

`cda69e3eecac80e7dc4d4a64290486c835d498a01adabecb36762424b072cd0c`

Q1 分支当前提交：

`c9d20dd62972855a32fcb24e7a6a3312b47598bf`

## 真机 FGP 运行数据

记录目录：

`/home/yu/projects/Q1_deploy/real_logs/q1_fgp_control_v2_real_20261122_run02`

- FGP 有效帧：`6603`
- FGP 状态：全部为 `tracking`
- `realtime_valid=true`：全部有效
- FGP 推理延迟：P50 `27.4 ms`，P95 `40.3 ms`，最大 `98.2 ms`
- human_smpl policy 帧：`9828`
- policy 看到的 FGP 年龄：P50 `31.9 ms`，P95 `53.0 ms`
- policy target 与真机关节误差：P50 `0.207 rad`，P95 `0.474 rad`，最大 `1.029 rad`
- 真机关节速度最大值：P50 `0.623 rad/s`，P95 `2.674 rad/s`
- target 单帧变化上限：`0.2 rad`

本轮退出时出现了 `PASSIVE_DEFAULT` 服务不可用，旧退出逻辑发送了 `stiffness=0, damping=5`，机器人在自主承重状态下摔倒。状态记录显示退出前倾角约 `3.4°`，没有证据表明 FGP 或 policy 在退出前先发生失稳。该事故属于退出流程问题，不作为步态模型失败样本。

## 离线仿真因果对照

输入是上述真机 FGP 记录中从 `175.3 s` 开始的同一段动作。所有对照使用同一个 human_smpl ONNX、同一个 Q1 MuJoCo 场景和 2 ms 物理步长。

| 适配器 | 根反馈 | 仿真时长 | 结果 | 最大倾角 | target error P95 |
| --- | --- | ---: | --- | ---: | ---: |
| 旧 FGP adapter | CGFA | 12.00 s | 失稳 | 67.31° | 0.685 rad |
| 旧 FGP adapter | zeroed | 66.58 s | 失稳 | 66.58° | 0.543 rad |
| A2 FGP V2 | CGFA | 119.72 s | 完成 | 13.50° | 0.515 rad |
| A2 FGP V2 | zeroed | 68.32 s | 失稳 | 68.04° | 0.563 rad |
| A2 FGP V2 | MuJoCo 真值根反馈 | 119.72 s | 完成 | 14.46° | 0.556 rad |

输出目录：

- `/home/yu/projects/Q1_deploy/verification/q1_fgp_run02_sim_v2_cgfa`
- `/home/yu/projects/Q1_deploy/verification/q1_fgp_run02_sim_v2_truth`
- `/home/yu/projects/Q1_deploy/verification/q1_fgp_run02_sim_v2_zeroed`

## 对训练端的判断

### 已排除的首要原因

- FGP HTTP 服务没有出现持续断流
- A2 的 paced FIFO 没有造成明显仿真失稳
- A2 的 CGFA 根反馈与 MuJoCo 真值反馈表现接近
- 当前问题不优先指向上肢特征或上肢 policy

### 当前首要问题

human_smpl 输入的 430 维特征包含 10 帧历史、骨盆运动、身体旋转、局部速度和 9 条骨骼方向。下肢部分只有髋到膝、膝到踝的骨骼方向，没有显式表达：

- 左右脚踝高度
- 足端水平速度
- 支撑脚与摆动脚状态
- 触地速度
- 左右脚步态相位
- 足端滑移置信度

上肢可以依靠骨骼方向直接跟随。下肢缺少接触和相位语义时，容易出现脚步拖滞、抬脚时机错误、左右腿配合不自然和骨盆上下振动。

## 推荐训练方案

### 第一阶段，冻结上肢并做相位诊断

对同一段 FGP 记录离线扫描：

- 当前相位
- 输入提前 `20 ms`
- 输入提前 `40 ms`
- 输入延后 `20 ms`

比较左右脚抬脚时刻、落脚时刻、足端滑移和 target error。该扫描用于区分输入延迟和模型本身的步态问题。

### 第二阶段，增加下肢接触语义

训练新的 human_smpl tokenizer 或下肢 residual 分支，增加脚部相关特征：

- 脚踝高度
- 足端水平速度
- 膝角速度
- 支撑置信度
- 摆动相位
- 触地相位

可以保持 tokenizer 输出 token 维度不变，继续复用上层 token head。上肢输入、上肢 token 和上肢 policy 参数保持冻结。

### 第三阶段，增加下肢专项损失

优先加入：

- 髋、膝、踝关节跟踪损失
- 支撑脚水平滑移损失
- 摆动脚离地高度损失
- 触地速度损失
- 左右腿相位一致性损失
- 骨盆垂向振动损失
- 躯干倾角损失

训练数据需要加入真实 FGP 噪声、`20–60 ms` 延迟和短暂丢帧。不要只用干净 SMPL 参考训练后直接回真机。

## 训练验收指标

下一版模型先在仿真验收：

- A2 + CGFA 连续运行时间不低于当前 `119.72 s`
- target error P95 相对当前 `0.515 rad` 明显下降
- 左右脚相位与 FGP 参考一致
- 支撑脚滑移下降
- 摆动脚不拖地
- 骨盆垂向振动下降
- 不牺牲当前上肢跟随质量

## 不建议的处理顺序

- 不要先删除 CGFA
- 不要先修改 FGP 服务端防抖逻辑
- 不要对全身 policy 重新训练并覆盖已经合格的上肢
- 不要直接用更强的目标平滑掩盖下肢相位错误
- 不要把本次退出摔倒记录作为 policy 失败样本

本报告总结了截至 2026-11-22 的部署和离线验证结果。当前还没有完成新的下肢专项训练，也没有用新权重进行第二轮真机验收。
