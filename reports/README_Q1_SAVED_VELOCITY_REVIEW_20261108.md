# Q1保存轨迹的速度与恢复窗口

2026-11-08，北京时间。八锚点/四条train、validation动作、16原分叉；仅Mac只读新测量，没有新增策略调用、物理、渲染或训练。整体低延迟自然遥操仍未通过。

done：

1. 严格复用原16条教师/学生轨迹和已审计共同非终止区间，62原文件SHA前后同。将26个运动body的世界平移速度误差分成根平移、相对根运动及带符号交叉项，不忽略全局漂移，也不把交叉项当可丢弃项。
2. 采用结束于当前帧的20ms后向位移差。它是区间平均速度代理，既不是模拟器瞬时线速度，也不是GigaBrain原奖励的精确复现；未分析角速度。原始人体前处理仍offline_unverified。
3. 七个有80ms共同支持的锚点，教师最初80ms世界速度RMSE均更高，增加1.80%至166.47%；但各自完整共同区间的世界速度RMSE均更低。剩余A033/96只有40ms共同支持，不作可靠恢复或时序推断。
4. 七项新CPU合同全过/.011秒，实际分析与绘图1.316770秒/RSS136970240B。新分析六宫PNG/PDF、逐区间784行及配对CSV已生成目视，六成员384480B证据ZIP逐件CRC/大小SHA核同。GPU成本0，原5036秒台账未改；服务器当前值仍待观察。

## 逐锚点结果

数值为教师相对学生的世界body速度RMSE变化；正值较差，负值较低。共同区间按每一对原始非终止支持截断，不能把不同长度拼成全片均值。同片两个锚点相关，四片不代表全库。

| 来源/锚点 | split | 共同步 | 最初80ms | 完整共同区间 |
| --- | --- | ---: | ---: | ---: |
| A095/64 | train | 80 | +1.80% | -68.13% |
| A095/96 | train | 51 | +18.67% | -63.68% |
| A075/64 | train | 47 | +166.47% | -72.17% |
| A075/96 | train | 15 | +16.02% | -16.47% |
| A267/64 | validation | 80 | +28.03% | -60.05% |
| A267/96 | validation | 80 | +31.81% | -61.87% |
| A033/64 | validation | 37 | +46.84% | -33.25% |
| A033/96 | validation | 2 | 缺失 | -1.92%，仅40ms，不作恢复推断 |

预登记80/200/400/800/1600ms完整窗口分别有7/7/6/5/3对。仅在这些固定窗口中，A095两锚点首个较低值在200ms，A075/64及A267/64在400ms，A267/96在800ms；其他窗口不足或尚未更低。不能据此估计精确改善时刻，缺失不代表永不改善，更不能当成硬件延迟。

最后子步>=5N的承重模式逐区间保留，不等于整控制周期接触冲量或计划接触。脚底离地、换脚、承重滑移、侧倾、上肢误差和动作突变继续以[原匹配分叉报告](README_Q1_TEACHER_RECOVERY_RESULT_20261108.md)为准，本项没有重算或替代那些维度。

## 对下一项的影响

这说明同一批纠偏动作存在短窗口与较长窗口的排序差异。直接把首80ms速度误差作为唯一目标，可能排斥有后续收益的动作；这只是由保存证据提出的假设，未证明该瞬态必要，也未证明它造成先前f0拟合失败。不据此采用教师、增加真实延迟或放宽旧能力门。

todo：

1. 恢复原生第二跳后先取回已提交的人体条件分支CPU审计和原曲线，完成那一轮训练六宫图。本页新图是保存轨迹分析图，不能冒充那一轮训练交付。45014当前仅Windows，重登Linux再次明确Connection refused；不循环认证或把它概括成整桥接断联。
2. 先独立验证FP32冻结WM在同保存状态和动作前缀上的1/2/4/8/16/32/40步预测，报告终端/支撑边界、根与相对运动误差及梯度有效性；原来只有一步校准，不能直接用多步模型梯度训练。仅验证通过才设计恢复时序或速度目标的新消融，不继续堆同一动作MSE。
3. 新WM诊断须先实现、来源/重建context/同时间锚点及资源预算审计并发布，通过服务器原运行时准入后才派发。模型初态由保存79维观测及冻结FK重建，不冒充完整MuJoCo隐藏状态快照。0新物理/0Adam，原共享、final和采纳null保持。

## 证据与复现

- [执行与逐对变化原记录](../logs/session_records/q1_saved_velocity_review_completed_20261108.json)
- [固定测量计划](../script/q1_checkpoint_review/saved_velocity_plan_20261108.json)
- [独立分析入口](../script/q1_checkpoint_review/saved_velocity_review.py)
- [新诊断证据ZIP](../data/deliveries/Q1_TEACHER_RECOVERY_VELOCITY_REVIEW_20261108.zip)：384480B，SHA `ae0fe8d2ff776a841520093f3332761fb6f744f3e29d5fe36d69baa25bc8673f`。原62文件保留在旧包，不复制或改写。

![速度与支撑六宫图](../data/analysis/Q1_TEACHER_RECOVERY_VELOCITY_REVIEW_20261108/velocity_six_panel.png)

```sh
data/environments/q1-teleop-mps-20261106/bin/python data/repositories/q1-rf12-20261028/script/q1_checkpoint_review/saved_velocity_review.py \
  --packet data/analysis/Q1_TEACHER_RECOVERY_NUMERIC_20261108 \
  --prior-summary data/analysis/Q1_TEACHER_RECOVERY_FIGURES_20261108/summary.json \
  --plan script/q1_checkpoint_review/saved_velocity_plan_20261108.json \
  --output data/analysis/Q1_TEACHER_RECOVERY_VELOCITY_REVIEW_20261108 \
  --archive data/deliveries/Q1_TEACHER_RECOVERY_VELOCITY_REVIEW_20261108.zip
```

在Mac主根执行；已有输出会拒绝覆盖，恢复时复用本次已成结果，不重算。源码和计划先同步主根；服务器镜像PENDING，发布与独立远端核验以后续publication为准。实验结果不是实测人体、端到端或自然遥操验收。
