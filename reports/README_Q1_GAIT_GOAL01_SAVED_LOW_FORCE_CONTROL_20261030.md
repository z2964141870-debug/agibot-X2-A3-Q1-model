# Q1低力贴地样本中的关节指令与响应

日期：2026-10-30，北京时间。阶段：GAIT_GOAL01_SAVED_LOW_FORCE_CONTROL。状态：保存数据CPU分析与小工件交付完成；普通自然步态目标未达，goal ACTIVE，用户采纳null。

## 目标与范围

上一轮确认：PPO最终起走右脚存在低力样本段，但仍贴地。本轮检查同一批逐重复样本的髋pitch、膝、踝pitch位置目标和实际20ms变化，区分“关节完全不动”与“关节虽动但未组成脚底离地”。沿用SuperMimic两条独立路线，不新增训练或物理运行。

此前关节时序CSV按三次重复逐帧取中位数，不能直接拼接单次受力。因此读取原15份NPZ及其15份结果JSON，覆盖45段固定动作、12015状态；复用已发布脚底时序和静态目标几何。生成1620行关节条件统计、540行脚底条件统计、2157行低力关节样本，后者为719个足样本各3个关节，不是新步数或独立训练重复。

0新控制环境、积分、FK、接触/动力学求解、策略前向、optimizer或训练。原底座301.1074845249823秒、PPO600.0709074319457秒，合计901.178391956928秒；原physical_regression、PPO原媒体失败和独立补交付均保留，不恢复或补预算。

## 定义与对齐

沿用原声明摆脚且参考全碰撞脚底Z>=10mm窗口；其中本脚力<=5N且对侧>5N称低力，本脚保存后态脚底<=5mm称近地。力仍是最后子步脚地接触world-Z分量绝对值之和，不是体重占比或连续卸载。逐重复独立统计，不能用不同阶段选出的低力集合直接估计训练因果收益。

pre关节来自initial或前一保存后态，避免用本步后态代替起点。位置目标和pre位于t，力求解样本位于t+15ms，post与下一参考关节位于t+20ms。目标虽在post时钟图上并列，仍是产生该后态的起点设定值。它是通过位置偏置产生控制力的PD设定值，不是下一帧姿态、足部轨迹或电机实际力矩。

单关节统计为：

- 位置请求偏差：target(t)-pre(t)的RMS。
- 观察运动：post(t+20ms)-pre(t)的RMS。
- 残差：原确定性head保存输出的RMS；只在最终动作边界内部扣回，保持观察到的教师目标不变。
- 朝参考方向：当|reference(t+20ms)-pre(t)|>.001rad时，比较目标偏差或观察运动与该差的符号。此为角度几何诊断，不是脚底Z方向、抬脚动作正确率或PPO奖励正确率。

## PPO最终起走：关节在动，脚底仍低

右脚低力且近地样本数14/14/16（repeat0/1/2），保存后态脚底均值为1.195/1.294/.944mm。

| 右腿关节 | 位置请求偏差RMS，rad | 20ms观察运动RMS，rad | 残差RMS，rad |
| --- | --- | --- | --- |
| 髋pitch | .572/.721/.733 | .0511/.0596/.0578 | .00368/.00315/.00328 |
| 膝 | .184/.161/.171 | .0617/.0673/.0671 | .00568/.00505/.00435 |
| 踝pitch | .263/.335/.364 | .0634/.0766/.0870 | .00419/.00507/.00452 |

这组样本的观察运动RMS约2.9至5.0度/20ms，不能将现象描述成这三个关节完全不动；各关节运动仍没有组成脚底离地。位置偏差较大不意味着应在一步内到达，也不能据请求与运动幅度之比计算电机增益或效率。

髋pitch在这组样本中，位置请求与实际运动朝下一参考角方向的比例均为92.9%/71.4%/56.3%；即便某个关节朝参考角移动，也没有保证足部抬高。膝的位置请求方向比例42.9%/57.1%/43.8%，实际运动比例78.6%/50.0%/50.0%；踝分别21.4%/42.9%/50.0%及28.6%/50.0%/50.0%。这些结果保留重复差异，不能用单关节符号判奖励方向或接触约束失败。

最终PPO在这些时刻的三个残差RMS为约.003至.006rad，并未接近.15rad半径；它相对当前教师位置偏差很小。但位置目标偏差不是所需最优补偿，以上不证明放大半径或训练更久会有效。

## 底座与其他动作

底座最终起走右脚对应11/11/10个低力近地样本，脚底均值1.274/1.216/2.092mm。髋观察运动RMS为.0681/.0587/.0768rad，膝.0526/.0653/.0550rad，踝.0837/.0532/.0852rad，同样有运动却保持近地，残差始终为0。

底座停走与PPO停走的关节条件统计分别保留，不把它们混成同一结果；尤其上一轮底座停走存在部分低力时离地，本轮“低力且近地”的选择自然排除了这些离地样本。该条件表不能据此反称整片从未抬脚，也不替换原330库、固定片完整率、姿态和上肢门。

## 复用静态目标几何的边界

PPO最终起走这些低力近地样本中，完整目标静态脚底Z均值为-10.575/-1.717/+.588mm，扣回残差后为-10.648/-2.068/+.212mm。同根参考关节的静态脚底为6.667/2.035/3.986mm；本轮只是条件拼接，未重新执行FK。

目标静态配置保持控制起点根姿态，不满足接触、速度或下一步可达约束，部分目标角可超模型范围；该组目标腿任一角越界比例为0/7.14%/0。不能将负Z当作控制器明确命令踩进地面，或把静态残差差值当作实测抬脚收益。它说明只看目标角、只看某关节是否在动，均不足以解释足部结果。

## 实现与验证

基础提交eea99281e0248a670035060b91b4759032ed957a；独立工具script/project_tools/analyze_q1_saved_low_force_control.py，最终SHA256 6cd749f4ab831ecce72fb7902248d0175194860640240df8b710fbb9c2ecb9e6，对应test_q1_saved_low_force_control.py。没有修改原训练、奖励、控制、参考或逐子步观测源码。

首次7项CPU测试通过，但首次实际离线入口在读取清单时发生TypeError：新入口假设artifacts.files，旧target/control清单使用直接artifacts列表。尚未读取原NPZ、尚未创建输出目录；原错误日志和旧源码SHA e6b7211358968b730a6187dbc1e389b7d482e971d0ce973300283a23429e9b14保留。修复后显式支持两种已发布结构，并为两结构的大小/SHA核验及篡改拒绝补回归；最终8测试通过、无告警。该CPU入口修复没有重启训练、仿真或旧物理预检。

最终完整保存分析仅执行一次，300秒外限内正常退出0、状态SAVED_LOW_FORCE_CONTROL_READY。14项脚本/清单/CSV/summary/原控制文件及34项轨迹、结果JSON、selection/library均在前后保持大小SHA；15份NPZ匹配原历史保存身份和已发布control来源，参考帧/原生力逐值匹配phase，脚底姿态和时钟逐样本匹配pose。原22关节顺序、默认角与action scale沿用已验收来源；最终动作边界内部才可扣回残差，全部样本均满足。

8个最终CPU反例覆盖：清单两种结构/篡改拒绝；请求向参考而实际反向；残差幅度与参考误差收益分离；空窗口/微小角差null；initial与pre无环绕；5N/5mm/10mm和参考窗口边界；非法维度或非有限值；静态姿态与force的帧/时钟/训练秒数/值一致性。

```sh
python3 -m unittest discover -s script/project_tools -p 'test_q1_saved_low_force_control.py' -v
```

已执行的服务器CPU命令，输出已存在且拒绝覆盖，不需重做：

```sh
env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout \
  --signal=TERM \
  --kill-after=10s 300s /opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python /opt/data/private/ZHY/script/supermimic_q1_hp/activate.py -u /opt/data/private/ZHY/script/project_tools/analyze_q1_saved_low_force_control.py \
  --phase-manifest /opt/data/private/ZHY/data/manifests/supermimic_q1_gait_goal_saved_phase_20261030.json \
  --phase /opt/data/private/ZHY/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030 \
  --pose-manifest /opt/data/private/ZHY/data/manifests/supermimic_q1_gait_goal_saved_target_pose_20261030.json \
  --pose /opt/data/private/ZHY/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_TARGET_POSE_20261030 \
  --control-manifest /opt/data/private/ZHY/data/manifests/supermimic_q1_gait_goal_saved_control_usage_20261030.json \
  --control /opt/data/private/ZHY/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_CONTROL_USAGE_20261030 \
  --base /opt/data/private/ZHY/data/supermimic_for_q1/Q1_GAIT_GOAL01_BASE_PHASE_SOLE_E4096_MAX20M_V02_20261029 \
  --ppo /opt/data/private/ZHY/data/supermimic_for_q1/Q1_GAIT_GOAL01_PPO_SOLE_Z_E4096_MAX20M_20261029 \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_LOW_FORCE_CONTROL_20261030 \
  --source-base-commit eea99281e0248a670035060b91b4759032ed957a
```

## 工件与发布

输出data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_LOW_FORCE_CONTROL_20261030，共8工件2952609字节：三CSV、summary.json、底座/PPO各一PNG/PDF。权威工作副本、Mac主根、25386逐大小SHA一致，两PNG均已目视无文字遮挡。图为repeat0逐帧，红点标记最终低力近地样本；全部重复与条件在CSV，不取逐帧中位数替代单次记录。

日志在logs/session_records/q1_gait_goal_saved_low_force_control{,_v02,_tests,_tests_v02}_20261030.log，保留首次错误与最终正常结果。权重和原大NPZ仍SERVER_LOCAL_ONLY，未取回；未重做旧视频、保存审计或新330评测。代码、清单、报告和索引显式stage/payload审计后推Q1，独立40位远端及记录三端验收以publication旁收据为准。

## 未解条件与下一步

本轮进一步排除“低力时整条腿完全不动”这一过度简化描述，却未区分关节时序、根运动、接触约束、实际力矩、世界模型和学习目标中的单一原因。关节角朝参考靠近、目标静态几何与确定性残差小，均不足以判断实际足部加速度或干预效果。

有区分力的下一项仍是已准备的独立逐5ms观测，核实际执行器力、根/关节运动和足地接触在卸载前后的同步关系。既有观测范围问题仍待用户答复，不重复询问或将goal续行当批准；V02 plan282源/86121字节/SHA c94cf9489c317ea6a2ca431af898bc4f8704e698067c5e927d17a389b1234942保持，实际观测未启动。

上一goal回合与本回合均为progress；本轮分析交付不等于步态目标完成，goal ACTIVE、用户采纳null。原训练V02授权已用，不新物理/推理/训练、不恢复/重试/扩时或补预算、不启动agent，监督/缓存维护与其他自动任务保持。

[本轮清单](../data/manifests/supermimic_q1_gait_goal_saved_low_force_control_20261030.json) / [发布旁收据](../data/manifests/supermimic_q1_gait_goal_saved_low_force_control_publication_20261030.json) / [低力与侧倾](README_Q1_GAIT_GOAL01_SAVED_LOAD_TILT_20261030.md) / [双线原结果与视频](README_Q1_GAIT_GOAL01_RESULTS_20261030.md)。
