# 真实推理入口与速度字段溯源（目标轮5）

2026-10-22；状态：静态溯源与隔离数值探针完成，未新增模型推理/仿真/硬件连接。Windows本轮可达huohuo，huanyu经跳板仍banner超时；停止额外重试。

## 新证据
1. FGP ZIP的online_fgp+LIP.py中RealtimeLIPPredictor.step返回pose_mat与zeros(3)，之后可能根据foot-lock和显示模式调整平移。它不是历史Q1离线链中的独立学习式速度头。因此“FGP全部入口天然输出同样根轨迹”不成立；需要明确要复现哪个入口。只对已读此版本断言，不概括所有FGP。
2. 本地提供的G1 demo不是同一入口：其474行实际i_session.run ONNX，再SMPL-X/GMR与ZMQ。文件名/同目录不证明与LIP权重输出等价，未执行demo。不要把旧A3 checkpoint、G1消息协议与人体估计器混为同一条已验收链。
3. G1 demo的_compute_joint_vel用publish时time.perf_counter差分（295–310、333行），而不是源帧时间。提取并仅执行该算术方法，对相同合成轨迹q(t)=t：30Hz发布速度约1rad/s，15Hz发布约0.5rad/s，60Hz约2rad/s（alpha=1以隔离EMA）。验证是源帧步长固定、发布时刻改变时的数学行为；未测真实调度、接收器语义或闭环收益，不能叫已证实bug/根因。
4. process_data.py加载发布PT、去11槽中的槽5并clamp；所读real序列加载函数没有text时间到PT的重采样映射。30Hz为全局常量。映射仍需找上游采集/标注生成代码，不能猜对应。

## 工件与复现
探针 script/teleop_research/probe_publisher_timing.py；清单 data/manifests/teleop_publisher_timing_probe_20261022.json。只AST抽出已读_compute_joint_vel方法，不导入原demo模块，未启动UART、ZMQ或viewer。源SHA 75cc1cdaafde2561fbb0296db2a13abdd36b6f333d238d2ef1db8145a32cf326。

## 后续可操作分支
- 先明确当前待测衣服推理基线是原ONNX还是canonicalizer+LIP；逐个记录权重SHA及root生成方式，不先换网络。
- 对离线重放应分别记录source_time和publish_time，再由接收器协议决定速度参考的时间定义；真实在线最新帧模式不能盲改成固定30Hz。
- 连接恢复后在不连接硬件的回放入口做同源数据、同速率闭环对照，然后才判断时序改动有无实用收益。

## 三维评估
创新性：没有算法创新宣称。实用性：找到与遥操直接相关的速度字段时钟依赖及入口差异，可作为等预算对照变量。严谨性：用实际源码的单方法数值试验支持可复查结论，严格区分合成输入探针与机器人改善。所有新工件LOCAL_ONLY，未提交/推送，目标active。
