# Q1原参考接触时段与足端目标核查

2026-10-29 05:15北京时间。精度与脚底几何诊断均已结束、无GPU作业。本步独立CPU只读核原重定向保存的contact/contact_local_points与实际参考足端运动，不仿真/策略推理/optimizer、不重转或筛数据。

## 核查依据

实际读到pinned tools/retarget_x2.py有源脚踝速度<.2m/s且源足底高度<.025m的接触标记、3帧中值和短段处理；接触段用中段脚底材料点建立世界XY锚、z.002m，Mink SupportPointTask cost200，同时足端位置cost20（非接触70），还有限位/逐关节速度截断及最后全身地面投影。这是软约束，不等于接触力或动态可行。Q1 retarget_soma.py只适配关节/球体几何/相对pelvis静态坐标，保留上述逻辑。

此前几何近地速度较高，不能直接推断脚锁失效或参考不可行。新工具script/project_tools/audit_q1_reference_support.py读取330原CSV与metadata逐SHA，native30Hz只在接触段内部测同一保存材料点速度、段内锚漂移和高度。另用原MotionLib50Hz与原RF15 FootReference准确重建其stance/swing启发式，接触标记保守映射为相邻源帧均为接触，记录二者差异；不把任何一方当原生承重真值。

## 验证与边界

平移速度、绕固定材料点旋转、段内部排端点与30到50Hz保守映射合成反例通过，真实审计尚未运行。输出独立Q1_REFERENCE_SUPPORT_AUDIT_20261029，660足行/固定三片时序及PNG/PDF，0积分/推理/optimizer。工具和报告独立于已完成实验冻结来源，不修改旧runtime/reference。07:00停止新物理探索、09:00交数据视频；用户采纳null。

如发现标记差异，它只支持下一步相位目标假设，不隔离历史训练退化，也不自动恢复RF15或填补未用预算。原所有失败证据、原300/30与展示片保持。
