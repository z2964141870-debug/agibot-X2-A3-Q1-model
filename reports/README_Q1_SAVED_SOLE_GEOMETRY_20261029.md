# Q1已保存姿态的脚底几何诊断

2026-10-29 04:44北京时间，CPU准备完成、尚未执行真实330片审计。AST与远端help通过；独立小型MuJoCo模型核编译mesh/球体最低面0.9m、同一材料点平移速度1m/s/另一脚0、90度旋转最低面0.8m。只做FK，不建环境、不积分/推理/optimizer，不改变原参考或数据筛选。

既有足高指标是踝点相对参考踝最低5分位，已标注不等于脚底间隙。本次使用实际Q1所有启用的足部碰撞mesh顶点及角球表面，求相对原水平地面z0的真实几何最低高度；最低材料点在相邻姿态用同一局部坐标差分，避免最低顶点切换制造虚假位移。每帧几何近地不等于接触/承重，速度不等于已求解接触滑速，不用它设置新数据硬门或宣称动态可行。

输入为已完成ACTIVITY full330 baseline_repeat0.npz、evaluation_library.json和原scene.xml。原330条CSV逐SHA并按同MotionLib30→50Hz重建参考、核原长度；不读取物理失败后已reset的参考后缀。全部330只测参考，物理对比仅固定两走/抬手已存姿态、无新仿真。输出独立`data/supermimic_for_q1/Q1_SAVED_SOLE_GEOMETRY_20261029`的660参考足行+6物理足行及固定时序CSV/JSON，真实结果待执行。本步不能隔离过去训练退化或自动开启新训练。[准备收据](../data/manifests/supermimic_q1_saved_sole_preparation_20261029.json)。
