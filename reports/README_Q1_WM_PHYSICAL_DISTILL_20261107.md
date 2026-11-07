# Q1当前与历史学生：冻结FP32世界模型物理蒸馏

2026-11-07。独立实现及五项CPU检查已成，真实GPU训练尚为0；须完成发布、原运行时与来源/资源准入。原共享训练继续暂停，采纳null。

## 用户版done与todo

done：同权重FP32精度诊断通过88/88，原BF16仅4/88的失败保留；下一项只检验加入物理特征损失能否改善当前与历史学生。

done：核查原PAIR02控制发现草案学习率0.0003与原0.0001不同，已经在任何训练前修订为0.0001、seed2026110703，其余更新/批次/验证与原控制相同。旧草案及证据ZIP作为历史保留。

todo：发布独立实现并在原运行时检查，准入后唯一有界训练；交付六宫图和实际选中策略。若仍选中同一初始策略，复用既有0/7失败，不重复同策略物理测试。

## 假设与边界

本项目独立假设：冻结一步世界模型在同状态下比较教师与学生动作的物理结果，可以改变动作蒸馏的优化几何。它不提供缺少的未来信息，不是GigaBrain损失的忠实复现，也未证明此前f0倒地由BF16造成。[精度结果](README_Q1_WM_PRECISION_20261107.md#真实完成结果)只认证保存样本上的数值一致性。

冻结原32k teacher、部署共享decoder、WM全部权重与buffers、原R2当前目标统计。独立同权重影子仅关闭BF16，TF32关闭。学生仅f0/history，部署不增加WM或辅助头。

只加`0.1 * physical_loss`：在相同保存实际状态和名义context下，教师目标为WM(state, saved_teacher_action)的detach物理特征；候选为WM(state, student_action)。二者都使用同一个detach当前状态坐标系、原std和feature_presence。末29命令缓存严格排除，缺失关节输入及导数贡献为零。不能把不同混合行为动作产生的日志下一状态当作学生或教师动作的反事实标签。

## 公平控制与来源

复用完成PAIR02的action-only，不重训控制。原R1与AGGREGATE01 train/validation、终止mask、clip/frame、f0/history/teacher_action原行保持，初始化encoder SHA、全部验证点batch SHA与24源曝光须精确相同，初始动作风险容差2e-7。只用validation动作风险选点，含update0；不读test或新增采样。

原控制设置：3000次Adam、batch128、lr0.0001、seed2026110703、clip1、每500次验证。最终每个origin:clip训练曝光16000，共384000条抽样；六个Adam计数均须3000。原始source与fit/trace/provenance按大小SHA核同，运行时另外核实际encoder初始SHA、冻结teacher/decoder/WM前后SHA。

## 独立实现

四个新执行文件`wm_physical_training.py`、`wm_physical_worker.py`、`wm_physical_fit.py`、`wm_physical_loss.py`，不改原178执行或200冻结来源。控制器沿原非阻塞GPU锁/guard/持久成本账本，只计入r3；worker复用原控制的转移校验和归一化，按source/frame精确连接保存实际状态，0新物理。

五项聚焦CPU检查已全过/0.675秒，包括：草案学习率错配拒绝、命令缓存与缺失关节排除、冻结与detach门、保存行连接/无未来输入变化，以及随机CPU夹具3次真实Adam与历史控制批次/曝光/选点一致。夹具不是32k GPU训练。新训练六宫图入口已实现，实际生成待真实记录。

唯一新训练根：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_WM_PHYSICAL01_20261107`。启动前核output/log/dispatch/receipt/实际进程，不能因观察无回显重复派发。

最多3000反向/4096次WM.step、1740秒拟合、1800秒阶段保护、1GiB输出。22GiB准入/24GiB退出、GPU18GiB80度、100GiB磁盘余量、private500GB及原设备/锁/自有子树回收保持；R2缓存边界复用，不再次advice/drop或删文件。累计1463/14400秒、r3已342/7200秒继续，不重置。真实训练前须发布源码/新清单、原运行时检查、原身份暂停/资源及来源准入。

## 后续验收

若选中策略发生实质变化，另行有界闭环复核名义与stop/reverse/dropout/late_burst，比较完整率、脚底离地/换脚/支撑滑移、侧倾、上肢时序、动作突变和恢复。验证动作MSE、WM物理MSE或loss下降都不代替自然遥操或硬件端到端验收；相同选中策略不重跑历史失败。

本阶段发布/部署/实际准入和结果分别以收据为准；不调用goal工具、不新agent。33391原Linux自主复用，断开后按用户原样source及ssh入口自行尝试，不询问用户代测连接。
