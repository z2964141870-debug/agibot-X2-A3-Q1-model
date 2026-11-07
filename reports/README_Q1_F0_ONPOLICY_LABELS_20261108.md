# Q1：纯学生访问状态与教师纠偏标签

2026-11-08，北京时间。阶段PURE_STUDENT_LABELS01：先采集与审计，暂不训练。原共享训练保持用户暂停；32k教师、原final门与用户采纳null保持。

## 问题与决定

历史残差的保存验证误差下降36.303%，真实七例却仍0/7完整；前走和迟到前缀延长，其余五例变短。[完整负结果](README_Q1_F0_PROPRIO_REVIEW_20261108.md)已交付36成员证据ZIP，不能再据单步loss继续相同拟合。

旧AGGREGATE01执行`.5*teacher+.5*student`，并非纯学生访问的状态。此轮单一主要变化是实际行为分布：原causal_history selected3000完全驱动物理，教师H10仅在同一个实际状态上每两帧提供独立特权标签，绝不混入动作、学生特征或下一步机器人历史。参考时钟、事件可见性、原CPU转换、原CausalReviewSession动作路径均保持。

这是本项目的学生状态分布诊断，不能称已复现TWIST的联合BC/RL或GigaBrain整套BWM。[既有一手源码与迁移边界](research/2026-11-07/LOW_LATENCY_SUPERMIMIC_REBASE.md)继续适用。教师标签是否具有恢复能力仍未验证；标签存在或动作差较大都不能单独证明可以恢复。

## 独立数据与预算

从既有候选计划的original_jobs按原顺序取前六train组和前三validation组，九组互斥，不根据新结果挑片。不读取test载荷，不采集A022/A023/A024比较片，不打开已登记的新留出事件。动作与物理任务是原名义参考，输入可见性保留对应nominal/dropout/late_burst。

每片最多192步，短片维持原边界；总1699个新物理步、最多1699次学生前向加850次教师查询、850条样本。遇原终止条件立即停止该片，保留失败前缀，不用补齐/教师接管伪造完整性。当前与下一实际状态按同一trace严格配对，保存terminal_after_action，不跨终止拼接序列。

学生输入仍f0与历史。120→50Hz插值依赖、到达时钟、历史flags、缺失与fallback沿用已审计合同；教师未来依赖单列并标特权，训练数据不把它作为f0输入。离线人体预处理仍offline_unverified，真实人体或硬件端到端未测。

独立输出`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_F0_ONPOLICY_LABELS01_20261108`，600秒/128MiB、0新增Adam、0渲染，成本记原r2。进入前原账本4256=1614+1271+1371秒，全SETTLED；总14400、r2上限4500均不重置。22GiB准入/24GiB退出、GPU18GiB80度、100GiB余盘、private500GB、原非阻塞GPU锁与自有子树回收保持，不清缓存。

## 实现与验证

[controller](../script/supermimic_teleop/student_onpolicy.py)、[worker](../script/supermimic_teleop/student_onpolicy_worker.py)、[数据门](../script/supermimic_teleop/student_onpolicy_data.py)、[不可变计划](../script/supermimic_teleop/student_onpolicy_plan_20261108.json)。只新增清单外文件，不改已执行216来源或六训练文件。

六本地检查全过/.002秒，CLI与导入通过：来源互斥/固定选择；教师变化不能改变执行动作；状态/动作/标签及下一状态配对；split/终止/启动样本门；教师未来与学生可见性分离；未采样步精确继承原学生回应。真实采样和教师查询尚未派发。

先发布源码/五文件清单，再服务器原运行时合同、1111既有来源及新输入、选点、资源、暂停、锁和账本准入。唯一输出/log/dispatch/精确进程全无后单派；缺回执先查实际结果，禁止重发。

## Done与Todo

已完成独立实现与本地检查，前一负结果和小记录已发布；真实新调用/物理/Adam当前0。下一执行一次有界纯学生标签采集，保留每片完整率、失败时间、标签与行为差异、支持可见性和真实转移。

只有来源/状态/标签及成本回收门通过后，才设计同容量/曝光的保存数据与on-policy数据对照，验证选点与旧锚点保持，随后独立真实闭环。序列目标需明确冻结WM有效范围和终止mask，不在本采集入口内隐式开训。新证据不足时可否定后续训练方向；不能仅凭此诊断完成结算整个探索。
