# Q1 f0机器人历史残差闭环检查

2026-11-08，北京时间。F0_PROPRIO_PAIR01两组训练已完成并发布；本页登记后续独立探索性闭环。当前仅实现与本地检查完成，服务器准入、新物理及视频尚待。原共享训练保持暂停，原final门和用户采纳null保持。

## 问题与范围

真实机器人历史组selected3000在保存验证状态的动作误差下降36.303%，对照组zero_history选0。此前COVERAGE01有相似的验证下降却更早摔倒，本次必须用同一参考和共同Primebot物理判断闭环是否改善。

先在CPU核对zero_history保存点：原R2 selected500的八个人体编码器张量/缓冲、归一化、decoder与teacher身份均相同，适配器末层精确为0，才复用旧七例0/7。不重复运行相同策略。非零causal_history selected3000才新增七例：三段名义前走/起走/停走，以及前走上的急停、反向、丢包、突发延迟。事件沿用原50/10/0参数，这是探索性对照，不冒充新留出事件。

新worker只改保存点重建，`action/start/preview_config`严格继承原CausalReviewSession；仍用原CPU状态转换，不混入CPU加速候选。冻结教师/decoder/归一化/动力学、参考时钟、目标、事件可见性与回退规则。GPU策略、共同Primebot CPU物理和NVIDIA EGL原速50fps媒体分阶段执行。每段记录完整worker、同步GPU目标路径和本机管道RPC时间，均不是硬件端到端延迟。

## 预算与来源

独立输出`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_F0_PROPRIO_PAIR01_REVIEW_20261108`。总1800秒/3GiB、每阶段1800秒；计原r2账本，进入前3671=1614+686+1371秒已全部SETTLED。沿用22GiB准入/24GiB退出、GPU18GiB80度、100GiB磁盘余量、private500GB、原非阻塞GPU锁及自有子树回收，不清缓存或恢复旧共享训练。

严格复用已保存的14条H4对照和7条original_f0失败，来源为COVERAGE01保存分析中的原始比较行。新七例与这21条合计28条；COVERAGE01本身七条失败不替代原f0控制。原78+6/84事件、R1旧物理和已完成训练不重做。

[计划](../script/supermimic_teleop/student_proprio_review_plan_20261108.json)、[controller](../script/supermimic_teleop/student_proprio_review.py)、[worker](../script/supermimic_teleop/student_proprio_review_worker.py)、[六项聚焦CPU检查](../script/supermimic_teleop/test_student_proprio_review.py)。训练执行的六文件不改。六检查本地0.082秒全过，CLI和编译通过；新增真实CUDA/物理/媒体0。

## Done与Todo

已完成独立实现和本地检查。下一先发布源码/不可变清单，再核服务器原运行时、真实选点CPU身份、旧来源、暂停、资源及原锁，最后唯一output/log/dispatch/进程防重单派。若缺回执先读实际状态，不能重复派发。

实际七例、全部守卫回收、来源终门和原账本结算完成后，仅读保存工件生成六宫图/对照表和原速视频。保留完整率、有效前缀/倒地、离地换脚、承重滑移、侧倾、上肢时序、动作突变/回退恢复和软件长尾；失败片不出全片误差或全组均值。只有探索性完整率改善后，再考虑训练前登记的新事件。不能据阶段文档、loss或平均软件耗时结算整项自然遥操目标。

最新发布与服务器镜像以独立publication收据为准；旧大包和全局索引镜像PENDING。
