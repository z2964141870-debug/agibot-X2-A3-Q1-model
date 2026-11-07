# Q1 f0机器人历史残差闭环检查

2026-11-08，北京时间。F0_PROPRIO_PAIR01两组训练和本页独立七例闭环均已完成。真实历史组验证风险下降36.303%，闭环仍0/7完整通过；不采纳、不延长相同逐状态拟合。原共享训练保持暂停，原final门和用户采纳null保持。

## 实测结论与交付

七例、14阶段守卫均清洁完成并回收，source_integrity PASSED。zero_history选0，CPU已证与原R2精确相同，复用旧0/7；仅causal_history selected3000新增七例，14条H4与7条原f0结果严格复用。

| 案例 | 原f0有效步数 | 历史残差有效步数 | 参考步数 |
| --- | ---: | ---: | ---: |
| 前走A022 | 185 | 302 | 1031 |
| 起走A023 | 119 | 105 | 151 |
| 停走A024 | 176 | 175 | 402 |
| 急停 | 194 | 112 | 1031 |
| 反向 | 210 | 105 | 1031 |
| 丢包 | 204 | 124 | 1031 |
| 突发延迟 | 199 | 229 | 1031 |

两例前缀延长、五例缩短，全部失败。不同长短前缀的误差不能当等曝光优劣；失败片不填全片上肢、滑移、侧倾等数值，也不出整组均值。逐例前缀、脚底/换脚、承重滑移、侧倾、capture一秒上肢、动作突变和回退记录全部保留。四事件沿用50/10/0探索参数，未打开新登记留出事件。

独立仅读保存分析已2.931346秒完成，1111份来源前后同、28条比较、零新推理/物理/训练；原分析输出保留，不重发。完整worker去每段前两调用后的逐例中位30.017至31.916ms，p95为34.613至42.518ms；原CPU转换未改，初始化和长尾也保留。这不是硬件端到端计时，f0不等于整机零延迟。

七张比较六宫图PNG/七PDF、逐例CSV/软件计时、总览，以及两张配对训练图已本机交付。新增起走原速视频268575B与旧f0起走对照均50fps/152帧/3.04秒，304帧完整解码、SHA匹配；总览、起走细图、六帧联系表已目视。新起走在2.12秒尝试失败后保留红字失败停帧，其余六段新视频仍服务器，不重渲染。

[完整证据ZIP](../data/deliveries/Q1_F0_PROPRIO_REVIEW_EVIDENCE_20261108.zip)：2930530B/SHA `158e5a3a4e8bc6140b8bcb8e56597786baa8b57ea6c0987c3b65ca029a771542`，36成员CRC、逐文件大小SHA和18相对链接通过。原始数值包200993B/SHA `bbe54bf4895147a40dc28e5bb2aacd1a93c1fa17242032d53e48b9363add7905`，37成员/35原始文件身份核同。[终态收据](../logs/session_records/q1_f0_proprio_review_completed_20261108.json)、[交付审计](../logs/session_records/q1_f0_proprio_review_delivery_20261108.json)、[简短done/todo](README_Q1_F0_PROPRIO_RESULT_20261108.md)。

![七例六宫图总览](../data/analysis/Q1_F0_PROPRIO_REVIEW_FIGURES_20261108/overview.png)

训练114秒、闭环585秒均已SETTLED；累计4256/14400秒，r1/r2/r3=1614/1271/1371，无RESERVED，余10144秒。06:33:22原root/boot/暂停及旧五PID退出核同，381297/381298与七例worker均退，原GPU锁可用/计算空；总8212525056B/限34292629504B、OOM0、GPU1MiB25度、余盘183575195648B。未清缓存、未发训练信号。

## 实际准入

源码`aa29660433513dce1f9145263e479261a3efa4ba`与216项V2执行清单提交`a1381355ffd7274ac69ad7da4af6956a71b92d7d`均push origin Q1/独立40位远端核同。首次接收器在读取不存在的归档副本路径时退出，尚未READY或传字节/写文件；修订只引用原authorization中的计划身份，旧清单和本地包保留。V2十一成员32292B包八个精确ACK部署，1036原来源前后同，只增加清单外源码与小记录/阶段文档。

原Morph运行时六项新检查全过，本体0.238秒/进程2.332183秒；原Primebot运行时媒体依赖实核NVIDIA RTX3090 EGL，未导入torch/未渲染帧。真实zero_history选点CPU核验：八base张量/缓冲精确相同、统计与teacher/decoder和时轴元数据一致、残差末层0；可复用旧0/7。causal_history3000非零、八base张量也精确同。

1048份保护来源、原暂停/资源/原锁准入13.844985秒通过；总8194355200B/限34292629504B、GPU1MiB24度/OOM0、余盘183587844096B，累计3671秒无预留，真实新策略调用/物理0。三服务器原收据5584B已Mac二进制大小SHA核同：[部署](../logs/session_records/q1_f0_proprio_review_deployed_20261108.json)、[原运行时](../logs/session_records/q1_f0_proprio_review_contracts_20261108.json)、[准入](../logs/session_records/q1_f0_proprio_review_admitted_20261108.json)。下一发布本准入后唯一防重单派。

## 问题与范围

真实机器人历史组selected3000在保存验证状态的动作误差下降36.303%，对照组zero_history选0。此前COVERAGE01有相似的验证下降却更早摔倒，本次必须用同一参考和共同Primebot物理判断闭环是否改善。

先在CPU核对zero_history保存点：原R2 selected500的八个人体编码器张量/缓冲、归一化、decoder与teacher身份均相同，适配器末层精确为0，才复用旧七例0/7。不重复运行相同策略。非零causal_history selected3000才新增七例：三段名义前走/起走/停走，以及前走上的急停、反向、丢包、突发延迟。事件沿用原50/10/0参数，这是探索性对照，不冒充新留出事件。

新worker只改保存点重建，`action/start/preview_config`严格继承原CausalReviewSession；仍用原CPU状态转换，不混入CPU加速候选。冻结教师/decoder/归一化/动力学、参考时钟、目标、事件可见性与回退规则。GPU策略、共同Primebot CPU物理和NVIDIA EGL原速50fps媒体分阶段执行。每段记录完整worker、同步GPU目标路径和本机管道RPC时间，均不是硬件端到端延迟。

## 预算与来源

独立输出`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_F0_PROPRIO_PAIR01_REVIEW_20261108`。总1800秒/3GiB、每阶段1800秒；计原r2账本，进入前3671=1614+686+1371秒已全部SETTLED。沿用22GiB准入/24GiB退出、GPU18GiB80度、100GiB磁盘余量、private500GB、原非阻塞GPU锁及自有子树回收，不清缓存或恢复旧共享训练。

严格复用已保存的14条H4对照和7条original_f0失败，来源为COVERAGE01保存分析中的原始比较行。新七例与这21条合计28条；COVERAGE01本身七条失败不替代原f0控制。原78+6/84事件、R1旧物理和已完成训练不重做。

[计划](../script/supermimic_teleop/student_proprio_review_plan_20261108.json)、[controller](../script/supermimic_teleop/student_proprio_review.py)、[worker](../script/supermimic_teleop/student_proprio_review_worker.py)、[六项聚焦CPU检查](../script/supermimic_teleop/test_student_proprio_review.py)。训练执行的六文件不改。六检查本地0.082秒全过，CLI和编译通过；新增真实CUDA/物理/媒体0。

## Done与Todo

已完成配对训练、真实选点CPU身份门、七新闭环/14守卫、来源终门、585秒结算、只读28条保存分析，以及六宫图/两段原速媒体/独立证据包。原执行216清单和六训练来源均冻结，后续分析与交付代码只在清单外新增。已发布只读分析源码`d97fb0319f0ef806a5b725debb4bdaa6eb5a6332`，四原运行时检查通过，原1111输入SHA不变。

下一步先核实际学生访问状态上的教师纠偏与序列闭环目标。旧聚合采用50%教师动作混合，不等同于纯学生状态上的完整纠偏覆盖；这只是下一假设，尚无新采样或训练。独立方案需先登记来源组、单一变量、验证选点、资源/成本边界并发布，再有界执行。原七失败/84事件/旧234状态/64动作/512配对不重复；新留出事件仍未打开。不能据本轮负结果或文档完成结算整个低延迟探索。

最新发布与服务器镜像以独立publication收据为准；旧大包和全局索引镜像PENDING。
