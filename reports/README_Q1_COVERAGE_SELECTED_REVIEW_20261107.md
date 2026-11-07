# Q1 覆盖学生已选500闭环评估

记录日期：2026-11-07。状态：服务器部署、四项原运行时检查及真实准入已通过；七例闭环尚未派发。最新自主固定入口两次尝试均在第一跳结束、未进入Windows；原Linux实时未核验，不据此判断服务器停止。

## 用户简报

done：扩大覆盖训练真实完成，验证选500。新控制器已在23:21:48部署，770保护来源不变；原服务器四检查1.534秒通过，23:23:03来源及资源准入通过。CPU审计证六个网络张量改变、统计和冻结元数据保持。三原始回执尚未传回Mac，当前记录来自实际收到的服务器输出。

to do：Agent自行固定入口恢复，取回既有三原始回执，不重做部署或检查；核实时身份/暂停/资源/锁和唯一输出、派发、日志、进程，再单派七新闭环。完成后仅读保存结果，复用十四H4及七旧f0形成28例比较，交付七六宫图和原速视频。验证改善尚不能当闭环或自然遥操结论。

## 方法与边界

仅使用COVERAGE01的selected500，2475597B/SHA`108e649d1466d755a66da84a61a5c6f6a552b788f945deefcbf515ff327874b3`，不使用last挑test。来源是已完成3000更新及30片9585步采集，父训练source PASSED、worker回收及575秒SETTLED均须通过。重新从实际validation点核最优且两个固定锚点不退，再CPU核八编码器张量有限、统计与冻结策略元数据同，并证实际网络权重已改变。

新物理仅前走、起走、停走及前走四合成事件stop/reverse/dropout/late_burst，共七。原R1的十四H4 teacher/student对照严格复用，旧R2 f0七例保存失败用于后续比较，不重跑原基线、采集或训练。共同Primebot CPU物理、CUDA已选策略、NVIDIA EGL原速媒体，旧时间可见性及120→50Hz插值右支持门保持。

诊断保留有效完整率、脚底离地/换脚、承重滑移、侧倾、上肢全程与事件窗/时序、动作突变、fallback和恢复。软件计时同时保留目标区间、完整worker和本机管道RPC及去前两调用后长尾；不相加嵌套计时，不称人体网络或执行端端到端延迟。失败只报真实前缀，不出全片误差或失败组均值。

## 执行和资源

入口`script/supermimic_teleop/student_coverage_review.py`；固定计划`student_coverage_review_plan_20261107.json`。只新增控制器/计划/检查，原193执行与200冻结来源保持。

唯一输出`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_COVERAGE01_REVIEW_20261107`。原非阻塞GPU锁、22GiB准入/24GiB退出、显存18GiB/80度、磁盘100GiB/private500GB不变；总1800秒/每阶段原1800秒/3GiB，原自有子树回收。没有新optimizer更新或缓存清理，不恢复原共享训练。首轮训练账本累计2181/14400秒，不重置或隐藏评估成本；本次评估耗时另报。

四新CPU检查0.758秒全部通过，随机张量只用于检查编码器身份，不是真实模型推理。首次本地测试命令相对解释器路径多一层而未执行，已用真实绝对路径运行。原服务器部署、检查及准入已由下节覆盖；新物理仍待。未知派发不能重发，第一跳固定用户source/ssh入口，由Agent自行尝试，不让用户代测。原goal清除、final门及采纳null保持。

## 真实部署及准入

源码`e8bfecc7ba2a70cf875f15168a098dbd315a7280`、195清单`f142429456c6d8d7245ee12b220096a349c750e2`均已显式push origin Q1且独立核远端完整SHA。清单SHA`c724c7f07b2d5d12ed0154422b8b4899db75bb59eac930d0a05437c03051567b`；四成员16943B部署包SHA`ca24ff8263b9a1f9fc474a7897a9146605cc2831d7c370efd438b532b423116e`。

2668沿固定入口进入原Linux。23:21:48已收到四目标SHA同、770保护来源前后同的部署回执；四原运行时CPU检查全过，测试1.534秒/进程2.165930秒。23:23:03实际准入3.221712秒通过：195执行、200冻结、2200保护记录及十四H4对照，已选500有限CPU身份审计通过。内存8447348736/34292629504B、GPU1MiB/26度/OOM0，原共享暂停，累计账本2181秒且无RESERVED。

原文件`q1_student_coverage_review_deployed_20261107.json`、`q1_student_coverage_review_native_checks_20261107.json`和`q1_student_coverage_review_admitted_20261107.json`尚未Mac字节核同。取回时2668明确Timeout255；服务器可能已生成`data/deliveries/Q1_STUDENT_COVERAGE_REVIEW_ADMISSION_RAW_20261107.tgz`，存在性及SHA待核。上述实测值按已收输出登记，不冒充原文件同步。恢复只补取回及实时准入，不重复部署、检查或旧CPU身份审计；执行入口开跑时仍会重核全部门。

没有提交review的Popen或派发脚本，当前七新闭环为0。随后自主固定入口56539及40576第一跳连接超时；本轮一次有界15017第一跳Connection reset，均未Windows shell、未尝试Linux。旧85082及2668已经退出，不再探测。这些只说明各会话结束，不证明Linux不可达；不会要求用户先验证连接。

## 保存结果分析准备

新增独立`script/q1_checkpoint_review/student_coverage_summary.py`和`deliver_student_coverage_review.py`，不改195执行或200冻结来源。原R2分析根由原聚合准入收据核为`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_REVIEW_ANALYSIS_20261107`。

仅在七例及十四阶段实际COMPLETED/source PASSED、worker回收、selected500身份和全部来源SHA通过后，复用旧21形成28例保存比较。失败保留真实前缀，不产生全片误差或三片均值；原f0失败不重跑。三层软件计时、事件窗、脚底/换脚/承重滑移/侧倾、上肢时序和恢复诊断均保留。绘图要求实际28例分析及审计成功，才生成七PNG/七PDF、总览和软件计时CSV，当前没有本候选闭环图或视频。

四新合同检查0.002秒通过，含错误/重复目标、缺失媒体阶段、未完成源门、末轮替换500、失败误差与组均值拒绝。这是分析合同，不是真实闭环。分析与绘图源码待本阶段发布，服务器部署、真实分析和新图均0。后续独立分析根`Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_COVERAGE01_REVIEW_ANALYSIS_20261107`必须不存在才执行，不重做采集训练。

## 证据与下一步

父阶段[实际结果和训练六宫图](README_Q1_STUDENT_COVERAGE_TRAINING_20261107.md#真实完成结果)，原始数值包80172B/SHA`ced82c96d72b85a3d58b24306dee79e9b1855769328a4f08f06001babcdeb11c`已Mac核同。新闭环实际派发与结果以后续收据为准。源代码和清单先显式push origin Q1、独立40位远端核同，再部署；新说明服务器镜像尚待。
