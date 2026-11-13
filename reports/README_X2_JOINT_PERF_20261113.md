# X2 联合性能优化与接口选择

2026-11-13，阶段 PREPARED。原衣服和终端2/3源码、作者模型/物理以及既有R3/R4工件保持；独立实验不发布电机命令，不改MC或训练。

## 受控优化

同一929帧夹具/30Hz释放/最新帧策略/真实只读ROS/native observe，先做三组各最多100秒控制器、120秒外层及900次FGP调用的独立比较。第一组保留原4线程Morph默认spinning，第二组仅改Morph为1线程且禁止ORT空转，第三组再将SMPL无状态FK按树深批处理。FGP TensorRT/CPU4线程/GP30Hz物理和180ms真实参考跨度均不变。进程内包装只选择本协调器自己启动的两个明确worker，原native命令和20ms接受边界原样复用。

三组是当前同输入有界对照，不用以前32输出探针与长录制直接作因果对照。保留所有输出率、跳帧、全部RPC超时和参考年龄长尾；性能不代表人体质量。没有未来帧预测、训练或时钟缩放。

资源保持原非阻塞项目锁、至少4GiB可用内存、80度及10GiB磁盘停止门。实验根`data/analysis/X2_JOINT_PERF_20261113/{control,sleep1,sleep1_fk}`，不覆盖历史或实时入口。

## 跨板现状

本轮从存活SoC1原生密码SSH进入SoC0，核同机器人序列。SoC0实际RK3588，非第二块Orin；8核A55/A76、16GB，无ONNXRuntime/torch/zmq，已有mc_main/EtherCAT等工作，负载约10。板间四次ICMP RTT .256–.764ms，不能当应用传输单向延迟。现阶段优先减少同板线程竞争，跨板迁移须先测CPU余量且不得影响原控制服务；不迁移电机writer或修改官方服务。Mac直达SoC0因旧IP host-key冲突被正确拒绝，未改known_hosts；随后沿SoC1已信任原生入口登录成功。

## Human 与 Motion

实际已部署ONNX包含`action_motion`和`action_human`两输出；不是只有human。X2配置为`policy_goal=motion`、`human_interface=true`、`goal_anchor=reference`。原接口读取930机器人历史，motion目标670，human目标2160＋mask/heading。运行时裁剪到选定分支。

可选链路为衣服IMU→FGP SMPL→X2重定向→机器人参考状态序列→motion分支→X2后端。SMPL不是X2机器人关节表示，不能把人体72旋转值直接塞进motion目标。原源码motion和human均使用10槽；motion切换不自动减少预览等待，还增加重定向和参考速度/坐标构建耗时。当前只核接口和制定候选，不切换真机默认、不声称重定向链已测试。

Done：实核双接口及两板身份/负载，独立性能适配器和有界比较准备。

To do：数值合同、板端同源资源准入后三组单派比较；若仍不足，再拆分骨架转换/通信或有界跨板候选。必须交付完整p95/p99和六宫图，不能只报平均值或用跳帧达标。

## 三组后追加的独立线程候选

三组已完成，各37.1秒、所有自有子进程退出0。control人体17.743Hz，sleep1为21.070Hz，sleep1_fk为22.672Hz；FK中位4.479→1.715ms，FGP仍36.213ms且未30Hz。来源和原限额不扩展，新增独立`all_sleep1_fk`根，仅在第三组基础上将FGP进程的ORT与torch也设1线程/禁止ORT空转，不改模型、物理步长、GPU引擎或原代码。相同929输入/最多900GPU帧/100秒＋120秒外层门，原三组不重做。32组有限合成特征在Mac及Orin各64个CPU graph调用、线程4/1动作最大差0；这不是人体质量认证。
