# X2 联合性能优化与接口选择

2026-11-13，阶段 FOUR_VARIANTS_COMPLETED / HOST_CLOCKS_VERIFIED。原衣服和终端2/3源码、作者模型/物理以及既有R3/R4工件保持；SDK/电机发布、MC修改、新BLE连接、训练和频率写入均0。

## 用户简报

同板线程和FK优化已将实际人体更新17.743→24.805Hz，Morph RPC p95 23.685→14.866ms，超20ms拒绝157→6。尚未30Hz，最好配置仍跳155/929原始帧；180ms预览不减。最老人体参考年龄中位保守295–309→268–281ms，p95上界351→309ms，不是衣服采样到电机响应实测。

用户提供run账号后原生密码登录同SoC1和同boot，以宿主机nvpmodel和sudo jetson_clocks --show读到当前MAXN_SUPER/8核performance、每核当前及系统上限均1.984GHz、EMC已3.199GHz并FreqOverride=1。CPU当前没有可见的官方提频余量；GPU当前306MHz/最高1.173GHz为动态频率，单个空闲快照不证明推理时降频或锁频收益。本轮不改功耗、频率、固件、风扇或原服务。后续设备管理以run账号为准。

当前已部署权重同时支持human和motion。重定向后输入motion可行，仍需10槽及正确X2角度/速度/躯干参考，不能省掉FGP或自动消除未来预览。作为效果和计算量对照候选，尚未新做重定向链推理/仿真或真机。

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

## 最终四组数值

| 配置 | 新人体Hz | 输出/跳输入 | FGP p50/p95 ms | 全部ready RPC p95/p99 ms | 超20ms/ready |
|---|---:|---:|---:|---:|---:|
| control 原4线程 |17.743|533/367|42.347/64.019|23.685/29.915|157/1453|
| sleep1 Morph1线程/无空转 |21.070|633/267|36.958/47.097|18.170/25.194|37/1481|
| sleep1_fk 再批处理人体FK |22.672|681/219|36.213/48.198|18.431/22.532|39/1485|
| all_sleep1_fk 再FGP CPU1线程/无空转 |24.805|745/155|34.422/38.972|14.866/17.420|6/1492|

四组墙钟各37.056/37.082/37.052/37.041秒，FGP帧2592次/graph初始化各2次共8enqueue；合计5911次ready真实反馈策略请求。最好组1607总请求/1492ready/1486接受，115次人体未就绪和6次RPC拒绝均保留。其RPC最大30.700ms、FGP最大85.803ms；FGP超过33.333ms的469/745次，全部744个输出间隔超过33.333ms，输出间隔p95 43.892ms。不能以策略近50Hz请求率冒充新人体30Hz。

| 最老人体参考年龄 | control | all_sleep1_fk |
|---|---:|---:|
| p50下界/保守上界 ms |295.282/309.288|267.931/281.155|
| p95下界/保守上界 ms |336.029/351.111|296.268/309.269|

下界是服务reply-ready年龄，上界再加全RPC，存在计算重复的保守项；没有native绝对接收时刻、BLE无线/机械响应或真人动作相位GT。最好组原始输入释放到HumanFrame确认中位56.270/p95 71.410ms；其参考最老槽仍含180ms真实窗口跨度。四组顺序运行、非随机重复，跳帧引起后续人体历史不同，不把全部差异解释为唯一机制或质量改善。

## 一致性与资源

Mac三合同.167秒、Orin三合同.622秒：128随机骨架树深批处理与原式误差≤1e-12，原native命令不改，worker选择仅本控制器明确的两Python命令。32固定合成特征的4/1线程策略动作比较在Mac和Orin均最大差0，各64个CPU graph调用，非真人精度。四真实组共同处理前缀仅4个输出，pose/root逐位同；后续采样不同不声称完整序列等价。

四组最高温69.875度，最低可用11562205184B，均未触80度/4GiB保护。末次只读来源检查与开跑前JSON逐字节同：247原始源码、24资产、原生库和历史探针工件不变。自有PID control33968/33983/34043，sleep134956/34969/35030，sleep1_fk35687/35705/35836，all_sleep1_fk38320/38335/38388全exit0；末次pgrep为空/项目锁可非阻塞取得。原Q1训练暂停未接触。

原control_analysis第一次使用缺matplotlib的环境，仅已保存数值/CSV后退出；原件保留。随后只用既有绘图环境在control_analysis_v2重绘保存数据，未重跑模型。原V1图表ZIP保留为草稿，最终用V2及comparison_v2。无新训练视频或机器人运动视频。

## CPU提频核验

agi受限会话内nvpmodel找不到配置，sudo show被明确拒绝；这只代表该会话权限和文件视图，不能推广为整板缺配置。用户要求后改用run@10.0.1.41原生密码SSH，核同boot df9b9a2e-b465-4057-86fb-4692ca5184b5/机器人序列/SoC1，nvpmodel -q --verbose与sudo -n /usr/bin/jetson_clocks --show均exit0。run为uid1000、具sudo权限；只读宿主机回执frequency_run_host.json已保存，不写认证材料。

原生Mac→SoC0登录因旧host-key冲突被客户端拒绝，未删除/改known_hosts或关闭验证。此前通过SoC1既有可信记录的原生逐跳密码登录核过同机器人RK3588。SoC0当前承担mc_main/EtherCAT等原工作，不能按“第二个空闲Orin”迁移CUDA或承诺无代价并行。

## 可复现入口与交付

单个未存在输出根的命令：`env -u PYTHONPATH -u PYTHONHOME timeout --signal=INT --kill-after=10s 120s data/environments/fgp-preview-py310/bin/python -B -m script.x2_joint_perf.run --variant all_sleep1_fk --fixture data/analysis/X2_JOINT_LATENCY_20261113/fixture_r1/imu929.npz --output data/analysis/X2_JOINT_PERF_20261113/all_sleep1_fk`。原已成根不可再执行覆盖。来源发布567c58d51f22932f95e310a9ccdddaa036136b43及5a35f4c0fc76c76295069e7893278c679bb22e4f均push并独立40位远端核同；最终结果以publication收据为准。

Mac原始根`data/analysis/X2_JOINT_PERF_20261113`，各组JSONL/完整姿态NPZ/native状态及分析PNG/PDF/CSV均保存；最终总览`comparison_v2/six_panel.png`。本地证据包`data/deliveries/X2_JOINT_PERF_EVIDENCE_V2_20261113.zip`包含README、四组逐请求/逐帧/拒绝及资源原件、图表/数值、前后来源检查和run宿主机频率观察；生成后逐件大小SHA/ZIP CRC验收。网盘未核验，LOCAL_ONLY。

## Done / To Do

连接规则更新：用户要求后续改回Windows桥接。已原生`ssh sshuser@100.92.227.62`后在Windows内`ssh run@10.0.1.41`，交互密码登录同SoC1和同boot，会话40802；后续同步沿此路径。上文直连是实际测试历史，不作为后续默认入口，不变更网络或认证配置。详见`x2_bridge_run_recovered_20261113.json`。

用户版Done：四组实际对照、数值一致性及只读频率核查已成，约18→25Hz、超时大幅减少，电机发布0。To do：仍需稳定30Hz、连续BLE/人体质量和动作相位验证；预览等待另行解决。

Agent版Done：原三组及追加独立all1预算已结清，不重复原929输入/4组作等待；优先复用run会话81221，agi9179仅为原有受限会话。全组来源/boot/fixture/engine/native一致，原保护JSON前后同/进程退锁空；当前默认衣服和终端2/3未切换。

Agent版To do：下一独立方案先把CANO与后续CPU物理及转换/ACK解耦，保留固定30Hz递归状态、明确有限队列与原capture支持；再核同板CPU/GPU流水重叠或跨板CPU阶段（先余量/时钟域/故障边界，单writer仍原SoC1）。motion路线使用同人体源的X2重定向参考作独立仿真/性能对照，核67D特征和原10槽，不用人体SMPL直接冒充X2关节；当前未实现/派发。CPU已到官方可见上限，若另测GPU固定频率须独立短时方案、原值保存/恢复和热保护，不在本阶段擅改或刷固件。
