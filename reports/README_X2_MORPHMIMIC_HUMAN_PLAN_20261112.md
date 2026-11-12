# X2 MorphMimic human 遥操接入方案

2026-11-12，北京时间。用户明确本轮只构思方案，`x2`分支仅作为既有RL部署参考，不要求真机测试。本轮没有模型推理、仿真、编译、训练或机器人命令，没有恢复应用goal或原共享训练。

## 用户版结论

主线固定为：**衣服/视觉SMPL → MorphMimic训练一致的标准人体骨架 → human策略 → X2关节目标 → 现有X2执行后端**。不再以在线GMR重定向为必要环节。

复用已有衣服输入和X2底层，新增人体消息、MorphMimic观测与策略适配器；保留SONIC为独立可选策略及对照。先保持原窗口跑通仿真，再研究短预览，避免把接入错误与降延迟退步混在一起。下一实际阶段应是离线接口与仿真任务，尚未执行。

## 已核对的参考

用户指定仓库的[x2分支](https://github.com/z2964141870-debug/agibot-X2-A3-Q1-model/tree/x2)，本次独立git远端核定提交`79c4a1e7c019fe57130a2f530fcec78875eb7e4f`。读取README、根/历史嵌套工作约定、现场卡、部署说明，以及下列固定提交文件；历史记录不是当前机器人状态证明。

- `script/smartwear_runtime/SmpleXConverter.py`：`convert_axis_angle_to_human_data_fast`可以返回24个SMPL关节的位置与世界四元数wxyz。其输出注释原用于GMR，不据函数名直接认定与MorphMimic的rest frame、轴向和骨长一致。
- `script/smartwear_runtime/garment_udp_v2_reconnect_safe.py`：现有路径仍构造G1 GMR并调用`retarget.retarget(human_data)`。新human模式应在该调用前分出，避免为不消费的G1参考继续付出IK开销。保留旧路径，不能直接全局删掉GMR。
- `deploy_ws/agi_x2_deploy_onnx_ref/src/smpl_intent_source.cpp`：当前读入`smpl_joints`72个位置值与`human_quat`一个四元数；环形缓冲使用接收端稳态时钟。该消息不足以提供MorphMimic需要的24关节旋转；接收时间也不能代替原始采集时间。
- `deploy_ws/agi_x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp`：当前部署绑定SONIC的SMPL观测、tokenizer和本体观测缓冲；有独立writer、状态机与`--no-publish`入口。源码默认writer500Hz且可配置，历史嵌套约定曾固定250Hz；未来以实际启动清单为准，不凭默认值改发布频率。

MorphMimic固定候选为上轮核定`770394b4a0cad9effbf807a6c03435d1699733d0`及现有v0.1统一发布X2权重。具体执行前须重新锁定其文件SHA、导出版本、机器人配置和资产；不拿Q1/G1自训权重冒充X2权重。

## 模块边界

```text
衣服LFP/视觉估计：SMPL局部旋转、根平移、采集时间戳
       ↓
SmplHumanAdapter（新增）
  24槽语义、局部→世界旋转、坐标/rest标定、统一骨长FK
       ↓
HumanFrame（新增消息）→ HumanWindowBuffer（原时序基线）
       ↓                                      ↑
MorphHumanPolicy（新增，ONNX Runtime） ← X2测量状态/历史
       ↓ canonical action29
X2TargetAdapter（使用模型导出的translator/default/action_scale）
       ↓ native target31
既有部署状态机/单writer/AimDK反馈与命令后端 → X2
```

### 可复用部分

衣服连接与推理、关节反馈接收、按名排序、时间和日志基础设施、机载运行框架、现有MC交接状态机及单writer所有权。复用的是机制，不是把SONIC的观测、动作尺度和增益当成MorphMimic参数。

### 必须新增的部分

1. **SmplHumanAdapter**：优先使用现有`human_data`中的各关节世界旋转，或从局部旋转重建；按MorphMimic的统一骨架求位置，保持root高度/地面与朝向语义。现有PICO→human的坐标变换不能不经校验直接施加给衣服数据，防止重复旋转。
2. **HumanFrame v1**：`sequence`、原始采集时间及其clock_id、接收时间、位置`[24,3]`、世界四元数`[24,4]`、有效性、标定id、骨架与坐标约定id。网络传人体数据，不传SONIC token，也不把人体关节当X2电机目标。现有消息可保留；新版本用独立类型，避免旧接收器误读。
3. **HumanWindowBuffer**：原版先保留十槽及其真实延迟；当前部署十槽含当前帧时跨度180ms，旧Q1离线下一至十帧为200ms，统一按槽时间戳记录。未知跨机时钟偏移不伪造单向网络延迟。未来80ms候选是新模式，预测槽必须标记，不能缩放时间轴冒充更低等待。
4. **MorphHumanPolicy**：严格复现上游human特征、mask、heading及normalizer，结合93维单帧机器人观测与模型要求的历史。SONIC的840维人体/990维本体特征不能直接复用；保留独立策略类型，而非替换SONIC模型文件名。
5. **X2TargetAdapter**：canonical29到native31、腕关节交换/符号、肘offset、default和action scale由导出的X2配置驱动，头关节保持模型定义的默认目标。确保缩放只应用一次，并明确历史中的last_action口径与上游一致。

IMU需单独核对：MorphMimic X2要求pelvis观测，而既有后端常读torso。可复用已验证的骨盆重建方法，但必须同时核姿态和角速度，包含腰部关节速度及传感器外参。不能只改四元数或将胸部gyro直接视作骨盆gyro。

为MorphMimic建立独立模型/资产/PD配方；现有SONIC配置保持。若需要不同执行器增益或动作过滤，必须随模型配方在仿真中核验，不能默认继承SONIC的经验补偿。任何跨策略切换涉及各自历史重置和输出过渡，首版仅启动前选定策略，不做运行中任意热切换。

## 后续阶段设计（本轮不执行）

| 阶段 | 要回答的问题 | 可交付证据 |
|---|---|---|
| 1 离线输入与动作契约 | 衣服的关节/旋转是否对应训练人体；ONNX和原策略是否同动作；X2映射是否正确 | 人体骨架可视化、逐字段/动作数值、源SHA与时间戳记录 |
| 2 同人体流驱动X2仿真 | human接口能否完整跟随；前走/起停/上肢转体及丢帧表现如何 | 原速人/目标/实际对照视频、完整率、离地换脚/承重滑移/侧倾、上肢时序、周期p50/p95/p99 |
| 3 影子推理 | 接真机测量时能否实时构建观测与目标而不发布控制 | 明确no-publish的完整链路日志、实际字段与动作一致性；也须作为新任务执行 |
| 4 未来真机 | 当前型号、反馈/执行器和状态机是否满足部署 | 由后续明确任务和现场条件确定；本轮未授权或执行真机测试 |

前两阶段可用保存SMPL录制，不以拥有PICO头盔为前提。实时衣服流与录制回放用同一适配器，记录输入丢帧、首目标年龄和重连标定变化。先让现成X2策略发挥作用，只有实际发现短预览或执行器失配才单独计划训练，不重开原G1+Q1共享训练。

## 本轮实际操作与边界

用户最初说“先测试”后，Agent启动固定第一跳并进入Windows，第二跳原25362在认证前返回Connection refused，没有进入Linux；用户进一步限定只构思后，未继续重连，退出Windows。无机器人连接/命令、无模型调用或仿真训练。一个仅为读取x2分支的git fetch较慢，已终止其自有下载进程；未切换/清理任何工作树。仓库原有未测试`dependency_preview_worker.py`草稿保持。

原应用goal仍用户暂停；本方案不是新目标模式，不自动派发测试，也不改自动任务。原训练和5036秒历史台账保持。源码核查与文件SHA见[收据](../logs/session_records/x2_morphmimic_human_plan_20261112.json)。服务器镜像未尝试；Git/Mac发布以publication为准。
