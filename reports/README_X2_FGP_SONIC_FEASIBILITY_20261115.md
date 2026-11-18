# X2 复用 FGP 接入 SONIC 的可行性

2026-11-15。根据用户提供的 Stage 3 操作书及 SoC1 实际源码、模型文件和 FGP HTTP 样本，接口上可行。旧程序已经有 FGP → `human_data` → `GarmentIntentAdapter` → `pico_intent` 的链路。现有 S1 GPU FGP 可保留衣服接收、标定、推理和网页，从 `/api/smpl` 分出 SONIC 人体适配器，复用原 `GarmentIntentAdapter`、tokenizer、SONIC X2 策略和监视设计。该输入分支不需要先 GMR 重定向成 X2 关节角。

现有接口提供 `pose_mat[24,3,3]`、根平移、序号、接收和输出时间戳。当前最简网页的 `joints_yup_m` 在关闭网格时为空，因此以旋转矩阵生成关节点。旧适配器用 24 个有名世界关节构造 `smpl_joints[72]`，减 pelvis 并去掉根旋转，另传全局根朝向 `human_quat[4]`（WXYZ）。不能直发矩阵或把绝对平移作为编码器输入，应优先复用旧衣服链的骨架转换并核对旧/新输入的静立、单侧抬臂、转身轴向。MorphMimic 的归一化骨架和根变换未经该对照。

原服务用机器人 base quaternion 与腕反馈生成 840 维观测、输出 64 维 token。窗口十槽步长 20ms、跨度 180ms，按 50Hz 网格取最近过去样本；原衣服链期望约 20–30Hz，现有 FGP 不需要产出 50 个不同源帧，但应检查连续性、重复帧及预览年龄。此次读到的是 `stale_hold` 中的保存样本，仅核实字段形状，未作为实时有效帧。旧 runbook 的 engage/点火命令仅作历史参考。

建议第一步用录制的同一套 SMPL 姿态对照旧转换器和新适配数据，检查关节、根朝向、时间顺序，再用模拟反馈在独立回放中验证 tokenizer 和 SONIC 策略目标。实际板端已核两份 ONNX 存在，本轮未启动服务、发布 intent/token、修改 MC 或实控。前次真机异常后继续暂停实控。
