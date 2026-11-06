# Q1 暂停后低延迟 32k GPU 实验

日期：2026-11-06。状态：目标已获授权、应用 goal 实读 active；独立入口准备完成，真实 GPU 回放尚未执行。

18:06 首次实际派发 `263048/start27248272`，18:06:56 在一致性阶段失败：GPU 独立脚本缺少 `primebot_q1` 公共模块路径，导入即退出，尚无 CUDA 推理/新物理/训练。服务器27运行时检查全过，但导入式测试未覆盖此独立启动方式；已补清空继承 PYTHONPATH 的子进程启动检查。失败根原样保留，修订代码公开后使用新输出根，非原任务自动重试。[真实失败及部署记录](../logs/session_records/q1_paused32k_gpu_attempt01_20261106.json)。峰总量15938453504B/GPU1MiB/OOM0，原冻结200源码资产和215评估源码核同；原训练未重启。

## 目标与范围

用户批准继续低延迟 SuperMimic 遥操探索并开启目标模式。原共享训练保持用户暂停状态，以保存的 32k Q1 候选研究把 200ms 真实预览缩至 80–100ms 的可行性。新训练累计不超过四 GPU 小时，是否训练取决于对照中的失败原因；不重启原训练。

本轮先在同 human 接口做 H10/H5/H4/H2/H1 和 H5/H4 的保持、阻尼预测尾部。H1 是 f+1 的 20ms，尚非 f0。每种配置固定前走、起走、停走三片和 seed0，共 27 例。短窗是未训练分布，`kp_masking=full` 时旧 `preview_mix` 没有训练短窗。

## 实现与决定

- 新增 `paused_runtime.py`：核原 boot、根、明确人工暂停收据、原五 PID 全退和停止状态；非阻塞获取原 GPU 锁，拒绝已有计算进程；核锁 PID/starttick/device/inode。
- 新增 `gpu_preview_worker.py`：严格加载原计划与 32k 保存，仅策略在 CUDA；共同物理仍使用固定 Primebot CPU 运行时。每次请求核锁所有者，RPC/退出复用原有有界协议。
- 新增 `paused_preview.py`：复用已完成双型 32k 有限性审计，先核所有参考帧 CPU 原生 H10 与 CUDA 流式 H10 的数值容差，再做新 GPU H10 和短窗物理；每例退出策略/物理后独立 NVIDIA EGL 原速渲染。GPU 后端改变后不直接把旧 CPU 轨迹冒充同后端基线。
- `preview_worker.py` 只提取计时 hook；`preview_review.py` 默认 CPU 语义保持。GPU 同步墙钟涵盖目标打包、传输与前向，禁止标成 CPU 或端到端机器人延迟。
- 预检/每阶段核发布源码 SHA；暂停前冻结输入不含这些遥操脚本。原 final CLI 完成门未修改。

资源：22GiB 准入、24GiB 总量退出、18GiB/80°C GPU、OOM0、100GiB 磁盘余量另留 3GiB 输出，单阶段 1800 秒、总 7200 秒。只回收本次自有子树，不做 cache 维护。原训练停止是用户暂停导致的 `FAILED`/`STOPPED_SIGNAL`，不改写完成。

## 验证与证据

本地相关预览 27 项运行、26 通过/1 原服务器运行时待；原完成门与资源检查 8 项通过；原阶段合同 10 项通过。三新模块编译通过。新增覆盖真实文件锁冲突/异常释放、错暂停身份、原 PID 重现、锁 inode/starttick、计时归属、27 例顺序与 H10 失败时不派物理。原服务器运行时与实际 CUDA 一致性需下发后核验，不能把本地测试当成已运行机器人策略。

- [授权与预算](../data/manifests/q1_low_latency_paused_goal_20261106.json)
- [已有多方向研究](research/2026-11-06/LOW_LATENCY_SUPERMIMIC.md)
- [原暂停凭据](../logs/session_records/q1_shared_user_pause_20261106.json)
- [已有保存点比较](README_Q1_CHECKPOINT_CONVERGENCE_20261106.md)

服务器唯一根 `/opt/data/private/ZHY`；输出预定为 `data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_GPU_PREVIEW_20261106`，执行入口 `script/supermimic_teleop/paused_preview.py`。参数为原 shared config/runtime、已核暂停收据、已完 convergence 审计根和本次公开代码清单/SHA/完整提交号。只派一次；断线先读该输出与控制器身份，不重新下发。

## 结果边界与后续

准备时新 GPU 推理、物理、训练更新均为 0，原训练与模型未改。固定三片已被原策略见过，后续同时报告完整率、足底离地/换脚、承重滑移、侧倾、上肢时序、动作跳变；根轨迹误差保留诊断。短窗配置即使三片成功，也不能当自然遥操或最终验收完成。

真实对照后继续急停、反向、丢包及恢复合同；依结果选择短窗适配或 f0+历史学生。实际新训练必须交付六宫图，候选交付原速媒体和逐片物理诊断。可以得出有证据的否定结论；不得仅凭 loss、预测误差或完成本轮文档结束整个目标。用户采纳仍为 null。

## 保存与同步

本次阶段代码与记录先按 Q1 分支显式 stage/audit/commit/push，独立 40 位远端核同后生成代码清单并部署。发布和 Mac/服务器镜像以随后真实收据为准；当前未声称服务器已执行。未新建 agent 或修改其他自动任务，未声称 q1 定时监督已存在。
