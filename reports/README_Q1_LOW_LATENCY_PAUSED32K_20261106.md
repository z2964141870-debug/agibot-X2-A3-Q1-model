# Q1 暂停后低延迟 32k GPU 实验

日期：2026-11-06。状态：目标 active；27例CUDA名义对照已完成，原共享训练保持用户暂停指令。

## Mac交付与解释

原生交互密码SCP已完整取回4863620B/44成员的结果包，大小和SHA与服务器完全相同。独立 `deliver_preview.py` 只读保存的27行数值，在Mac生成三张六宫图PNG/PDF和三张媒体联系表；九段H10/H5阻尼/H4阻尼视频逐份核原物理记录SHA、50fps/尺寸/时长及4761帧完整解码，均通过。三张图和三张联系表已目视，机器人与参考均可见，无空白媒体。服务器原始缺matplotlib失败日志和115输入未变审计继续保留，不能将后补绘图写成原分析命令成功。

本机交付根为 `data/deliveries/Q1_LOW_LATENCY_CKPT32000_COMPLETED_20261106`，机器审计为其中 `mac_delivery_audit.json`。原共享训练暂停后新增训练仍0，本次只补传、绘图和解码，没有重复27例或重审32k权重。

- [前走六宫图](../data/deliveries/Q1_LOW_LATENCY_CKPT32000_COMPLETED_20261106/mac_figures/walk_forward_loop_001__A022.png)
- [起走六宫图](../data/deliveries/Q1_LOW_LATENCY_CKPT32000_COMPLETED_20261106/mac_figures/walk_forward_start_002__A023.png)
- [停走六宫图](../data/deliveries/Q1_LOW_LATENCY_CKPT32000_COMPLETED_20261106/mac_figures/walk_forward_stop_002__A024.png)
- [80ms阻尼前走原速视频](../data/deliveries/Q1_LOW_LATENCY_CKPT32000_COMPLETED_20261106/tail_damped_cv/h4/walk_forward_loop_001__A022/review/walk_forward_loop_001__A022_s0/tracking.mp4)
- [交付与事件准入实据](../logs/session_records/q1_password_policy_delivery_event_admission_20261106.json)

保存的CUDA目标调用p95在H10三片为7.27–8.25ms、阻尼H4/H5六片为12.19–15.85ms，但各例最大调用仍1.22–1.71秒，不能省略或假定成因。这个计时不含完整观测预处理、RPC和执行；还需预热及真正实时deadline验证，相关相位也不是实测端到端延迟。低离地、参考幅度差及H4起走右滑移回退仍在，名义结果只支持继续事件对照，不构成自然遥操采纳。

保存telemetry补定位显示27例各自最大在第0帧，但去首帧后仍95次调用超过20ms、最大519.54ms（总13473调用，含首帧共122次超线）。不能将全部尖峰归为启动或据预热直接宣称实时可用。新事件入口的第一次派发在响应字段组装时报错、0事件物理，已修独立worker并准备新根；详见[事件进度](README_Q1_LOW_LATENCY_EVENTS_20261106.md)。

18:54:10逐跳交互密码桥接恢复：原boot/GPU/唯一根相同，原五PID全退、STOPPED_SIGNAL与GPU锁释放符合用户暂停；当前显存1MiB/27度、计算空、cgroup13204852736B/限34292629504B/OOM0。R2于18:47:50完成56阶段/source_integrity PASSED，27段原速媒体全部通过、26例完整；H1前走在248帧触发失败、有效247/1031步，不给整片误差或时序。R2峰总16302190592B、GPU400MiB、新训练0，不能据低显存称GPU训练在跑。

| 前走方法 | 真实预览 | 上肢RMSE rad | 左/右承重滑移 mm/s |
| --- | ---: | ---: | ---: |
| H10 masked | 200ms | .085931 | 7.49 / 7.68 |
| H5 masked | 100ms | .105290 | 20.06 / 29.73 |
| H4 masked | 80ms | .100756 | 21.18 / 25.41 |
| H5 hold | 100ms | .087695 | 9.61 / 9.66 |
| H4 hold | 80ms | .088671 | 9.45 / 9.01 |
| H5 damped_cv | 100ms | .087150 | 8.40 / 9.05 |
| H4 damped_cv | 80ms | .087862 | 8.06 / 9.12 |

保持/阻尼尾部在三片都完整，H4阻尼起走右滑移54.22仍高于H10的47.73mm/s，H5阻尼起走则为39.29；不能只报前走获益或称自然性合格。这里80/100ms是合成调度下所需真实预览，后面的120/100ms槽是预测，并未把200ms全部标成实测帧；不是硬件端到端延迟。下一项事件压力与后续适配/因果学生仍属整体目标。

原未回执分析经实核收据/日志/进程均无后，已单派279381。27例数值和上肢相关相位已保存，绘图因原Primebot环境缺matplotlib而失败，保留日志；115源文件SHA再次核同，生成独立input_integrity_audit而非伪造原analyze成功。9段H10/H4阻尼/H5阻尼媒体、数值及记录共44成员4863620B交付包在服务器，SHA `9b43b58dccf271ed918d007a2c8ce2dec3c8f8f1345a5bd93c26d91e6dcdbaba`，Mac绘图和目视待取。连接规则及新事件准备见[最新实据](../logs/session_records/q1_password_bridge_and_preview_completed_20261106.json)。下方6例/UNKNOWN为恢复前历史。

最新可恢复进度：[实据](../logs/session_records/q1_paused32k_progress_and_bridge_20261106.json)。H10/H5三片共6例物理全完整，5段原速媒体当时完成，最后阶段H5停走媒体；100ms前走上肢RMSE从.085931到.105290rad、左右承重滑移7.49/7.68到20.06/29.73mm/s，说明直接mask缩窗有损失，尚不代表保持/阻尼或其他路线的结论。后续新事件合同已本地6检查过，物理未执行，详见[事件记录](README_Q1_LOW_LATENCY_EVENTS_20261106.md)。

随后63485 Timeout255；本独立故障的一次有界恢复84492到Windows认证后Broken pipe255，未Windows/Linux shell，不重复认证或改入口。R2控制器263415/start27279136是否已完成、实时GPU/cgroup/OOM均UNKNOWN，不据失联称任务失败。保存工件分析命令已发送但未收到启动回执，必须先核 `logs/session_records/q1_paused32k_analysis_dispatch_20261106.json`、`q1_paused32k_analysis_status_20261106.json` 和对应日志，再决定是否需要启动；禁止盲目重发27例。新记录服务器镜像待恢复，旧5a07231七份617197B镜像PASS已实收。

18:11:13 修订版 `184e59bfe52bddad4d6d3e87dcddd26991005ee6` 已经独立40位远端核同、两代码/核心记录/新清单六文件同步后，用新根 `Q1_LOW_LATENCY_CKPT32000_GPU_PREVIEW_R2_20261106` 一次派发 `263415/start27279136`。服务器独立入口7检查全过/0跳过；215源码再次核同，原200冻结源码资产不变。随后三片H10全1584帧CPU/CUDA动作一致性通过，最大绝对差分别2.536922693e-6、1.102685928e-6、1.125037670e-6，低于预设2e-5绝对/相对容差；不是逐bit完全一致。首前走物理全长完整、原速NVIDIA EGL媒体已完成，其余27例对照仍在推进，尚不判断短窗质量。新训练更新仍0。

新增 `script/q1_checkpoint_review/preview_summary.py` 仅分析保存工件，在本次campaign源码清单之外，避免改变在跑代码。待27例结束后生成三片各六宫图/CSV、完整片上肢相关相位诊断、动作变化及同步GPU目标调用耗时；失败例不计算整片误差和相位。相关相位是离线描述、GPU目标调用不含观测预处理/RPC/执行，不是实测端到端延迟。

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
