# FGP-S 与 EgoLocate 权重核查

记录日期：2026-11-13。状态：本地资产核查完成，FGP-S 已训练权重未找到。

**后续范围更正**：用户要求读取作者仓库后，实查发现此前遗漏的 fgp_realtime 分支（2026-11-09 更新）已说明 CPU ONNX 及 onnx_s1/onnx_cano_s_ep120_fast 两套外置资源。本报告“未找到”限于此前提供的 ZIP/main，不推广为作者无轻量部署。当前仍未拿到新资源包；旧 0.78 秒不是新版本测速。详见 [实时分支核查](README_X2_CLOTHO_REALTIME_BRANCH_REVIEW_20261113.md)。

用户后续明确“EgoLocate 用哪个权重就跟着用”。当前实施选择已固定为包内 `FgpMocapAdapter` 的默认：去噪开启、30 帧历史、`U100hL7.07h_residual.pth`＋`Full_GR_OV_10.forward_frame_fast0`；用户最新 CLOTHO 中该类默认解析到 `checkpoints/globalpose/weights.pt`。不再以找到 S 权重作为继续准备的前置条件。预览启动校验该两份权重及 SMPL 的精确 SHA，并在 API/doctor 输出 `model_profile`。外部 `ble_web_pose` 缺失，不能宣称整个旧实现逐字等同；模型选择及现有已提供权重的身份已固定。

用户进一步明确“权重使用 EgoLocate 的，代码使用新版”，并引用作者对 `online_fgp+LIP` 姿态 bug 修复及新增 global translation 的说明。本次实查作者 `humanplus-ai/CLOTHO-mocap` main 为 `bed3424b86caa5849c75b5046252331f73cdce20`（提交信息 `Update realtime translation`）；本地与该提交的 `neural_backbone.py`、`online_fgp+LIP.py`、`LIP_model.py`、`full_GR_OV.py`、`stage_tran/online_guarded.py` 五份逐字相同。

新版 LIP 的两层 `EasyLSTM` 已显式使用 `act_func="tanh"`；激活层不在 state_dict 中，不能仅凭权重加载成功判断此修复生效。新版 LIP 默认 `stage_tran/OnlineGuarded`，需要单独 `root_velocity.pth` 与 `foot_geometry.pth` 两权重；当前提供的 checkpoints ZIP 不含这两份，旧本地 runner 的 LIP 简单 foot-lock 不能当作已接入新 StageTran。此限制不改变当前按 EgoLocate 默认选择的 **GlobalPose** 路径：它从新版 `forward_frame_fast0` 返回原生三维米制平移，不置零，不使用 Unity 的 Z×0.7 或 Y＋1m 显示变换，也不把 LIP 的平移模块混接到 GlobalPose 上。

## 结论

用户要求在 `/Users/yu/Human+/key_project/EgoLocate.zip` 中查找可在 CPU 上运行的 FGP-S 权重。本次直接读取 ZIP 目录、相关源码和权重张量，没有启动衣服采集、标定、训练或机器人控制。

1. 包内可见的神经模型/数据 `.pt/.pth/.ckpt/.onnx/.safetensors` 成员只有 `EgoLocate/mocap/assets/weights.pt`（23,224,559 B）和 `EgoLocate/example_data/imu.pt`（1,615,097 B；示例 IMU 数据）。没有包内 `FGP/` 目录，也没有文件名包含 `FGP-S` 或 `FGP_S` 的成员。另有 SMPL 体模及 URDF 资产，不是 FGP-S 网络。
2. `mocap/assets/weights.pt` 是 **PIP/FastPIP 权重**。包内 `mocap/README.md` 明示使用 PIP 的同一份权重；`mocap/net.py` 定义 `FastPIP` 并从该路径加载。安全张量读取（`weights_only=True`）得到 72 个有限张量、5,800,184 个元素，首层为 `[256,72]`，与六 IMU 的 PIP 路径一致。SHA-256：`94511b459a185f013afb52fd53fb8b17c101b83a97a8a1beb390067790303500`。
3. `fgp_mocap_adapter.py` 的 `FGP_ROOT = ROOT.parent / "FGP"` 指向 **EgoLocate 目录之外的同级 FGP 工程**；适配器从 `ble_web_pose` 加载 `Full_GR_OV_10`，可选规范化权重路径为 `checkpoints/canonicalization/U100hL7.07h_residual.pth`。这些外部 FGP 资产没有随当前 EgoLocate ZIP 提供。不能把包内的 PIP 权重当作这个 FGP 适配器的权重。
4. 用户提供的最新 CLOTHO 源码确实定义 **S 规格**：4 层规范化块、encoder 维度 256、3 层 encoder；B 规格为 6 层、384 维、4 层。以源码结构在 meta 设备上核对，S 为 10,144,521 参数，B 为 32,051,721 参数。这里只核对结构，未用随机 S 网络冒充可用模型或测其动作效果。
5. 现有 `U100hL7.07h_residual.pth` 的 `model` 为 180 张量，与 B 的键和形状全部匹配；对 S 有 48 个多余键、131 个形状不符。它是 B 规格，不能改配置为 S 后使用。384,845,450 B 文件还包含 optimizer 等训练状态，不等于纯模型参数大小。

范围内没有找到 FGP-S 的已训练权重，不能据此否认其他位置存在该版本，也不能确认用户所称 FGP-S 一定就是当前源码的 S 配置。接下来需要对照同级 FGP 工程或确切的 S checkpoint，再做同输入、同设备的 CPU 速度与输出质量比较。

## 更正此前取证

此前 [最新 FGP 预测试报告](README_X2_LATEST_FGP_PRETEST_20261113.md) 和原始 `x2_latest_fgp_pretest_20261113.json` 把 23.2 MB 文件归为“旧版单阶段 FGP 权重”，该身份判断错误，现由本次直接源码/张量核查更正。原始 JSON、旧 ZIP 和已保存轨迹不重写；历史人体流里根平移为零只是流的事实，不能由此认定其采用了这份 PIP 权重、单阶段 FGP 或特定版本。两条保存流的角度差也不构成两个已知版本之间的准确率对照。

## 证据与复核

结构化收据：[x2_fgp_s_asset_audit_20261113.json](../logs/session_records/x2_fgp_s_asset_audit_20261113.json)。

只读检查的包内来源为 `mocap/README.md`、`mocap/net.py`、`mocap/config.py`、`fgp_mocap_adapter.py`；95 个相关 `.py/.md/.txt` 文件中无显式 `FGP-S` 文字。未读取 `.git`、凭据、SLAM 构建二进制或录制内容来寻找认证材料。源码尺寸表来自最新 CLOTHO `train_cano.py:init_cano_model`，模型实现为 `neural_backbone.IMUCanonicalization`。

```bash
unzip -Z1 /Users/yu/Human+/key_project/EgoLocate.zip | rg '\.(pt|pth|ckpt|onnx|safetensors)$'
unzip -p /Users/yu/Human+/key_project/EgoLocate.zip EgoLocate/mocap/README.md
unzip -p /Users/yu/Human+/key_project/EgoLocate.zip EgoLocate/mocap/net.py
```

PyTorch 检查使用既有 `data/environments/clotho-fgp-20261112/bin/python`，全部 `torch.load` 显式 `weights_only=True`，PIP 权重检查有限性，CLOTHO 在 meta 上仅检查张量键/形状，不声称重新验证其有限性或 CPU 性能。

## 前一阶段准备保存点

`script/x2_fgp_preview/` 为此前用户要求的自启动衣服网页准备，当前保持开发状态。已有 BLE 协议解码、静态标定、SMPL 网页与独立安装/运行脚本；新增错误权重拒绝检查后，本机 10 项测试 0.206 秒、设备 10 项 0.278 秒均通过。桌面 1366×900 与移动 390×844 的网页非空/运动、暂停、定位、骨架开关、未授权请求拒绝及无水平溢出通过，截图已目视。历史回放按记录原始时间推进，约 1 秒墙钟对应 0.92–1.00 秒源时间。浏览器验证目录：`data/analysis/X2_FGP_PREVIEW_20261113/web_checks_final/`。

此前在 SoC1 的 32 输出录制输入测试：当前 **B＋GlobalPose** CPU 中位 780.537974 ms、p95 854.949824 ms；它与 Mac 保存输出匹配（旋转最大绝对差 1.452863e-6、平移 7.264316e-7），但不适合 30 Hz 动态实时重建。该数值不代表 FGP-S。在线采样若跳过大量帧会破坏固定时间步假设，因此尚不交付为实时遥操前端。SMPL API 的 150 ms 新鲜度检查不会把这类旧输出标成实时有效。8 件设备原始 JSON/日志已通过原生两跳 SCP 取回，包 4,243 B / SHA `c9b9e6215183fae0854de718782c2ca7f3f376399876fc790f2b6f7669e39fd8`，位于 `data/analysis/X2_FGP_PREVIEW_20261113/preview_evidence_20261113.tgz`，已核 SHA 并解包到其同级 `device_evidence/`。

此前 native 180 秒扫描发现两件衣服，但新的 Bleak 扫描没有收到通知；差异尚未定位，真实 BLE 接收和真人 T-pose/SMPL 均未通过。用户将自行启动采集/标定，本次不再启动。Mac 本地历史预览服务 `http://127.0.0.1:8768` 保留；设备尚未启动网页采集服务。最新 20 成员源码/manifest 小覆盖包（39,555 B / SHA `1480da5478628606321142650e3bf60a6e753d7a6dab4fdc7c8c978fa95ebd3c`）已两跳 SCP、SHA 核同、在本项目非阻塞锁保护下解包，214 项完整资产/源码检查和真实模型加载通过。旧 manifest 另存、旧安装包不覆盖。

本次复用 92286 时第二跳回报连接结束并回 Windows cmd；随后同句柄原生 `ssh agi@10.0.1.41` 交互认证已成功。14:43:24 实核 SoC1 与原项目存在，boot 为 `39cb8e5a-9c22-451e-a3ba-23c98bebccd2`；当前 92286 在 SoC1 shell，80856 在 Windows cmd。该会话退出不表示整个桥接不可用，未改转发或认证方式。

## 15:06 权重未替换与分段耗时复核

用户追问“小权重为何还是 0.78 秒”后，重新核实了板端实际加载项：**没有替换成小权重**，仍是 U100h B 去噪＋GlobalPose。此前 32 输出的 780.54 ms 是这套组合的历史测试，不能作为 FGP-S 或更小权重的测速。此前将“按 EgoLocate 适配器默认配置准备”表述得像已换成小权重，说明不准确，现明确更正。EgoLocate 外部 FGP 工程/权重缺失，当前使用用户提供的 CLOTHO 权重解析该配置，不能声称与缺失的旧权重逐字等同。

同一 SoC1、同一录制前缀、同一权重进行了有界 CPU 分段诊断，模型只加载一次，各设置重置状态，顺序为 4 线程后 1 线程。每组 8 输出，下面为去掉前两次后余下 6 次的各项中位数；分项中位数不要求相加等于总中位数。

| CPU 线程 | U100h 去噪 | GlobalPose | 整段 |
| --- | ---: | ---: | ---: |
| 4 | 818.73 ms | 70.46 ms | 888.29 ms |
| 1 | 653.55 ms | 83.47 ms | 741.78 ms |

这是小样本顺序诊断，不是完整线程优选或小模型对照，不据此改动当前 4 线程默认。两组相对保存参考的姿态最大差均为 1.206994e-6、根平移 3.576279e-7，低于 2e-5 门。当前测到的主要耗时在 U100h；仅替换下游姿态权重不会移除该阶段。0.78 秒不是不变的模型常数，不将该差异归因于已证实的降频、BLAS 缺失或某个单一原因。

实际权重为 `checkpoints/canonicalization/U100hL7.07h_residual.pth`（384,845,450 B，含训练状态）和 `checkpoints/globalpose/weights.pt`（134,051,421 B），SHA 与此前加载记录一致。SoC1 实读 8 核亲和性、performance governor、当前/最大 1,984,000 kHz；Torch 2.5.1 为 CPU 构建、OpenBLAS/OpenMP 可用，尚未尝试 GPU 后端优化。文件体积不等于模型参数量或推理耗时。

原始记录位于 Mac `data/analysis/X2_FGP_PREVIEW_20261113/cpu_split_recheck_20261113.json`，6999 B，SHA `b63a5815193b0158f055a4eb8d15af381caf802742b234db7753c1918cced599`；已从 SoC1 原生两跳取回并核同。板端原件为 `data/analysis/X2_FGP_BLE_LIVE_20261113/cpu_split_recheck_20261113.json`。结构化更正见 [分段复核收据](../logs/session_records/x2_fgp_cpu_split_recheck_20261113.json)。新诊断无 BLE/标定/机器人反馈、电机、GPU 或训练；尚无已实际加载测试的小权重 FGP 结果。

## 用户自启动命令

在 SoC1 终端执行以下命令可先检查衣服数据，不启动 FGP 重建：

```bash
cd /home/agi/x2_morph_human_20261113
bash script/x2_fgp_preview/run_orin.sh transport
```

需要慢速 SMPL 诊断时，先 Ctrl-C 退出前一个，再运行 `bash script/x2_fgp_preview/run_orin.sh smpl`。CPU 实测限制如上，不称实时完成。浏览器须能访问机器人 LAN，再打开 `http://10.0.1.41:8768`；Mac 仅通过原生 SSH 连接并不自动获得这个 HTTP 路由，可在 Windows 浏览器或 Mac 直连机器人 LAN 后查看。扫描上限 180 秒，每件连接最多两次、每次 60 秒。用户点击标定才开始三秒静态收集；至少 60 对新鲜样本通过稳定性门才重建，静止不等于 T-pose 几何已正确。`/api/state` 和 `/api/smpl` 可供后续只读检查。

## 保存、下一步

资产与浏览器输出保持 `LOCAL_ONLY`，不将用户模型/第三方源码进入 Git。阶段记录与自有准备代码按 Q1 分支发布；具体发布状态见 [publication](../logs/session_records/x2_fgp_s_audit_publication_20261113.json)，设备镜像未完成则如实待补。

当前源码/manifest 设备覆盖与检查已完成；由用户启动衣服接收、标定，再只读检查真实数据。板端慢 CPU 输出仅用于诊断，实时后端性能需另行解决；不继续因 S 缺件停住准备，也不换成 PIP。`hardware_commands=0`、`feedback_subscriptions=0`、`publish_allowed=False`；Q1 训练保持暂停。
