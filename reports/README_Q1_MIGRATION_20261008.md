# Q1 资源续传与仿真入口恢复

记录日期：2026-10-08。阶段编号：Q1-P0T。状态：当前开发所需模型与 SDK 核心资料迁移、核验及 MuJoCo 基础运行检查完成；完整 SDK 与工具链 ZIP 仍未全部迁移。

## 目标与范围

用户要求解决迁移阻塞，继续在 hp3090 的 `/media/yu/FAFF-E977/YuanQi_Q1` 开发。本阶段交付完整机器人模型及轻量 SDK 核心资料，并实际验证模型加载；完整 SDK 和板端编译资料保留为后续迁移项。Mac 不新增大型资产副本；第三方资产保存在服务器被忽略的 `data/references/q1/`。

## 完成内容

- 已重查 Q1 分支工作区干净，恢复入口以服务器最新文件为基线。
- 当前磁盘可用约 591 GiB；已有环境 `x2-sonic-isaaclab` 实际导入 MuJoCo 成功，版本 3.3.7。
- 模型 ZIP 应为 11,065,174 字节；初查服务器为 4,702,208 字节。该文件 SHA-256 与 Mac 原件的前 4,702,208 字节一致，证明此断点的前缀可用于续传。
- 本轮第一轮 `rsync --append --partial --timeout=25 --bwlimit=64` 运行后 SSH 报超时，退出 255。发送端显示约 6.8 MB，实际远端仅 5,062,656 字节，重新核对前缀后才继续使用断点。
- 后续 16 KiB/s rsync 追加实际将模型增至 5,849,088 字节；发送端积压并出现长时间停顿，本轮已停止该进程。该 5,849,088 字节前缀再次与 Mac 原件一致，远端无遗留 rsync 作业。
- SFTP 改用 `-R 1 -B 8192` 后，实际远端文件增至 6,291,456 字节，并能同时执行小型 SSH 核验；之后在约 6,676,480 字节时停止该轮，单独将窗口调为 `-R 4`，保持块大小 8 KiB，即最多 32 KiB 未确认数据。
- 用户已确认两台机器不在同一局域网。hp3090 的 Tailscale 连接走香港 DERP 中继；Mac 探测曾出现连续 3 次超时。本轮读取网络状态，未修改系统网络、SSH 持久配置或账号。
- 为当前仿真与接口适配选择 SDK 核心资料：排除附带的 OpenCV/Ruckig 源码，保留 103 个文件（961,744 字节，包括 54 个 msg/srv、4 份根文档及 C++/Python 示例与构建配置）。原创打包脚本直接从原 ZIP 读取，未在 Mac 展开大型依赖；压缩后仅 235,438 字节，每文件生成 SHA-256，包内附 `SHA256SUMS`。
- 模型 SFTP 最终正常退出。已在服务器实际核验 `q1_v3.zip` 为 11,065,174 字节，完整 SHA-256 与原件一致；`unzip -tq` 通过。该结果才作为模型迁移完成证据。
- SDK 核心包已完整到服务器；压缩包 SHA-256 一致，解压后 `sha256sum --check --quiet SHA256SUMS` 通过，103 个文件逐项一致。来源 ZIP、核心包与各文件身份见[核心资料清单](../data/manifests/q1_sdk_core_20261008.json)。
- 已解压模型并运行既有加载脚本，3 个 MJCF 均编译成功，各完成 100 步无 policy 被动仿真，初始/最终数值有限且无引擎警告。实际结果见[验证清单](../data/manifests/q1_mujoco_validation_20261008.json)。

## 关键决定及依据

- 使用标准 OpenSSH SFTP 小窗口完成续传，完整 SHA-256 与 ZIP 检查通过后解压模型。保留 rsync 失败与前缀核验记录，不能以发送进度或退出码代替完整性证据。
- 优先完成机器人模型，模型加载检查无需等待板端交叉编译资料。
- 先前命令的 SSH 保活设为 5 秒、3 次无响应即终止。慢传输下是否造成过早中断尚待验证；本轮只调整命令级设置，不修改系统网络或持久 SSH 配置。
- 当前采用 OpenSSH SFTP 的小窗口逐块确认；假设是约束发送端未确认队列能改善这条链路的可用性。仅记录实际续传证据，不将此机制推断写成已确认的网络根因。
- 仿真加载与 SDK 接口分析需要的核心文件已验证；OpenCV/Ruckig 源码及 Docker 工具链在真正构建/板端部署时继续获取。当前核心包没有完整 C++ 依赖，不宣称 SDK 编译通过。

## 版本、命令与证据

- 开始时 Q1 提交：`3951b69b998ead443dfc657345c6fe33630de08a`。
- 模型期望 SHA-256：`332a629e51fa7559e0cb16acb7276d600cb990102934e26861f80ccaa1443fb0`。
- 4,702,208 字节前缀 SHA-256：`7a3954742cd4e24f4f20e1bc086cfb68fdcac37f043340784ed59b9653684106`。
- 关键执行摘要：[本阶段日志](../logs/session_records/q1_migration_20261008.log)。原始 stdout/stderr 包括 `logs/q1_transfer_rsync_model_20261008.log`、`logs/q1_sdk_core_verification_20261008.log`、`logs/q1_mujoco_validation_20261008.log`，小型原始日志在 Mac 保留并复制到服务器 `logs/`。
- 后续日志：`logs/q1_transfer_rsync_model_retry_20261008.log`、`logs/q1_transfer_sftp_model_20261008.log`、`logs/q1_transfer_sftp_model_r4_20261008.log`。SFTP 被手动中断也可能返回 0，完成与否必须用完整大小/SHA-256 判断。
- 新打包脚本 Python 语法及 3 份 JSON 清单解析通过；服务器 `git diff --cached --check` 无错误，Git payload 初审通过：12 个文件，74,128 字节。补写本条记录后，提交前再次执行相同检查；最终字节数以终端日志为准。

本轮成功的模型续传命令（Mac；SFTP batch 为原创传输脚本）：

```bash
cd /Users/yu/Documents/ChatGPT/元启
sftp -R 4 -B 8192 -o BatchMode=yes -o IPQoS=none \
  -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
  -b script/q1/transfer_model.sftp hp3090
```

确认模型完整后，同样参数使用 `script/q1/transfer_sdk.sftp` 继续其余 3 个开发 ZIP；各自最后核验完整 SHA-256。

`reput` 要求目标已存在；如首次传不存在的 `docker_x86_native_x86.zip`，先在服务器 `data/references/q1/archives/` 创建空文件。SDK 核心包的首次传输使用 `put`，不能根据缺失目标的 `reput` 失败认定链路失败。

本轮已传 `script/q1/transfer_sdk_core.sftp` 中的轻量核心包并验证。原始 SDK ZIP 和编译工具链仍单独记为未完整迁移；核心包不等于可完整构建的 SDK。

SDK 核心包生成命令（直接读取原 ZIP，Mac 只生成约 230 KiB 的小型包）：

```bash
python3 script/q1/package_sdk_core.py \
  --source /Users/yu/Documents/ChatGPT/Q1/sdk_q1-v1.0.0.0.zip \
  --output data/sync/q1_migration_20261008/q1_sdk_core.tar.gz \
  --manifest data/manifests/q1_sdk_core_20261008.json
```

模型基础运行的实际命令（hp3090）：

```bash
cd /media/yu/FAFF-E977/YuanQi_Q1
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python \
  script/q1/check_mujoco_assets.py \
  --asset-dir data/references/q1/models/332a629e51fa/q1_v3/xml \
  --output data/inspection/q1_mujoco_20261008.json
```

SDK 核心包解压位置为 `data/references/q1/sdk_core/sdk_q1-v1.0.0.0/`；在其父目录运行 `sha256sum --check --quiet SHA256SUMS` 可重复核验。重新打包会将新清单设为 `PACKAGED_LOCAL`，必须重新传输并核验后再更新状态。

复现前必须核对远端实际状态、原件完整性与是否有本轮遗留 rsync 作业。完整文件的身份清单为 `data/manifests/q1_resources_20261008.json`。

## 结果与边界

最低开发资产已在 hp3090 到位，可以开始 Q1 运动学重定向、仿真环境与控制适配开发。实际运行结果如下：

| 资产 | nq / nv | 执行器 / 传感器 | 实际检查 |
| --- | --- | --- | --- |
| `q1_v3.xml` | 29 / 28 | 0 / 0 | 编译 + 100 步被动仿真通过，无警告 |
| `q1_v3_full_range.xml` | 29 / 28 | 0 / 0 | 编译 + 100 步被动仿真通过，无警告 |
| `q1_v3_mc.xml` | 29 / 28 | 22 / 49 | 编译 + 100 步被动仿真通过，无警告 |

此检查只验证模型/网格加载和基础引擎运行，不证明站立、跟踪、训练或真机控制通过。完整 SDK 与工具链 ZIP 仍未迁移完成，真机 IMU、关节参数与动捕合同仍待确认。异地链路仍慢，未宣称网络故障根因已修复。

## 保存与同步状态

本阶段 README、代码、执行摘要与 3 份小型清单按 Q1 分支规则审查后提交。最新同步结果以本地 HEAD 与远端 `refs/heads/Q1` 比对为准，不写入自身提交号。原始工件不进入 Git。

模型与 SDK 核心包均已在服务器验证；完整 SDK ZIP 和两个工具链 ZIP 尚未全部迁移。百度网盘登录/传输能力仍未验证，本轮未上传到网盘；浏览器连接器遇到不支持当前认证方式的错误，未获取可用网盘传输通路。备份仍为 `LOCAL_ONLY`，原始附件保留，没有删除原件。

## 下一步与恢复入口

1. 从已验证的 `q1_v3_mc.xml` 开始建立 22DOF 控制/重定向适配与可重复的跟踪环境。基础加载已通过，无需等待完整 SDK 工具链才能开始仿真开发。
2. 确认动捕样本的字段、旋转表示、坐标系、尺度及时间戳；真机侧继续澄清 IMU 与关节限位/力矩差异，暂不操作机器人。
3. 后续需要完整构建/部署资料时使用小窗口续传其余 ZIP，并逐项完整 SHA-256 核验。继续前读本文件、[当前状态](README_CURRENT_STATE.md)、[资料评估](README_Q1_SMPL_20261008.md)及资源清单，重查实际作业和 Git；本轮传输和运行检查最终均已结束，不遗留训练或控制作业。
