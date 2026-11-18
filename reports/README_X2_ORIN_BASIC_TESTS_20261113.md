# X2 SoC1 文件迁移与基础测试

记录日期：2026-11-13；阶段：X2_ORIN_BASIC01。状态：文件迁移与无电机输出的基础测试完成。设备末次取证：13:13:55 北京时间。

## 用户 Done / Todo

- Done：已将 MorphMimic X2 human 模型、源码、资产和两段最新 CLOTHO FGP 人体录制迁入 SoC1，建立独立环境；19 项测试、C++ 本机 IPC、两段离线推理和 Mac/Orin 输出对齐通过。
- Done：Orin CPU 离线处理 p95 约 8.9ms；修正服务端新鲜度参数未生效的问题。六宫图与数值证据已保存。
- Todo：后续再验证 Mac 最新 FGP 到 Orin 的实时数据传输、时钟与完整周期。当前没有连接机器人反馈或电机输出，不据本阶段宣布真机遥操可用。

## 范围与实际设备

用户要求“开始尝试把文件迁移过去，先不要真机运行，先做基本测试”。本轮只建立项目目录和用户虚拟环境，运行离线模型与 loopback IPC；`publish_allowed=false`、`hardware_commands=0`、机器人反馈订阅 0、训练更新 0。未使用 sudo，未改 MC、ROS、厂商服务、SSH 或网络配置；Q1 原暂停训练未操作。

固定原生交互密码路线：Mac `ssh sshuser@100.92.227.62` → Windows 内 `ssh agi@10.0.1.41`。文件使用相同两跳原生 SCP，未用转发端点、代理、隧道或免密替代。Windows 是 cmd.exe，提交使用 CR。认证内容不落文件。

| 位置 | 实际核验 |
|---|---|
| SoC1 | Orin NX，Ubuntu22.04.5，内核5.15.148-tegra，L4T R36.4.3，系统TensorRT10.7.0，CPython3.10.12/aarch64 |
| SoC0 | RK3588 AGI Board；本轮未安装文件，不当第二台Orin |
| 设备项目根 | `/home/agi/x2_morph_human_20261113` |
| 设备环境 | `data/environments/orin-py310/bin/python` |
| Windows 暂存 | `C:\Users\sshuser\x2_morph_staging_20261113` |

SoC1 末次 `MemAvailable=12041908 kB`，磁盘可用 `377744633856 B`。没有清理系统缓存；自己的测试/策略进程已全部退出。

## 迁移与环境

V2 包 99,500,160B / 80 成员，Mac、Windows、SoC1 SHA 相同：`802a0f0a791520ef06f932369527578d1e0ada2542441c000b6ca8296aeb73c4`。包含固定官方源码、许可证、X2 human 权重、参数、MJCF/mesh 与 C++ 探针；不使用自训 Q1/G1 权重代替 X2。模型 SHA：`cf95194793e4732447f66432fc2381d82dd7c3d82c94da15ad074f2cfe295512`。`doctor --meshes` 核 56 资产 / 160958236B 通过。

另传 `real_worn_globalpose.jsonl`（2527813B，SHA `44e2087ec2e08674624ce87843b501c594ca9879b7293ac4a0b0736fc9ff0fcb`）和 `real_manual_active_gp.jsonl`（2530042B，SHA `c33d19548cb90281fb7b09b280b1fa891229abbe32db2f936f80e189701a68ea`）至设备 `data/`，逐件核同。两份为最新 CLOTHO FGP 已保存人体输出；实际 FGP/PyTorch 前端仍在 Mac，未迁移或测量它在 Orin 上的性能。

联网 pip 两次等待无进展后仅终止自己的安装命令，保留日志。改为在 Mac 准备 Linux aarch64 / CPython3.10 的 14 个轮子，`X2_ORIN_PY310_WHEELS_20261113.tgz` 为52265906B / SHA `0f725b13fdb5f6c2c4e53da62e7bc250514d1fe9c1b5fe31e0ab7a1b03da350f`，设备逐轮子 SHA256SUMS 全过并离线安装。完整锁见 `script/x2_morph_human/requirements-orin-py310.lock.txt`。

登录 ROS `PYTHONPATH` 会让 venv 的 pip check 看到系统 launch-ros 缺 PyYAML；使用 `env -u PYTHONPATH -u PYTHONHOME` 后项目环境 check 通过，未安装/修改系统 ROS 依赖。新增 `run_orin.sh` 做当前进程隔离。CPUExecutionProvider 实际使用；系统有 TensorRT 不代表此 CPU wheel 已提供 CUDA/TensorRT EP。ORT 的 DRM GPU discovery 警告保留，未影响 CPU 推理。

## 实测结果

| 检查 | 实际结果 |
|---|---|
| 首次原始18测试 | 18/18，3.640s；原日志保留 |
| 服务修订后测试 | 19/19，4.875s；新增50ms服务新鲜度、150ms静默后无目标断言 |
| C++17 编译 | `-Wall -Wextra -Werror`，系统libzmq4.3.4，无厂商SDK链接；二进制SHA `a0a67411112cc2686dabb6e2b93a89fe01a3b43a6c84f4b6f237664dd1d1d529a` |
| C++ IPC 修订后复核 | 80请求，31真实ONNX调用，9启动等待，0.952358s，HAL命令0 |
| Mac/Orin匹配回放 | 1723帧，22次相同Hold，状态序号/tick/时间字典/关节顺序完全同；最大目标差1.445413e-6rad < 2e-5门 |

两段人体录制均设置 `--clock-50hz --max-interp-gap-ms 100 --max-freshness-ms 100 --frames 2000`，使用固定合成机器人反馈，**没有物理仿真**：

| 已录人体流 | 输出 | 启动等待 | 中途Hold | CPU处理p50/p95/p99/max (ms) | 离线墙钟(s) |
|---|---:|---:|---:|---|---:|
| real_worn_globalpose | 991 | 11 | 0 | 7.830 / 8.896 / 10.868 / 15.451 | 8.469 |
| real_manual_active_gp | 1723 | 11 | 22 | 7.548 / 8.952 / 10.592 / 17.767 | 14.192 |

旧 Mac 开发仿真的 active 流为21次Hold；本轮显式100ms freshness为22次，不能跨参数直接说部署造成回退。因此仅补做一次同参数 **Mac离线** 回放，1723帧/22Hold与Orin精确匹配，未重跑原八组物理。六宫图为跨平台离线处理/一致性图，**不是训练曲线**。

代码检查发现 `serve --max-freshness-ms` 原来未传给 Runtime，实际沿用底层3000ms。已补 `max_age_ns` 传参和启动JSON声明；独立进程测试先证31目标就绪，再在150ms无新人体输入后验证 `ready=false` / 不含 `q_target_rad`。设备覆写只涉及 `__main__.py`、`test_runtime.py`、`run_orin.sh`，与Mac大小SHA相同；原V2压缩包未变。

## 命令与证据

环境安装/复现命令见 `script/x2_morph_human/README.md`。设备原日志为 `logs/unittest_orin{,_v2}.log`、`logs/cpp_build.json`、`logs/cpp_ipc{,_v2}.log`、`logs/requirements-orin-py310.lock.txt`、`logs/orin_basic_v2_receipt.json`；输出 `data/analysis/orin_{doctor,doctor_v2,cpp_ipc,cpp_ipc_v2,real_worn,real_active}`。

Mac原件与图表：

- `data/analysis/X2_ORIN_BASIC_20261113/`：首轮27文件SHA全过、两段原始回放和同参数Mac回放。
- `data/analysis/X2_ORIN_BASIC_R2_20261113/`：修订后11文件SHA全过及原设备收据，不覆盖首包。
- `data/analysis/X2_ORIN_BASIC_20261113/parity_analysis/`：summary、逐帧CSV、`orin_offline_parity_6panel.{png,pdf}`；PNG已目视。
- `data/deliveries/X2_ORIN_BASIC_RESULTS_20261113.tgz`：5121223B / SHA `8ee1fb9177b24744f0ca7aedfade36949084ee1c6cd96e018f4cc490de00fcfc`。
- `data/deliveries/X2_ORIN_BASIC_RESULTS_R2_20261113.tgz`：36622B / SHA `7effeb83944daa117a6bd1f5b5fdc4554d247868a2678c0bb281d6c6489e4230`。
- `data/deliveries/X2_ORIN_BASIC_EVIDENCE_20261113.zip`：本阶段README、六宫图、关键数值、日志与逐件清单；无新增MP4，因为本阶段没有物理或真机录像。

小记录：`logs/session_records/x2_orin_basic_tests_20261113.json`。Git和镜像核验以 `logs/session_records/x2_orin_basic_publication_20261113.json` 的实际收据为准；模型、轮子、录制和压缩包不入Git，网盘状态 LOCAL_ONLY。

## 边界与恢复下一步

本轮证据支持“设备侧可加载X2官方模型、可靠输出31槽建议目标、与Mac数值一致”。**不支持真机平衡、自然步态或实测端到端延迟结论**。`--clock-50hz` 是离线虚拟时钟，不是实时调度认证；180ms真实预览等待仍在。原 summary 的静态 `simulation_fallen=false` 在 `simulation_metrics=null` 时没有无跌倒证明力；doctor 的 `orin_latency_status=NOT_MEASURED` 是该命令自身的范围，处理耗时以回放数据为准。

Agent Done：基础包/两录制/14轮子完成两跳迁移，独立venv/lock/进程环境隔离完成；19测试、C++编译与新IPC复核、两离线回放和单次匹配Mac对照完成；新鲜度接线修复已上板，临时服务已回收。没有电机命令、反馈订阅、GPU配置/功耗调整或训练。

Agent Todo：先保留当前advisory部署，核新镜像回执；后续独立验证Mac实时FGP输出至Orin的时钟/丢包/完整周期，并保持固定合成机器人反馈。用户本轮禁止真机运行，**不自行启动shadow_reader或接writer**。真实反馈/外参/PD/厂商SDK ABI和硬件动作均留后续明确任务。若未来改用TensorRT/CUDA，另做模型等价与完整周期测试，不覆盖CPU证据。
