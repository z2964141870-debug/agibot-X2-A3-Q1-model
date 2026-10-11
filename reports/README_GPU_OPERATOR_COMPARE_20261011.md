# GPU调用时序对照 E27–E29

记录日期：2026-10-11。状态：E27通过，E28确认重启并停止，E29未登记/启动。

## 目标与范围

用户“继续找凶手”授权继续普通GPU隔离排查。E24普通1024首操作附近重启，E26一次同步诊断通过，不能直接归因。此次固定三个独立编号，逐项完成保存后才登记下一项，首异常/新boot停止整链；无A3/Isaac/RL/模型加载，不改驱动、功率或BIOS，不复用历史job。

| 实验 | 与前项相比的主要改变 | 固定配置 |
| --- | --- | --- |
| E27 | E26的CUDA_LAUNCH_BLOCKING从1变0 | 一次1024 float32随机矩阵×单位矩阵、seed0、TF32关闭、全部额外同步与细日志 |
| E28 | 单次改为30秒循环 | blocking=0、额外同步与细日志保留，占空比目标25%、最少10ms sleep |
| E29 | 移除四个中间显式同步 | 同E28循环与校验，分配和每轮末尾同步保留，趋近E24原操作序列 |

E27单次模式计算路径与E26相同，新增循环计数/计时和固定比较入口有轻微CPU记录差异；不宣称两进程二进制完全相同。E28引入循环/间歇，不能把成功差异只解释为时长。E29仍保留逐调用fsync及细分finite.all/item，时序仍与E24不同，成功不排除原问题。每条BEGIN/CALL_RETURNED/END记录只表明相应调用边界，异步matmul返回不代表GPU完成；E29的allclose内部还会读回布尔值，不能细分其内部CUDA调用。

## 完成内容与关键决定

新比较入口绑定固定case、原boot、前项COMPLETED且无ERROR/未完成边界、5源码SHA。READY一次消费；所有unit非自启Restart=no，RuntimeMaxSec240、CPUQuota100%、MemoryMax4G，worker最长120秒。每项结束后60秒独立停载监测，后续登记重新核验GPU无计算进程/温度安全。CPU90/GPU85°C连续两次5秒采样、RAM保留4GiB；遥测/内核不可读、关键内核事件、计算异常、超时均停止。未知SSH失联不当作重启，不重复启动。

Mac只读SSH镜像每秒复制阶段/健康/内核记录，前1MiB逐条fsync，最后500条按收到的块fsync并替换小型滚动文件；完整原始日志留服务器、不入Git。短E27能完整复制，循环分支可能仅保留前缀与最后窗口，必须按实际覆盖报告，不宣称完整镜像。前项报告及清单先发布再进入下一项，避免仅在聊天中留中间结果。旧hp3090/yuanqi-a3仍PAUSED，日研不变，不需要开机恢复或新heartbeat。

## 版本、命令与证据

- 主机：ssh hp3090；根`/media/yu/FAFF-E9771/YUANQI_A3`。
- 代码：`script/diagnostics/gpu_operator_compare.py`、`collect_gpu_operator_compare.py`；不修改E26执行版本。
- 现有环境：PyTorch2.8.0+cu128/CUDA12.8/驱动595.84 open module，版本以每次实际日志为准。
- 登记：`/usr/bin/python3 -m script.diagnostics.gpu_operator_compare E27 register`；E28/E29替换case，必须前项通过，无重复/重置。
- 监督运行：同模块`<case> run`；只读镜像`<case> watch`；只读遥测总结`<case> summarize`。
- job/state/results：`data/experiments/gpu_operator_compare_20261011_<case>/`。
- 原始日志：`logs/gpu_operator_compare_20261011_<case>/`，LOCAL_ONLY。
- 小清单：`data/manifests/gpu_operator_compare_{registration,result,evidence}_20261011_<case>.json`。

## 实际验证与结论边界

开始前确认原boot c7805e85，CPU48/GPU30°C、无GPU计算、可用RAM约29GiB；Git main干净。13项无CUDA安全/日志契约检查在Mac和服务器通过。代码/E27登记发布9bbf363f3fa2b65222153a3b887cb3b945ec769f，main远端SHA一致。

### E27：关闭blocking的单次对照完成

17:56:12登记，17:56:40.605启动，17:56:41.602所有调用完成、校验true，worker退出0。17:56:45开始独立停载观察，17:57:46.222完成超过60秒观察；unit inactive/dead、Restart=no，原boot不变。15采样CPU最高49/GPU30°C，无保护/记录关键内核事件。

56阶段记录全部完整、无ERROR/未完成/半行，Mac镜像56/56与主机逐条匹配；worker_result SHA独立核验通过，核验代码`script/diagnostics/record_gpu_operator_compare.py`及小清单`gpu_operator_compare_evidence_20261011_E27.json`。本轮仍保留中间显式同步和fsync，只有一次运算，不是完整异步长时间负载通过。它与E26均未复现重启，因此不能据此认定blocking是原重启的原因，也没有排除硬件或库问题。

E27结果发布48c8ddbb11883a54d9ed5cabd090da0d934c6883并核验远端后，独立登记E28；登记提交912218bb3f263ea8c07d636872af6af8c56e6cb2已推main、远端SHA一致。所有case不可恢复，禁止重复运行E27。

### E28：30秒计划中确认重启，全链停止

18:00:33登记、18:01:04.257启动。原boot `c7805e85-7467-4691-9037-b805ccae131f`，18:02:45只读恢复核验确认新boot `34cc7c88-6bb0-491d-800a-3714e7db9f37`。该次uptime74.33秒，推算新boot约18:01:31，非故障精确时刻；确认观察区间为末次阶段18:01:05.596574至首次新boot读取18:02:45。18:04:10标准库只读总结把state保存为STOPPED_REBOOT/CONFIRMED_NEW_BOOT_NO_RESUME。没有worker_result，未完成30秒测试，没有计算错误返回或整体数值验收结果。

| 操作阶段 | 最后持久化证据 |
| --- | --- |
| 导入、CUDA初始化、分配和分配同步 | 全部完成 |
| 第1、2轮矩阵乘、校验、有限性、归约、标量读回及全部同步 | 全部完成，校验true |
| 第3轮MATMUL与MATMUL_SYNCHRONIZE | 完成 |
| 第3轮ALLCLOSE | 18:01:05.596197 END；18:01:05.596574 VALUE=true |
| 下一预期ALLCLOSE_SYNCHRONIZE | 没有保存BEGIN，不能证明该调用已经进入或未进入 |
| 第3轮后续操作、结果保存、60秒观察 | 未完成，无result |

服务器94阶段记录、3个MATMUL BEGIN和END、前2个完整循环，无ERROR/半行；没有未闭合span，表明日志停在两调用边界之间，不能据此认定程序无故障。记录写入/fsync与GPU异步时序仍可能受故障影响，最后记录不是根因证明；第三轮allclose已经返回true，不能把它直接称为失败API。

外部SSH镜像只收到13条记录，其中8条阶段日志，最后CUDA_INIT BEGIN；11条有offset的记录与服务器逐条匹配，但未覆盖后续86条阶段日志。初次巡检仅能报告镜像停在初始化；恢复后已明确更正为服务器保存的第3轮allclose完成。不能把较旧的镜像尾当实际最后阶段。

仅2健康样本（登记与启动），CPU最高46/GPU30°C；最后启动采样CPU40/GPU30°C、21.93W、可用RAM约29.18GiB。采样未覆盖运算后空窗，不能排除瞬时温度/功率变化。仅1次运行内核查询为空；按UTC epoch再次查询前boot故障附近内核返回`-- No entries --`。前boot系统journal覆盖末端18:00:41、内核尾停于15:40启动段，没有故障附近崩溃证据，不能据无Xid/匹配排除驱动或硬件。`/sys/fs/pstore`普通读取权限不足；仅尝试sudo -n只读列目录，返回需要密码，未获取提权、不读取凭证。`/var/lib/systemd/pstore`可读但目录为空，无法取得崩溃转储。原始检查保存`after_reboot/pstore_access.json`。

重启后只做标准库日志/元数据/哈希与nvidia-smi只读遥测，没有CUDA计算、模型加载或训练。unit inactive/dead/Restart=no，无GPU计算进程；旧A3自动恢复两个unit inactive/disabled。E29前置门禁不满足，确认没有E29实验目录，全部后续停止。

### 结论与下一步边界

| 对照 | 实测结果 |
| --- | --- |
| E26 blocking=1、单次、额外同步 | 通过及60秒观察同boot |
| E27 blocking=0、单次、额外同步 | 通过及60秒观察同boot |
| E28 blocking=0、循环30秒计划、额外同步 | 前2轮完成，第三轮校验后日志中断并确认新boot |
| E29移除额外同步 | 未登记/启动，按首故障停止 |

新的独立证据再次证明无需A3/Isaac/RL专有训练代码也能观察到重启，优先对象仍为通用GPU执行链路及其触发的系统问题。一次单次通过与一次循环重启不能建立单变量因果，也不能定位驱动/CUDA库、GPU/PCIe/供电或其他硬件。显式同步保留时仍会重启，不能据E26通过宣称问题已经解决。

本轮不再升档或重复复现。下一轮计算前先改善故障附近内核证据覆盖，核对已有其他软件环境是否可作同程序对照；仍不升级/更换驱动、改变功率/BIOS或盲目恢复A3。项目卷只读核验为NVMe ext4，不能凭路径名称猜测外置NTFS问题。具体API/部件根因仍UNKNOWN。

重启后仅读取已安装包METADATA，没有import torch或GPU计算。现有`/home/yu/miniconda3/envs/twist`登记torch1.13.1+cu117，是与当前x2-sonic-isaaclab的2.8.0+cu128不同的运行库候选；未验证它当前可导入、驱动兼容或能通过同程序。`h-gpt`登记torch2.8.0和同nvidia-cublas-cu12 12.8.4.1，不宜当作充分独立的CUDA版本对照；`xrobo-pico`登记CPU版2.13.0不作GPU对照。原始包清单`after_reboot/environments.json`。未来同一最小程序/日志/负载比较现有twist与当前环境，可帮助判断问题是否随软件栈变化，但两者仍共用驱动/硬件，单次通过也不能排除它们。本阶段未运行这个对照。

### 最终保存与核验

结果/镜像覆盖/停止证据分别为`gpu_operator_compare_result_20261011_E28.json`、`gpu_operator_compare_evidence_20261011_E28.json`、`gpu_operator_compare_stop_20261011_E28.json`。完整阶段日志29,828字节，SHA e7e7dbe1a69656a372ccb3a2376cfa8001729e52480e33ded469ec2497179f12，Mac下载SHA核验；外部镜像5,396字节。原始数据、内核与snapshot留logs/LOCAL_ONLY，小清单和README/索引/current state经审计scoped提交推main并核验远端；最终SHA由Git历史提供，保留并行修改。不启用自动重试或A3恢复。

## 保存、同步与下一步

代码与预登记报告经diff/载荷审计后scoped提交推main并核验远端；大日志LOCAL_ONLY。先完成E27并发布，再根据实际结果进入E28、E29。首次故障即停，保存最后完成/未完成调用、确认boot区间和未做分支，禁止自动继续A3。
