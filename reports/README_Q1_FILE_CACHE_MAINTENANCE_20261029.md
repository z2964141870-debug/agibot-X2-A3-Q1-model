# Q1文件页缓存维护

日期：2026-10-29，北京时间。用户明确要求“没有用的文件缓存全清了啊，都是我们诞生的，定期定量清理掉”。此最新授权覆盖此前本轮不清缓存的约束，仅缓存释放；文件、权重、原始数据和历史实验保留。

## 范围与实现

唯一用户25386、实体根`/opt/data/private/ZHY`、boot`4d859008-4c83-4a6d-814d-cbb9dfe885c9`与GPU`GPU-26028785-9fb9-45af-e614-4cc53af15d08`。维护工具`script/project_tools/reclaim_q1_file_cache.py`只处理根下data/logs/script/reports的普通文件，不跟随符号链接、跳过.git及其他进程已打开/映射的inode；先核空闲和原GPU锁，遇项目训练/渲染/交付或GPU计算即跳过。未修改宿主机全局drop_caches，其当前接口亦不可写；不触及其他容器或路径。

用Linux mincore和PROT_NONE映射读取驻留位，不读取文件内容或使其重新入缓存；仅对有驻留页的文件范围执行POSIX_FADV_DONTNEED，释放可释放的干净页。未刷全局sync、未清脏页或锁定页，不能承诺缓存绝对归零；记录前后cgroup实际cache和total差值，advice范围字节不是实际释放字节。不删除数据、升级环境、动认证或提高24GiB/30GiB保护线。

首次有界清理：最多180秒/256GiB advice范围，目标cache<=0.25GiB，覆盖本项目普通文件；范围和时间上限触发就停止，不无界重试。后续每30分钟检查，cache>8GiB且项目空闲时每轮最多8GiB advice范围/120秒，目标cache<=4GiB；保留游标避免每轮只扫描同一大文件，1800秒冷却。正常无变化保持静默，只在失败、异常回涨或需要处理时通知。定期维护复用可用原监督或独立本线程heartbeat，不修改yuanqi/supermimichp-x2等任务。

## 验证与状态

实现提供路径边界、符号链接/已打开文件和inode替换保护的三个本地反例测试。首次本地inode反例因Mac临时目录别名路径不同触发边界保护，改为测试根先resolve，不放松实际服务端根检查。真实Linux mincore/FADV和前后释放量将在独立首次清理收据登记，准备或单测不代表已经清理。

当前实测清理前：容器总29398269952B、cache27909169152B、shmem0、dirty0，无项目训练/交付和GPU计算。首次工具在/proc/1/cwd权限保护处停止、0文件删除且未执行advice，原失败收据保留；PID1实测为不可ptrace读取的基础设施pod进程。修正仅允许记录并跳过该精确PID1/pod，其他不可读取进程仍停止，不修改权限或容器基础设施。现有一次修复base16预检授权尚未使用；清理满足原准入后可继续该一次零更新，不增加训练预算或故障重试次数。GAIT_GOAL01保持ACTIVE，采纳null，不因维护或等待结算目标。

重要代码和本页/索引/CURRENT_STATE/AGENTS须显式audit/commit/push Q1、独立远端40位SHA，再镜像Mac主根和25386。实际命令、清理与自动维护状态将在本页和小清单更新。

## 已完成的清理与定期维护

代码`b6c28069be567e4f4d77c4b8dcd8d5e96daccfba`独立远端已核，修正后首次范围扫描于北京时间21:59:16开始，125.646547秒正常完成，375504候选文件/67947有驻留页文件获得advice。执行mincore扫描的230291508730B是文件范围而非内容读取量；advice范围31575094422B亦不当作释放量。

- 文件cache：27909169152B（25.99GiB）到711090176B（0.66GiB），实测下降27198078976B（25.33GiB）。
- 容器总量：29400752128B到工具收尾2062270464B；工具退出后实测1533300736B（1.43GiB），RSS18907136B，shmem0。收尾匿名RSS曾包括工具目录扫描列表，自身退出后释放，不能当训练内存。
- 原OOM0；没有读取大文件内容、删除文件、全局drop_caches或改24/30GiB守卫。剩余0.66GiB不在本轮可释放普通文件范围内或仍受保护，没有强制归零；mincore驻留和cgroup记账也不是严格同一归属口径。

实际命令：在唯一服务器根用python3执行`script/project_tools/reclaim_q1_file_cache.py --once --trigger-gib 0.25 --target-gib 0.25 --batch-gib 256 --maximum-seconds 180 --output logs/q1_cache_initial_v02_20261029.json`。原/proc/1保护失败收据保留，修正后完整扫描不是训练重试。

已通过原生automation工具创建本线程独立heartbeat `q1`，名称“Q1文件缓存定量维护”，ACTIVE每30分钟；本机automation.toml已读回核实。它只在空闲且cache>8GiB时最多处理8GiB advice范围/120秒，低于阈值或忙碌静默，不启动任何物理/推理/训练/agent/goal。原q1-rf08-4096监督及yuanqi/supermimichp-x2等其他任务保持，不以旧监督覆盖本目标的新授权。该自动化是缓存维护，不是定期删文件。

## 一次修复base16已通过

清理后原24GiB准入已满足，使用用户“允许修复后仅一次预检”的唯一追加授权。北京时间22:03:11启动controller211075/start_ticks19623628、worker211081；22:04:10全流程COMPLETED、guard WORKER_COMPLETED/reap0，1564冻结来源前后不变。准入容器1564528640B、距原30GiB线30647726080B，原boot/GPU/OOM0。

真实H32 native WM/actor/critic反传通过，同replay末历史起点逐值一致且输入未变；三个有限非零梯度范数约1，WM1/actor1/critic20次optimizer调用全部拦截，optimizer实际0、参数未变、normalizer恢复、独立评测保护活跃状态/RNG。worker收据记录采集272转移、独立评测960环境转移；这是预检，不是正式训练或步态改善，旧跨仿真物理方向结论仍未成立。

原controller/worker已退出、GPU15MiB/0%无计算；两线正式训练仍0秒。该追加授权已用完，不自动再做失败预检。base4096/PPO16/4096尚未执行；下一步先完成既有离线提前跌倒指标/结果归档与交付工具，再推进依赖阶段。goal保持ACTIVE、用户采纳null，不因清理或预检通过结算目标。[全部实测与小工件清单](../data/manifests/supermimic_q1_cache_maintenance_base16_20261029.json)。
