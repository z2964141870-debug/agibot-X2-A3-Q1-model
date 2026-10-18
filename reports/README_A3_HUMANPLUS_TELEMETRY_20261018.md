# humanplus内存与内核日志适配

10/18北京时间10:40接续：[HP05运行验收](README_A3_HUMANPLUS_RUNTIME_20261018.md)已集成温度/进程/GPU/三次启动门/900秒监督器，13标准库控制Mac/新机各通过。CPU/fused CUDA4事务及固定MJCF10步通过；PhysX在EULA EOF启动错误处STOPPED，完整环境仍未验收，旧遥测探针不复跑。12实际监督采样无OOM/内核异常、CPU最高81/GPU42°C。当前0策略训练，但已有合成张量/optimizer及MuJoCo仿真；覆盖下文历史“尚未集成/0张量”状态。

10/18北京时间10:20接续：[HP04直接迁移](README_A3_INPUT_MIGRATION_20261018.md)已完成，035 PT及10历史轨迹与52小文件共63项独立大小/SHA通过，输入10/10。当前INPUTS_VERIFIED_WAIT_RUNTIME_VALIDATION，覆盖下文历史WAIT_USER_LARGE_TRANSFER；本遥测探针不复跑。完整监督器和环境小运行仍须独立登记验收，0张量/模型/训练。

记录：2026-10-18。状态：标准库代码及只读元数据检查完成；科学运行保护集成、环境验证和训练未开始。此为基础设施适配，不是新的策略实验。

## 目标与范围

在/home/humanplus/YUANQI_A3实现可审查的cgroup v2内存与内核日志证据，不沿用ZHY的v1口径、boot或历史OOM例外。仅本工程小代码、标准库合成测试及只读/proc、cgroup、journal访问；未改共享环境、内核权限或其他进程。

## 完成内容

新增script/a3/humanplus_health.py及test_humanplus_health.py。可用量取物理MemAvailable与每个可见祖先有限memory.max/high减memory.current的最小值，不另加文件缓存、不扣减RSS冒充可用量。读取每层memory.events与memory.events.local，保存OOM/oom_kill/oom_group_kill基准，增量即拒绝；计数倒退、层级、限额或字段改变也拒绝。保留8GiB启动和4GiB运行保留线。

真实v2根层/sys/fs/cgroup没有上述memory控制文件。只对完整挂载根、五项文件全缺的原生根层显式记录root_unconstrained，并以物理MemAvailable和内核日志覆盖全局内存；非根缺项或根层部分缺项仍失败。没有把任何子组的缺失限额默认为无限。完整层级不可见时拒绝。

内核读取普通用户已验证的journalctl -k -b -o json --all --no-pager，无sudo。核验所有记录boot、唯一cursor、单调时间与字段；给固定历史前缀计算SHA，下一次全量读取必须保持该前缀，日志丢失/改写/轮转丢基准均拒绝。原FATAL事件及明确OOM消息都拒绝，包含外部OOM，无ZHY历史豁免。未删除日志或重基准。该证据只覆盖可读、实际记录的journal，不证明未记录的故障不存在。

## 版本、命令与证据

- 代码SHA：humanplus_health.py为36b20feb2a2d7d3e177c9352763e1f5414590069102f49c6b86bb99251e67fa0；测试为df85aa54f0ca34fcfe5f071e4a1f6532984890e0374266a2183eab49f56ed0fb。
- 标准库测试：python3 -m unittest script.a3.test_humanplus_health -v；Mac与humanplus各14项通过。覆盖祖先限额、high、超限、根层无控制文件、缺项、OOM计数、日志追加/静默/丢失/篡改、boot、重复cursor、外部OOM与Xid、原内存阈值。
- 只读计划data/manifests/a3_humanplus_telemetry_plan_20261018.json预先固定源码SHA、预期boot、2次采样、间隔5秒、60秒总预算、无自动重试及0张量/模型/仿真/optimizer。
- 2026-10-18北京时间01:16单次只读探针5.126秒完成，两次可用22,654,746,624/22,636,445,696字节；同boot afb6796b-3883-4df4-99ea-bc381c52ea13。可见会话/祖先max和high均max，OOM计数均0。两次journal均1410条、前缀SHA相同、0FATAL、0新增OOM。
- 原始收据logs/a3_humanplus_20261017/telemetry_readonly_probe_20261018.json，SHA256 7ca40672be0646e8c816d8116e55632619d98017d57b3281fd0fb5ad780aab21。该探针已完成，不重复执行以堆检查次数。
- 精简结果data/manifests/a3_humanplus_telemetry_result_20261018.json。

## 资源变化与未完成项

01:09只读盘点inventory_heartbeat_1708.json显示EgoLocate计算进程已退出，无GPU计算进程和可见Python计算，GPU为1408MiB/0%/43°C，可用22,635,192,320字节。未终止任何进程，未清缓存；这是一个历史快照，后续启动仍须重新核验，不能当三次5秒启动验收。11个待迁大文件仍0到位，035模型仍缺。当前转为WAIT_USER_LARGE_TRANSFER，文件迁移仍按原交接清单，不重复催用户或重试HP02/HP03。

仅内存和日志采集/比较原语完成。尚未集成具有CPU90/GPU85连续两次5秒、可见进程/GPU空闲、三次启动采样、4GiB保留、900秒期限及只停止本实验子进程的完整阶段监督器。没有登记科学运行、导入张量、加载模型、运行ABI/PhysX/fused CUDA控制或产生新策略结果。元数据检查通过不等于运行保护或环境已验收。

## 保存与同步

代码、计划、精简结果、阶段README、索引和current state按scoped审计本地提交。原始采样LOCAL_ONLY、不进Git；GitHub既有SSH鉴权失败，未恢复写权限，不称同步成功。保留既有5个未推送提交及其他修改，不复制密钥或改认证。

## 下一步

035 PT等必要输入现已逐文件SHA核验。资源重新满足时，在独立登记的新机运行验证中完成监督器集成与环境小验证，再固定新机六衣服/四官方035匹配基线，最后原预算8更新短训练及最终checkpoint匹配评测。原实验、质量门和停止规则不变。恢复仍先读本页与README_A3_INPUT_MIGRATION_20261018.md。
