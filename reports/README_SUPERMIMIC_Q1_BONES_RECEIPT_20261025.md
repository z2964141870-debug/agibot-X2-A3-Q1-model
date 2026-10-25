# BONES-SEED真实G1原包接收

记录日期：2026-10-25，北京时间。阶段：SM-Q1-BONES-A02。状态：原包完整接收与冻结候选提取完成；原速小批转换待，尚无新BONES合格参考或训练。

## 目标与授权

用户通知BONES即将拷贝，并确认`root@huanyu:/opt/data/private/ZHY`根目录已有`g1.tar.gz`。本次接收和使用该已有文件，覆盖此前暂停处理BONES资源的安排；不恢复网络下载、凭据申请或旧失败实验。继续SuperMimic for Q1世界模型复现。

## 现场核验与决定

原文件`/opt/data/private/ZHY/g1.tar.gz`实测23499973647字节、分配23499984896字节、inode99228145，gzip头1f8b。两次核验mtime_ns1792899332166613109和大小保持，第二次未见可读proc中的同inode打开FD、0 proc目录权限拒绝。尚未核完整SHA或gzip CRC，不能先称完整接入成功，也没有源端SHA证明传输双端一致。

当时容器总用量34359689216字节接近32GiB，但rss约0.573GB、cache约32.92GB、shmem约12MB、OOM14。归档顺序处理需文件级释放已读缓存、限制Python解码缓冲；不清空全机缓存，不降低旧训练保护。只在统一根四目录内工作，完成稳定性检查后同盘归档到`data/references/bones_seed/archives/g1.tar.gz`，不复制一份23.5GB原包或全量解压。

## 处理方案与证据

复用已发布的P01 train1024/dev128/test128候选、CSV V2协议（120Hz、厘米/度、完整29关节列）、A01归档安全扫描及独立22门验收。新增入口显式使用M01迁移视图核旧路径，不改旧收据。原包完整SHA、gzip CRC、成员索引和冻结名单按实际结果登记，再选择原名单内有限train小批重定向/验收；不将原包到位或metadata秒数视为Q1可用时长。

代码版本、实际命令、接收产物和独立检查结果在执行后补充。原始数据/大型工件仍LOCAL_ONLY，Mac仅保存代码、报告及小收据；不得提交第三方完整CSV、模型或归档。

### 归档与新入口准备

已使用`mv -n`将原根文件同盘移到`data/references/bones_seed/archives/g1.tar.gz`，没有复制。新增`bounded_archive_io.py`对自己的顺序输入用16MiB窗口POSIX_FADV_DONTNEED释放已读页，SHA固定1MiB读取，拒绝无界raw.read；不修改文件内容、不全机drop_caches。新增`process_bones_received.py`依次prepare/intake/batch，用新guard逐阶段单次运行，保持100GiB/12GiB/250GiB、28.75GiB/OOM14和900秒进展线。

prepare核完整本地SHA与有限CSV命名空间预览，intake复用原安全扫描/CRC及1280冻结来源，只提取名单；batch默认仅8个原train来源，保持120至50Hz原速和原22门。三个旧模块只在进程内读取M01路径视图，原源码/收据不改；旧下载脚本不执行。本地两项控制通过，包含流式哈希/显式长度、gzip/tar提取和CRC损坏拒绝。服务器部署和真实执行待。

计划命令（实际阶段结束后另补结果）：

```bash
ROOT=/opt/data/private/ZHY
PY=$ROOT/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -u "$ROOT/script/supermimic_q1_sources/process_bones_received.py" prepare \
  --root "$ROOT" --archive "$ROOT/data/references/bones_seed/archives/g1.tar.gz" \
  --output "$ROOT/data/supermimic_for_q1/BONES_A02_received_20261025" --expected-oom-kill 14
```

后续intake/batch仅改phase，复用同一已准备根，各有独立guard与log，不覆盖已有阶段；intake完整性通过前不启动batch。

### 11:48真实原包哈希与协议预览完成

源码发布`2d162366bbad2b218c72d7e2c0c6a970f79b6f41`已推送并独立核Q1，部署三文件SHA完全一致。首次服务器发现发生在复制完成前，实际0 tests，不计通过；复制完成并核SHA后真实两项控制全部通过。归档后inode99228145/大小23499973647/mtime保持，没有新增原包副本。

11:47:21至11:48:00 prepare原生退出0/guard完成回收，完整SHA为`52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a`。实际目录从`g1/csv/230509/`开始，首个CSV为`vertical_mountain_climber_R_003__A359_M.csv`；36列表头与V2一致，member-prefix为空。仅预览，不把该未选片算训练动作或运动学验证。

[原样准备收据](../data/manifests/supermimic_q1_BONES_A02_prepared_20261025.json)。输出根`data/supermimic_for_q1/BONES_A02_received_20261025`，guard_prepare.json与logs/BONES_A02_received_20261025_prepare.log为恢复证据。结束共享内存约20.97GB/OOM14、盘余约2.095TB；新实验约0.42MB。完整gzip CRC、目录扫描及候选提取仍待，不宣称双端传输SHA一致。

### 11:58整包完整性与冻结候选接入完成

11:50:41至11:58:45，真实intake原生退出0，guard为WORKER_COMPLETED且worker已回收。完整SHA与prepare一致，gzip读到EOF并验证CRC；142346个归档条目含142220个常规CSV，元数据catalog142220条全部存在、缺失0、额外CSV0。声明全量展开51527587825字节，但只提取原P01冻结1280个非镜像候选，train1024/development128/test128保持，实际CSV402815302字节。没有全量解压或新增原包副本，源inode/大小/mtime检查保持。

[原样intake收据](../data/manifests/supermimic_q1_BONES_A02_intake_20261025.json)及[终态guard](../data/manifests/supermimic_q1_BONES_A02_intake_guard_20261025.json)已取回，分别核对服务器与Mac SHA为fead7230a3bc17fd786c26b3e67f36d949e61ab6ca883f4e110325a7088ec65f、d211437f4c6a7a0c5b8808ea964941c758d3864e7873961a95e30e4d27a7e0b8，完全一致。原生成索引27798997字节/SHA098c16fd01c2d9dbf12b9838dd71c7461436cf1fa7cf5ef6d4df18cf89880d6d保留服务器，不下载Mac。

使用已发布2d162366代码，实际命令为前述prepare改为intake，其余参数相同。结束整个A02实验唯一分配435138560字节，盘余2094464159744字节，约2.094TB；容器内存约21.38GB、OOM14不变。主机boot仍d994e71f-bae0-46ba-a5f7-6c2163771c6c，原日志logs/BONES_A02_received_20261025_intake.log，原生收据intake/receipt.json、guard_intake.json。原包与提取工件仍LOCAL_ONLY。

本阶段只证明本地归档完整性、目录覆盖及按名单提取，没有源端SHA证明传输两端一致。每个CSV真实帧轴、数值重复、Q1转换/接触和独立22门仍待。下一仅转换冻结名单首8个train来源，不使用held-out筛方案，也不因文件到位启动大规模训练。

## 下一步

原包与冻结提取已完成，下一执行batch --count 8及真实独立22门，依据实际失败项决定转换改进或扩量。保持[DEV-B/P15R结论](README_SUPERMIMIC_Q1_RELOCATED_DIAGNOSTICS_20261025.md)，不恢复旧训练或将新来源自动计为成功。
