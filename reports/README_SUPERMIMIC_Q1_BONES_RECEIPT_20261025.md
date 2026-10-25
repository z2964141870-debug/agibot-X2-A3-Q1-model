# BONES-SEED真实G1原包接收

记录日期：2026-10-25，北京时间。阶段：SM-Q1-BONES-A02。状态：原包发现，接收及原速小批处理准备中；尚无新BONES合格参考或训练。

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

## 下一步

归档原文件，完整哈希和成员前缀预检，通过后流式接入冻结候选；真实小批独立验证通过后才扩大。保持[DEV-B/P15R结论](README_SUPERMIMIC_Q1_RELOCATED_DIAGNOSTICS_20261025.md)，不恢复旧训练或将新来源自动计为成功。
