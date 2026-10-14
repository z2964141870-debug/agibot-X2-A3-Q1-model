# ZHY GPU 占用任务归属

记录日期：2026-10-14，北京时间10:12–10:13。阶段：用户要求查询谁占用GPU。状态：只读定位完成，具体自然人身份未确认。

## 目标与完成内容

通过hp3090到ZHY，只读取nvidia-smi进程列表和目标PID的/proc元数据。PID1183477为当前唯一去重后的GPU计算进程，Linux用户root，工作目录`/opt/data/private/hechengzhu/hxr/taco_multiview`，入口`replay20.py`。执行环境为`/opt/data/private/action-reproduction-20261011/env-chord/bin/python`。

该任务带`--record-video`、`--video-name`、`--headless`及`--enable_cameras`等参数。依据入口名和参数，推测为动作回放/录像任务；没有读取其源码、数据或日志内容，不据此断言具体内部执行流程或是否正常推进。

stdout/stderr指向`/opt/data/private/hechengzhu/hxr/taco_multiview/taco_brush__brush__helmet_20230919_040_render.log`。父PID1，按/proc启动ticks及uptime换算，约北京时间01:45:07启动，10:13已持续约8小时28分钟；开始墙钟时间为近似换算。

## 实际占用与归属边界

GPU0为RTX3090，总利用率45%，总已用3879MiB；该进程显存3822MiB（约3.73GiB），进程RSS5287284736bytes（约4.92GiB）。nvidia-smi返回两条相同GPU/PID记录，按GPU+PID去重为同一个进程，不把显存重复相加。

目标进程memory cgroup与ZHY查询进程相同：这是同一容器中、ZHY项目目录之外的另一项工作。此前导致宿主内核门拦截的已归属OOM发生在另一个容器；两者不能混为一件事。

`root`是共享Linux账号，`hxr`是工作目录名，仅凭这些信息无法确认启动者真实姓名。没有查他人的shell历史、环境变量、认证材料或无关私人文件，没有发送消息或停止任务。

## 版本、命令与证据

读取时ZHY代码版本沿`767e733a89a48c0afe515821be2eedc9155f82d8`。只读命令为：

```bash
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
```

对去重PID读取/proc/status、cmdline的解释器/脚本/参数名、cwd/exe/fd0–2链接、stat启动ticks和cgroup。只保存必要归属信息，未保存完整参数值或环境变量。

原始小记录`logs/zhy_a3_heartbeat/20261014T021316Z_gpu_owner.json`；精简清单`data/manifests/zhy_gpu_occupancy_20261014.json`保存身份、占用、边界和证据SHA。

## 关键决定、保存与下一步

本次仅回答任务归属，未清缓存、终止进程、启动训练、修改实验/保护或恢复自动任务。E51仍按原阻塞入口处理，官方035仍baseline，本查询不是效果进展。

报告、索引/current state和清单按scoped审计推送main；实际发布状态以Git回执为准。原始小记录在Mac及ZHY保存并核对SHA，LOCAL_ONLY，无网盘验证。

若需确认具体负责人，可以向管理该共享容器或`hxr`目录的人核对上述脚本/PID；代理未代发消息。后续操作前重新核验PID/路径和实际占用，不能凭本快照永久认定某进程仍在运行。
