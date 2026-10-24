# BONES新主线磁盘与训练进程保护

记录日期：2026-10-24，北京时间。阶段：SM-Q1-BONES-P00-RUNTIME。状态：保护入口实现、两端8项控制及真实预算非GPU探针完成；新训练尚未注册或启动。

## 目标与范围

落实[BONES目标](README_SUPERMIMIC_Q1_BONES_GOAL_20261024.md)的有限存储与可停止执行。新入口只监管自己启动的进程组，不改变旧实验、旧源码或历史终态。

## 实现与决定

`script/supermimic_q1_sources/runtime_guard.py`从冻结目标配置读取三条预算：数据盘至少100GiB空闲、单实验最多12GiB、新来源区最多250GiB。按device/inode统计实际分配字节，硬链接只算一次，软链接不跟随目标；每5秒及worker结束重算。原包保留，Mac不存大型数据。

状态文件独占创建，日志拒绝覆盖，无自动恢复。worker使用独立进程组；控制器收到停止、预算异常、普通异常或状态I/O失败时，先SIGTERM/最多45秒等待，必要时SIGKILL，并wait直接worker，之后才尝试保存终态。训练可指定progress-file，900秒无更新即停止。正常退出也关闭所拥有进程组的残余成员。

这是新命令的保护入口，只有明确经此入口启动才生效；不是给既有训练追加了保护。温度/OOM等原trainer保护仍须在新BONES入口独立绑定，未以本次无GPU探针证明64环境训练适配。

## 实际验证

本机8项控制全部通过，1.46秒；huanyu同版8项全部通过，1.535秒、原生退出0。覆盖硬链接去重与外部软链接、空间保留和源/实验限额、启动前拒绝、worker运行中预算异常回收、状态I/O异常回收、SIGTERM停止及wait、无进度停止、非零退出与单次运行边界。三个部署文件SHA与本机一致。

20:40:36以真实100GiB/12GiB/250GiB配置执行一个只打印JSON的Python子进程。guard为WORKER_COMPLETED，子进程原生退出0且已回收；无物理步、无优化器。来源区唯一实际分配209379328字节，探针目录结束检查8192字节，数据盘余1965380427776字节。主机同boot、RTX3090空闲17788MiB/39摄氏度、内存26966880256字节、OOM14。

收据：[supermimic_q1_bones_runtime_20261024.json](../data/manifests/supermimic_q1_bones_runtime_20261024.json)。源代码SHA、目标配置SHA、全部探针状态与控制输出均在收据。服务器详细日志`logs/BONES_P00_runtime_controls_20261024.log`、`logs/BONES_P00_runtime_probe_20261024.log`，原状态`data/supermimic_for_q1/BONES_P00_runtime_probe_20261024/guard_status.json`。

## 版本与复现

阶段前已发布并独立核验Q1提交`050a991ec2ab04767a6be4de46d761e301f121b7`；新保护与报告属于本次后续提交，不修改该历史发布。控制命令：

```bash
/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -m unittest discover \
  -s script/supermimic_q1_sources -p test_runtime_guard.py -v
```

新worker调用模式为`runtime_guard.py --goal-config <冻结JSON> --experiment-root <已登记目录> --source-root data/references/bones_seed --workdir <Q1根> --status <全新状态> --log <全新日志> --progress-file <训练status.json> -- <Python> <新BONES训练入口> ...`。这只是后续调用模式，BONES训练入口与输入还未注册，不能把示意命令称为已训练。独立探针目录/状态不复用为训练账本。

## 结果与下一步

保护逻辑与直接worker回收已验证；不是训练效果证据。用户确认完整BONES包仍在D:\tmp下载，下载时刻以用户完成通知/实际稳定文件为准。P01候选已固定，接下来P02接收原包、检查真实CSV格式/时间轴/匹配模型，再P03重定向与独立准入。大文件LOCAL_ONLY，未删除数据或权重。

README、当前状态和索引同次保存；Q1提交/推送及独立核远端，确切提交由Git历史/同步收据提供，推送失败不得报告已同步。
