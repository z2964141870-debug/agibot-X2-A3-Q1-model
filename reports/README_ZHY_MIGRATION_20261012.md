# A3到ZHY迁移

记录日期：2026-10-12。状态：进行中，尚未声明完成。

## 目标与范围

用户确认宿主同意使用 `/opt/data/private/hechengzhu/ZHY`，从原3090机器迁移数据，保持目录结构一致。当前迁移源为hp3090的 `/media/yu/FAFF-E9771/YUANQI_A3`；通用YUANQI/Q1整个目录尚不在本次范围。目标按 `data / logs / script / reports` 组织，原机文件保留，不启动训练/评测或搬运系统服务与认证材料。

## 已核验与决定

- 目标目录新建，未覆盖已有hechengzhu工作。目标磁盘约2.5TiB可用，个人配额未验证。
- A3实物文件约21GiB，另物化vendor约328MiB与旧全链路参考约104MiB；旧绝对软链接映射为目标内部相对链接，文件内容与历史账本不批量改写。
- 原 `data/environments/a3-sonic` 依赖原机conda解释器，其真实conda环境约22GiB，不可直接当作可移植环境。暂不迁移该入口；记录包版本，后续环境重建单独核验。
- 原机A3自动恢复/长训服务inactive且disabled；本任务仅文件I/O。新机不注册自动运行服务，不修改驱动/功率/BIOS。
- 目标未提供rsync，使用既有tar/SSH，分批≤512MiB，首批官方PT测速并核对SHA。明显慢速（首批<10MiB/s或批次180秒超时）停传，遵守用户此前大文件规范，不持续强传。

## 可复现命令与工件

脚本：`script/project_tools/migrate_zhy.py`。需要在hp3090已有经交互认证的SSH ControlMaster连接；凭证不落盘。源码清单/批次/状态/校验记录写原机 `logs/zhy_migration_20261012/`，大型文件服务器间直传，Mac只保留小代码/报告/清单。

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
nice -n 15 /usr/bin/python3 -u script/project_tools/migrate_zhy.py \
  --logdir logs/zhy_migration_20261012
```

只在本次新目录迁移上下文使用；不要重新启动已有迁移或覆盖部分目标。恢复前先读state/batches/source_manifest与README，SHA已通过的文件应复用，失败/缺失批次才续传。

## 结果与保存状态

目录创建已完成；完整迁移/所有SHA核验尚未完成。原始工件及迁移详细清单LOCAL_ONLY，服务器副本不是百度网盘备份，不删除原件。Git只保存本脚本、阶段README、索引/current state与小型回执，按audit_git_payload审计后scoped提交推送main，核验远端。

## 下一步

执行官方PT测速与哈希校验，再按实际速度继续或停传。完毕后核对全部文件数/字节/SHA、软链接与Git副本；环境可执行性与历史绝对路径未适配，迁移成功不代表可直接训练。
