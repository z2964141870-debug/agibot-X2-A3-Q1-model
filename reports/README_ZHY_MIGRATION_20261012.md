# A3到ZHY迁移

记录日期：2026-10-12。状态：文件迁移与完整SHA校验完成；运行环境未迁移、未验收，不启动训练。

## 目标与范围

用户确认宿主同意使用 `/opt/data/private/hechengzhu/ZHY`，从原3090机器迁移数据，保持目录结构一致。当前迁移源为hp3090的 `/media/yu/FAFF-E9771/YUANQI_A3`；通用YUANQI/Q1整个目录尚不在本次范围。目标按 `data / logs / script / reports` 组织，原机文件保留，不启动训练/评测或搬运系统服务与认证材料。

## 已核验与决定

- 目标目录新建，未覆盖已有hechengzhu工作。目标磁盘约2.5TiB可用，个人配额未验证。
- 已迁移8440个普通文件，共22619100762字节（约21.07GiB），1370个目录、68条软链接。包含vendor实物197875995字节与旧全链路参考108511218字节；vendor自己的Git对象/缓存不迁移。旧绝对软链接映射为目标内部相对链接，文件内容与历史账本不批量改写。
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

## 实际结果与边界

- 2026-10-12北京时间10:13:58–10:28:56完成清单、传输与完整验证（898秒），51批，首批20.876MiB/s，随后主要25–28MiB/s。目标所有8440文件大小/SHA均一致，68链接指向正确且无断链；验证failures为空。
- 官方PT为402050931字节，SHA256 `9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d`，独立目标校验通过。没有导入或反序列化模型。
- 清单 `logs/zhy_migration_20261012/source_manifest.jsonl` 的SHA256为 `2180ac2b4573115c20b6bac96012713e86084c2cca288cafb64b35fe46267282`。完整state/batches/verification与296包版本metadata留服务器；小型回执见[清单](../data/manifests/zhy_migration_20261012.json)。
- 源码快照为main `8baf73aea38ff3abf230bf1d772d79089aec8756`。开始前已快进合入当天学习/研究/脱敏记录，保留并行修改。最终文档更新在完整数据校验后单独同步，以最终Git提交和目标工作树核验为准。
- 不搬运SSH钥匙、认证材料、原机服务或整个旧Git历史。目标Git由源最新提交的本地浅副本建立，远端仍使用指定仓库origin/main，避免旧凭据历史随数据迁移；没有给新机配置GitHub登录凭据。
- 原机环境未复制；其解释器/依赖路径不具备直接可移植保证。已导出296项包版本，仅metadata读取；例如torch2.8.0+cu128、numpy1.26.0、mujoco3.3.7。新机默认Python3.10.12不是原机Python3.11环境，不据此认为Isaac/A3可运行。
- 历史JSON/README内原机绝对路径保留为来源证据，新实验需要独立初始化和路径适配；不将旧running状态视为目标任务已运行，不自动恢复旧账本。

原始工件及迁移详细清单LOCAL_ONLY，服务器副本不是百度网盘备份，原件未删除。Mac只保存小型代码/README/回执。Git只保存本脚本、阶段README、索引/current state、AGENTS与小型回执，按audit_git_payload审计后scoped提交推送main，核验远端；本次完整提交号由Git历史提供，不在文件中循环写入自身提交号。

## 下一步

在新机独立建立兼容环境并处理新实验路径，先核验资源与依赖，再按用户后续指示验证运行。当前只完成文件迁移与校验，不启动训练/仿真/评测，原机自动恢复继续关闭。新机实际额度28 CPU等效/16GiB内存，见[系统盘点](README_REMOTE_HOST_INSPECTION_20261012.md)。
