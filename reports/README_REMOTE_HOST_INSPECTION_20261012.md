# 新服务器系统与存储只读盘点

记录日期：2026-10-12，北京时间09:56–09:58。状态：只读盘点完成。

## 目标与完成内容

按用户要求经 `ssh hp3090` 再连接目标 `root@59.77.15.15:25115`，只读核验系统、资源额度、GPU、存储及默认Python。登录凭证不保存。首次连接确认的SSH指纹为 `SHA256:JiB72vZg5qqp85lPhM6Ns2NMh3MldA05ZK729E3l1PI`。

| 项目 | 实测 |
| --- | --- |
| 主机/系统 | interactive60229，Ubuntu22.04.3 LTS，kernel5.4.0-42-generic，Docker/Kubernetes容器 |
| 实际CPU额度 | cgroup quota2800000/period100000，28 CPU等效；不是宿主128逻辑CPU全部可用 |
| 实际内存额度 | 17179869184字节，即16GiB；检查时cgroup计费约3.92GiB，包含缓存等 |
| GPU | 一张RTX3090，24576MiB；driver570.153.02，nvidia-smi兼容CUDA12.8，不证明toolkit安装 |
| GPU空闲状态 | 26°C、33W、0%利用率、13MiB显存，无计算进程 |
| GPU功率上限 | 默认/当前/最大350W，最小100W；仅查询，未修改 |
| 存储 | /opt/data/private为独立ext4读写挂载，/dev/nvme2n1p1；7.0TiB总量、2.5TiB可用 |
| 精确磁盘容量 | total7619864297472 / used4499375910912 / available2736396623872字节 |
| 权限与配额 | root:root0755，权限只读测试通过；账号配额与容器重建持久性未核验 |
| 默认软件 | /usr/bin/python3版本3.10.12；git/curl/wget/scp存在，未发现rsync、默认conda/nvcc |
| 默认Python包 | metadata查询未发现torch/torchvision/numpy/ultralytics/isaacgym/isaaclab/mujoco |

宿主机显示双Xeon Gold6530、128逻辑CPU、251GiB RAM；使用时必须遵守容器28 CPU/16GiB限制。/dev/shm显示126GiB也不能突破内存额度。已有epic-contact、action复现及hechengzhu目录；存在多个隔离venv，未检查其完整依赖，不据此认定服务器所有环境都没有PyTorch。现有他人目录不覆盖。

## 版本、证据与复现

使用uname、/etc/os-release、lscpu、/proc/meminfo、/proc/self/cgroup、cgroup v1额度文件、df/findmnt、nvidia-smi及importlib.metadata查询。未加载模型、未计算CUDA、未安装软件、未改驱动/功率。登录时出现 `/generate_container_user` 缺失提示，但连接和查询成功。

原机报告仓库 `/media/yu/FAFF-E9771/YUANQI_A3` 起始代码HEAD为 `8d47d5b9c0c59a211ea5efcfc2931d1026667327`。只读盘点本身未传输项目工件。

## 结果与下一步

确认登录及目标存储存在；不能据此认为A3运行环境就绪、硬件长期稳定或配额充足。随后用户明确授权在 `/opt/data/private/hechengzhu/ZHY` 建目录迁移原机数据，具体状态见[迁移记录](README_ZHY_MIGRATION_20261012.md)。记录按项目Git规范保存，不含凭证。
