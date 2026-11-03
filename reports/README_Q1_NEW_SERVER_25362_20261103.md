# Q1 新服务器 25362 连接与保留检查

记录日期：2026-11-03，北京时间。状态：新原生SSH连接成功，驱动查询正常，项目数据与Python环境可见；尚未做CUDA执行验证或重启训练。[结果清单](../data/manifests/q1_new_server_25362_20261103.json)。

## 已完成

用户提供的新入口为 `root@59.77.15.15:25362`，经Windows `hp_han` 内原生SSH登录，容器为 `interactive67462`，没有本地端口转发。新端口ED25519指纹与既有可信记录完全一致后显式接受，连接的known_hosts目标为NUL；密码仅用于交互认证，没有写入项目、Git或镜像。

14:46:36的只读查询确认这是另一张RTX3090，UUID `GPU-c602a45a-0dd8-9ad2-9adc-4f863facd1d4`、PCI `00000000:16:00.0`，boot `90395f47-4035-4785-858e-b3d05c44a738`，driver595.112.07。recovery_action=None，利用率0%、23C、used1MiB/free24126MiB、无可见计算进程。该结果证明驱动当前可查询且未要求恢复，不替代CUDA kernel与训练容量验证。

唯一根 `/opt/data/private/ZHY` 及data/logs/script/reports均保留。原环境Python3.11.18可以运行，包元数据为Torch2.11.0+cu128、Warp1.17.0、MuJoCo3.11.0；没有导入Torch建立CUDAcontext或安装环境。新容器memory.limit=34292629504B，当前OOMkill0。

G1配对559/559文件存在、330362332B；Q1配对559/559存在、282149476B；human50递归559文件、122602178B。两原配对YAML和旧数据清单仍在。此处只核文件存在和stat大小，不声称重新逐NPZ校验内容SHA或模型权重。

初版stat把相对clip路径误放YAML父目录，产生错误的0/559；按各YAML声明root修正后均559/559，无数据移动或重建。初版和修正原始记录均保留。服务器是文件镜像，git rev-parse报非仓库；权威Git仍是Mac `data/repositories/q1-rf12-20261028`，没有擅自初始化或修复服务器Git。

## 连接方式与设备身份

```sh
source /Users/yu/.zshrc
ssh -tt hp_han
```

Windows内：

```powershell
ssh -p 25362 root@59.77.15.15
```

此次新入口覆盖后续Q1默认25386入口。旧25386故障、一次GPU复位Not Supported和旧boot/GPU均为历史；没有探测旧入口、再复位旧卡或重启宿主机/容器。新GPU可用状态不证明旧GPU已恢复。

已有 `script/morphmimic_q1/run_native.py` 支持 `--expected-boot` 和 `--expected-gpu`，未来新阶段应显式传入本次身份，保持原锁、内存、温度、OOM及源保护；不直接沿用旧身份默认值或恢复失败输出根。缓存工具常量仍固定旧身份，本阶段未运行或修改缓存工具及自动任务。

## 下一步与边界

先在新设备做独立、有界CUDA可用性与原生执行检查，再判断新高容量profile和足量训练启动条件。本次只完成连接和保留检查，新物理、推理、optimizer、CUDAcontext及复位均0。旧profile失败状态保留；正式Q1/共享训练尚未启动，不续profile权重、不创建goal/agent，采纳null。

详细原始输出在[连接记录](../logs/session_records/q1_new_server_25362_20261103.json)，旧GPU诊断见[故障报告](README_Q1_GPU_ERROR_20261103.md)。README/索引/当前状态/AGENTS和小记录按显式暂存、payload audit、提交推送Q1、独立40位远端及逐大小SHA镜像；实际结果见 `data/manifests/q1_new_server_25362_publication_20261103.json`，尚未核验的同步不预称完成。大权重/NPZ/BVH保持SERVER_LOCAL_ONLY。
