# CS02：25386入口恢复与证据取回

记录日期：2026-10-27，北京时间。状态：连接、必要环境与小型诊断证据恢复完成；原生接触/候选契约和正式训练未完成。本阶段0物理积分、0optimizer、无子agent。

## 目标与范围

用户提供新入口`root@59.77.15.15:25386`，沿用Windows`yu@100.65.183.91`桥接。恢复项目访问，核文件和真实资源，取回停服前未交付的小结果。唯一根保持`/opt/data/private/ZHY`，不复制大数据或创建平级根，不重启旧容器/实验。MG05工具本次读取仍为paused，原四项目标未完成；本次没有创建、恢复或结算目标工具，也没有启动训练。

## 实际恢复结果

- 新容器`interactive40173`，boot`4d859008-4c83-4a6d-814d-cbb9dfe885c9`。与CS01不同boot和GPU，不沿用旧PID/OOM计数或声称完全相同执行环境。
- 挂载仍为`/dev/nvme2n1p1[/default/ZONES/p3090-8/PROJECTS/robotics01/PRIVATE/zhy]`，原ZHY/环境/runtime/权重/参考均可见。RF09最终权重和300/30参考清单SHA与CS01一致；330条参考路径缺失0。未重扫/重哈希全量历史数据。
- Python3.11.15、Torch2.9.1+cu128/CUDA12.8、MuJoCo及MuJoCo Warp3.11.0、Warp1.16.0、unilab1.5.0。实际通过导入、CUDA小张量有限性、Q1 XML加载、Q1Kinematics构造和RF09 checkpoint的`weights_only=True`加载。模型nq29/nv28/nu22。没有创建4096环境或重做profile。
- 17:29后采样GPU为RTX3090/24576MiB，UUID`GPU-26028785-9fb9-45af-e614-4cc53af15d08`；检查后15MiB/26度/0%利用率。进程扫描未见Q1训练命令。
- private可用`1651439865856`字节，约1.65TB。新容器cgroup v1上限`34292629504`字节，初始用量`54272000`字节、OOM kill0。以上是本阶段采样，后续正式试验仍须重新核验；保留原100GiB余盘、12GiB单实验、30GiB共享内存保护。

小收据：[CS02](../data/manifests/q1_container_restore_CS02_20261027.json)。权重208865366字节，SHA仍为`d834ac94a9ce9d8eca91fc576f4ffb92f0ad54386c40f39539cf0c352debf6d0`，不拉Mac。

## 取回的证据与勘误

已从服务器取回并核大小/SHA：

```text
Mac:
/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/TRACKING_BD02_PHYSICS_V02_20261027/result.json
853171 bytes, a6b5f5b869af4432efec43771979c8e673ce60faa339b3f369afe9d60a39c881
/Users/yu/Documents/ChatGPT/元启/logs/TRACKING_BD02_BACKEND_INSPECT_20261027.log
10739 bytes, 74c03dadd283afb3bca9d514b46bc2d895bfa998991211c1409325370abf8331
```

独立V02结果含4条接触片和12状态H1/H4/H8方向摘要，原探针10448控制积分步；这不是本阶段新积分。原scope中的FP32描述仍以[此前勘误](README_SUPERMIMIC_Q1_MG05_SIGNAL_ANALYSIS_20261027.md)为准，不能将状态FP32等同网络纯FP32。该探针未复现原生RF09轨迹，不作为原摔倒归因或新训练效果证据。

**原生检查日志纠正：** 已取回原件第67至69行实际有`BACKEND`、完整`ATTRS`和`CONFIG {}`，因此此前“prepare阶段失败且未获得对象清单”的说法不成立。139退出的确切发生阶段/原因没有从该日志确认，日志没有原生足地力结果。保留失败记录，不据此宣称原生接触验证通过，也不重跑旧命令。

## 连接与版本

当前实际可用命令：

```sh
ssh -S /tmp/yuanqi-q1-25386.sock -T -p 18873 \
  -o BatchMode=yes -o ConnectTimeout=8 \
  -o 'HostKeyAlias=[59.77.15.15]:25386' root@127.0.0.1
```

Windows转发18873到新25386，桥控制socket`/tmp/yuanqi-windows-25386.sock`。密码仅在内存中交互使用，未写入文件、命令参数或Git。最初交互认证发生断开，改为及时自动交互后实际认证与命令完成；断开不解释为密码无效或容器停服。旧25245/25252及师兄入口本阶段均未探测。

权威Git`/tmp/yuanqi-supermimic-q1-20261023`，Q1分支；本阶段前提交`e92ae1cc1fad77b698698623dd61406843cb85a5`。RF09冻结训练`a553b6a5a3c44167042355ff2f2fbd42ff558342`、HP固定`310244b73e45775744a75b789eb25a6c9c5a81c9`，不改旧runtime/权重/参考。

## 保存、边界与下一步

本记录、小收据、AGENTS和入口说明按payload audit、push Q1、独立远端40位SHA核验发布，小文件镜像Mac和25386并核大小/SHA。发布身份由Git历史提供；模型/大轨迹仍留服务器，不宣称完成网盘备份。停服期准备的候选只同步小源码，生成/张量/原生契约仍需独立验证，不视作部署后的性能改善。

下一步沿[MG05](README_SUPERMIMIC_Q1_MG05_DIAGNOSE_TRACKING_20261027.md)先核同原轨迹/时序的真实足地受力与滑移，再验证候选观测、reward、采样以及replay/imagine契约和H32方向。完成诊断与方案选择后，才进入此前授权的唯一4096环境、最多3600秒独立试验；不续训RF09、不自动追加轮次或五小时预算。本阶段训练未启动。
