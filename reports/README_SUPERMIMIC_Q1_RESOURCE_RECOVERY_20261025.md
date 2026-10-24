# Q1 R02失败后资源恢复

记录日期：2026-10-25。阶段：RECOVERY-A。状态：精确回收完成，模型审计/开发评测待执行。

## 目标与范围

用户将BONES获取交给师兄，继续使用现有PHUMA、AMASS与SMPL-X。恢复huanyu连接，核实PHUMA R02原生失败状态，处理已确认的本项目旧评测残留，为5000轮保存审计及独立开发动力学评测恢复资源；不重启原FAILED目录、不恢复旧A3/STOPPED。

## 完成内容与证据

02:40恢复Windows跳板到huanyu的18864转发。默认认证曾在密码后Broken pipe，显式password-only成功；仅记录观察，未认定认证故障根因。凭据未写文件。boot仍`d994e71f-bae0-46ba-a5f7-6c2163771c6c`。原Q1 guard为Z，controller/trainer不存在；controller FAILED/train、trainer退出1和guard WORKER_FAILED/已回收原样核实。02:08:57第5300轮触发33833570304字节内存保护；5000轮保存待独立张量审计，无成功final及配对结果。

02:46:02至02:46:05实读六个PPID1旧G1/SONIC评测：399565/3587246244、400254/3587250492、400945/3587254861、401666/3587259231、439955/3588563843、452846/3588655310（PID/start_ticks）。工作目录均`/opt/data/private/yuanqi_ops/data/g1_leg/run`，命令均`eval_smoke.py`及`+run_once=true`。四个onset日志于10月21日以`KeyError: 'fps'`结束，两个PHUMA日志以`AssertionError`结束；3秒间隔CPU ticks/I/O/日志大小均无变化，与02:40检查身份相同。六个RSS各约3GB，不能直接相加当作独占内存。

02:40容器内存33964470272/34359738368字节，OOM14，GPU39摄氏度/6348MiB/利用率0。磁盘可用1807355035648字节，较此前减少约158GB但归因尚未完成，不能称本次训练产生。尚无删除文件、清全机缓存或信号。

## 关键决定、版本与命令

恢复前Q1发布`bc89746b033867720048ef92add6514eb2198782`。新只读/精确回收入口`script/supermimic_q1_sources/recover_legacy_g1_resources.py`不在469冻结文件中。默认只检查；先保存原样观察，再用`--terminate-from`绑定相同脚本、boot、PID/start_ticks、日志SHA和连续无进展证据，通过pidfd仅回收这六个残留，先TERM、10秒未退才KILL。文件与历史日志保留。新机器启动或身份变化须重新分析，不重用本入口信号名单。

```bash
python3 script/supermimic_q1_sources/recover_legacy_g1_resources.py \
  --output data/supermimic_for_q1/RESOURCE_RECOVERY_A_20261025/observation.json
```

huanyu根`/opt/data/private/YuanQi_Q1`，CPU检查默认系统python3，内存路径为cgroup v1的`/sys/fs/cgroup/memory`；此前按v2读取路径失败，不是内存信息消失。正式观察/回收输出尚未生成，下一节将补实测路径、SHA及释放量。

## 结果、边界与下一步

已有证据支持六个本项目失败评测残留可以精确回收，但不能单凭PPID1、RSS或GPU空闲认定其他作业可终止，也不能据此断言它们独自导致R02内存失败。下一先保存本阶段、取回原始终态和日志，执行只读观察，再精确回收并重测实际内存/显存；随后核469冻结/5000保存/Adam，在独立目录评测step0与5000的六段development。test不用来选方向。资源允许再完成P07来源审计与PHUMA扩库。

## 保存与同步

README、索引和当前状态已在本地保存，代码及原样小证据随本阶段审计后提交、显式推送Q1并独立核远端；推送结果以Git历史和实际工具结果为准。大工件仍服务器LOCAL_ONLY，无新增Mac大资产。BONES不再获取。

## 02:48原样观察与失败记录接收

正式只读入口退出0，六个`identity_matches/no_progress`均为true，0信号/0文件删除。服务器观察`data/supermimic_for_q1/RESOURCE_RECOVERY_A_20261025/observation.json`已原样接收为[小收据](../data/manifests/supermimic_q1_resource_observation_A_20261025.json)，SHA`7ef4b9ac6c4c651ef7a45d820806251e8a27e3c8e276aa6d25cf918cb965f817`，两端一致。脚本语法检查通过，实际源码SHA`9483ebd2…`。

R02三个原生终态分别原样保存为[controller](../data/manifests/supermimic_q1_R02_failed_controller_20261025.json)、[guard](../data/manifests/supermimic_q1_R02_failed_guard_20261025.json)、[trainer](../data/manifests/supermimic_q1_R02_failed_trainer_20261025.json)，SHA依次`219e6cc1…`、`137d6e0d…`、`9a3de402…`；[原训练日志](../logs/session_records/supermimic_q1_R02_failed_train_20261025.log)284839字节/SHA`ac085172…`，5份原件两端SHA一致，替代此前仅有衍生故障观察的证据缺口。

原guard终态明确R02分配仅1095950336字节，结束时盘余1965468422144字节；02:48观察盘余1765965627392字节，外部磁盘写入仍在发生，不能将约200GB下降归到这个约1.10GB实验。内存33966661632字节，OOM14，尚待精准回收。

## 02:51精确回收完成

前阶段发布`50f529140792882bc60f546295b90c4afd69b9b1`已显式推送Q1、独立远端ref一致。按原观察收据再次核验六个进程与日志身份，pidfd逐个TERM后等待10秒仍未退，向相同pidfd发KILL。02:51实际退出0，六个均为Z、RSS资源已释放；PID1尚未收尸不等于仍运行。0文件删除、0全局缓存清理，未操作其他任务，原R02失败目录及日志不变。

[原样回收收据](../data/manifests/supermimic_q1_resource_cleanup_A_20261025.json)服务器位置为`RESOURCE_RECOVERY_A_20261025/cleanup.json`，预信号快照同目录`cleanup_before_signal.json`。容器内存33968070656→12764688384字节，实际下降21203382272字节（约19.75GiB）；RSS15424708608→571363328、shmem6203080704→12021760字节。GPU6348→14MiB/39摄氏度，OOM保持14，磁盘余1752820707328字节。此为回收前后实测，不能把差值全当单个权重或R02自有内存。

下一只读检查第5000轮保存、469冻结输入与三优化器，再进行六段development配对评测。原5300失败计数不能冒充5000权重轮数，剩余原目标预算若续训应为7000轮并新登记。
