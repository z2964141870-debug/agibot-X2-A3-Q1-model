# ZHY 容器内存核验与定向缓存释放

记录：2026-10-13 北京时间22:57–23:01。状态：本次核验与释放完成；E47已REGISTERED，尚未启动worker。

## 目标与范围

用户新增要求自主排查内存，适当释放。只操作ZHY项目已结束实验的文件页缓存，不改16GiB容器额度、原available_estimate记账或8GiB入口/4GiB运行保护。E47登记源码与输入保持不变。

## 实测与判断

- 释放前容器使用9,347,063,808字节（8.705GiB），其中RSS约0.340GiB、文件cache约5.703GiB、kmem约2.659GiB；当前可见进程没有RSS>1GiB的训练残留。宿主约251GiB总内存不是容器可用额度。
- cgroup自身memory.kmem.slabinfo中ext4_inode_cache分配页约1.781GiB、dentry约0.361GiB、ovl_inode约0.192GiB。内核内存主要落在文件系统元数据；本次不能证明系统内存泄漏或归因到某个历史扫描任务。
- 175个僵尸都由PID1收养，无VmRSS；不是175个仍持有训练模型的活进程。本次不重启PID1或基础服务。旧failcnt=99,539,871在检查与释放间未增，oom_kill一直0，同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c；历史failcnt不作本次OOM。
- 只读检查可见FD/maps；pod/ssh-agent/gpg-agent三小型基础进程不可读，实际RSS约1.70/1.06/0.29MiB，覆盖缺口明确保存，未假称全系统全部可见。首次释放脚本在PID1权限检查中拒绝，未调用释放；记录这些有限例外后执行只读文件缓存建议，没有绕过计算遥测保护。

## 实际释放及证据

使用标准库os.posix_fadvise(POSIX_FADV_DONTNEED)，只对E44 step0、E45 step2两个独立inode的checkpoint只读描述符发出建议。先验原实验STOPPED、E47 REGISTERED、目标不是E47登记输入、可见进程未打开或映射。fincore确认两目标驻留从441,622,528字节（421.16MiB）降至0。

容器总使用降至8,893,702,144字节，原可用估计从8,682,958,848升至9,136,320,512字节（约8.087→8.509GiB）。总使用变化含少量同期内核/进程波动，不能全归为两个文件。inode/大小/mtime/ctime未变，文件只读打开，保存释放前SHA；不额外读回大文件以避免重填缓存，因此不声称做了释放后全文件SHA。未删除checkpoint、未终止服务、未执行全宿主drop_caches。

## 版本、命令与保存

E47已登记版本d26d9596235cf3666e18cca4c99bba23c95edd1a；本次小脚本不在其固定计算代码集合内，不改原登记。一次性入口（已有receipt禁止再执行）：
```bash
cd /opt/data/private/hechengzhu/ZHY
/usr/bin/python3 script/project_tools/release_zhy_closed_cache_20261013.py
```
原始小记录：logs/zhy_memory_20261013/inspection_145700.json和cache_release.json；[清单与SHA](../data/manifests/zhy_memory_20261013.json)。原始日志LOCAL_ONLY，未核验网盘；小代码/README/清单按scoped审计提交并核对远端，实际提交见Git。

## 下一步与边界

按原三次5秒门启动已登记E47一次，不重复register。保留当前需要的输入缓存，避免继续盲目清缓存。剩余kmem虽大，本次没有可验证安全的容器专属强制回收手段，不改内核参数或重启共享服务。后续仍以实时容器快照为准，不能拿本次释放后的数值作未来预检。

后续23:07：E47已单次COMPLETE，运行最低可用7.172GiB，同boot/OOM0；见[计算结果](README_A3_ADAM_DIRECTION_20261013.md)。本次释放没有改E47任何保护门。

## 用户后续要求：缓存增长时自动回收（23:24）

已新增标准库维护脚本script/project_tools/zhy_file_cache.py，接入现有zhy-a3工程heartbeat，不另开自启服务。计算结束后与每30分钟巡检，只有GPU空闲、无其他计算/大进程、同boot/OOM、遥测可靠时才执行。总file cache超过4GiB触发检查，每次最多32文件/2GiB驻留页，成功释放后至少30分钟间隔；这些是维护界限，不改原实验8GiB/4GiB资源门。

只考虑有COMPLETE/STOPPED/REJECTED/FAILED账本的项目工件/日志目录。保护未终结实验输入、下一阶段计划输入、可见打开/映射文件；按inode去重，不跟随目录或文件软链接，不扫描环境/宿主目录，不删任何文件、不动进程，不全局drop_caches。仅给只读fd发送DONTNEED；没有写入工件的代码路径。三小基础进程FD/maps可见性缺口仍明确记录。调用前后状态异常则拒绝并记证，不强制清空全部cache。

实际先inspect选出9个闲置文件，然后单次trim释放161,927,168字节（154.43MiB），所有目标fincore驻留降为0，文件inode/大小/mtime/ctime不变。本轮两次定向释放累计603,549,696字节（575.59MiB），不把同期其他内存波动计入。第二次后原可用估计约8.597GiB、OOM0/failcnt未增；仍有约5.17GiB其他file cache，当前不在合格闲置项目目标中，没有声称全部清零。紧接一次维护调用验证COOLDOWN，不再次释放。

使用最新计划/登记作为保护列表与boot/OOM基线，后续阶段更新这两个参数：
```bash
cd /opt/data/private/hechengzhu/ZHY
/usr/bin/python3 -m script.project_tools.zhy_file_cache --trim \
  --protect-plan data/manifests/a3_fresh_batch_plan_20261013.json \
  --baseline data/experiments/a3_zhy_adam_direction_20261013_E47/registration.json
```
不加--trim只检查。每次小receipt保存在logs/zhy_file_cache/，[本次策略/实测SHA清单](../data/manifests/zhy_file_cache_policy_20261013.json)。本脚本不自动启动任何训练或切换实验；E48仍是待实现草案，主线状态按最新README。
