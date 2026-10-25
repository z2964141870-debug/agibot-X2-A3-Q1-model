# Q1项目与private目录存储盘点

记录日期：2026-10-25，北京时间。阶段：Q1-STORAGE-I01。状态：只读盘点完成；共享盘仍无可用空间，正式训练和动力学评测继续阻塞。

## 目标与范围

用户要求查明项目总占用、private下有哪些目录，以及平级和嵌套工作目录是否重复。本阶段只统计huanyu容器可见路径、文件元数据和少量重复文件的完整SHA，不删除、移动、安装或启动训练，不继续BONES获取。

北京时间09:34:35至09:42:12现场采集。所有GB使用十进制10^9字节；目录数值使用du -x -B1的实际分配量，不跟随软链接。单次du对硬链接去重；三个相关工作树的常规文件遍历均无错误、各树内没有重复inode。本记录不是原子文件系统快照。

## private一级目录

| 路径，相对/opt/data/private | 实际占用GB | 说明 |
| --- | ---: | --- |
| hechengzhu | 380.27 | 其中历史A3工作区ZHY为94.38GB，其余285.89GB含其他项目、软件和环境，不归入当前Q1 |
| yuanqi_ops | 366.02 | AMASS、历史A3产物、FGP/运行环境等资源 |
| action-reproduction-20261011 | 41.15 | 仅记录目录大小，未确认与当前任务的归属关系 |
| epic-contact-reproduction-20261011 | 24.01 | 仅记录目录大小，未确认与当前任务的归属关系 |
| YuanQi_Q1 | 19.97 | 当前Q1工作区，含旧Q1试验及本轮PHUMA记录 |
| 合计 | 831.41 | 831413243904字节，前后两次统计一致 |

hechengzhu进一步可见：lijianxing约182.61GB、ZHY94.38GB、Proj55.38GB、hxr26.33GB、Forces14.75GB、Applications3.98GB，以及其他小目录。完整二级元数据保存在执行日志；未读取这些其他项目的业务文件内容。

与本项目有关的三个实体工作区为：

- /opt/data/private/YuanQi_Q1：19969957888字节。
- /opt/data/private/yuanqi_ops：366016385024字节。
- /opt/data/private/hechengzhu/ZHY：94377095168字节。

三者合计480363438080字节，即480.36GB。此前约20GB的说明只覆盖Q1目录，不能代表历史A3、共用动作原料和环境的总占用；本阶段补齐这个统计范围。private其余可见内容约351.05GB，不能全部当作本任务资产。

## 相关目录主要内容

| 路径，相对/opt/data/private | 实际占用GB | 说明 |
| --- | ---: | --- |
| yuanqi_ops/script/amass_bridge/data/amass_raw | 180.26 | 现有AMASS原库，当前Q1直接使用此来源路径；Q1未另存整库 |
| yuanqi_ops/data/training | 106.73 | 历史训练工件，其中p2_main为69.89GB；不是本轮Q1新增 |
| yuanqi_ops/data/humanplus_preview | 37.66 | 91个HDF5等历史预览/标注资源 |
| yuanqi_ops/envs | 9.48 | 主要FGP环境；当前Q1依赖需保留 |
| yuanqi_ops/data/supermimic_for_a3 | 8.02 | 旧A3 SuperMimic试验 |
| yuanqi_ops/.git | 6.52 | 现有Git对象，count-objects约3.42GiB松散对象及2.65GiB pack，未执行gc |
| yuanqi_ops/data/phuma | 3.42 | PHUMA原包及来源入口 |
| hechengzhu/ZHY/data/env | 23.68 | 旧Isaac运行环境 |
| hechengzhu/ZHY/data/environments | 22.99 | 其中isaac-runtime-raw为22.90GB，另有小入口和软链接 |
| hechengzhu/ZHY/data/experiments | 32.78 | 历史A3实验与复制checkpoint |
| hechengzhu/ZHY/data/training | 13.65 | 历史A3训练 |
| YuanQi_Q1/data/supermimic_for_q1 | 18.83 | 当前和历史Q1训练、候选转换、评测和小型证据 |
| YuanQi_Q1/data/references | 1.12 | Q1资产、SMPL-X、来源代码和其他输入 |

表中条目按各自根统计，不能和上一级总量重复相加。Q1完整目录内85个.pt文件共16.56GB；R03目录5.34GB、R02 cachefixed配对3.01GB、本轮R02D2.465GB。R02D退出权重约205MB已包含在2.465GB中。

## 是否存在重复

三处工作区是不同inode的真实目录，Q1位于private一级，历史A3位于hechengzhu/ZHY，另有平级yuanqi_ops。这证实存在分散的项目和资源位置，但不等于三个目录的全部内容都是重复。

真实重复已经确认：

1. 旧E53的arm_i_tier32/model_step_000032.pt在yuanqi_ops训练输出和ZHY实验checkpoints各有一份，308187151字节，完整SHA均为5c2e9fa8ba2f94261d6c80efd94a191f44b3b1cab3015f59cd8a246bcdce8cff。
2. 旧E52的arm_c/model_step_000008.pt在两处各有一份，307570959字节，完整SHA均为b9f2ce63e4bd797876b9441558dfcef842723b06fa2c72857af5b3aba02efbe0。
3. 两组环境库libcudnn_engines_precompiled.so.9和libcusolverMg.so.11，各在FGP、ZHY env和ZHY raw环境有三份独立文件，完整SHA分别一致。

两组checkpoint哈希前后dev/inode/大小/mtime/ctime保持、nlink均1，确实各自占块。精确路径、大小、inode和SHA在日志及收据中。环境库的重复不自动表示可以拆删，当前依赖关系没有完成清理审计。

64MiB以上文件按同名同大小筛出32组跨相关工作树候选，只有上述抽样已内容核验；不能把其余候选、整套环境或107GB历史训练全部计作已确认可释放。没有发现Q1中另存完整AMASS原库的情况。SMPL-X入口、ZHY的isaac-runtime入口及许多参考库使用软链接，不是额外复制原件。

## 为什么突然没有空间

09:34:35整块设备已用7534695383040字节，09:42:12为7562420568064字节，期间增加27725185024字节，即27.73GB；两次private总量均为831413243904字节。09:42可用空间仍0。

findmnt确认private只是/dev/nvme2n1p1的一个绑定子目录，其宿主路径为/default/ZONES/p3090-8/PROJECTS/robotics01/PRIVATE/hcz。df显示的是整块7.62TB文件系统，不是给本项目独占的private目录额度。当前整盘已用约7.56TB，与private可见831.41GB差约6.73TB；差额不能在当前容器内归属到具体目录或用户，也不能全部当作本任务写入。

可见的projectcommon、zonecommon、common/default分别仅4096、4096、12288字节，不能解释差额。容器可见的已删除仍打开文件探针返回0分配字节，但有16项/proc权限失败，且不能看到其他容器全部进程；这不是宿主机完整排除证据。used/available采用statvfs/df口径，保留空间不视为本任务可用额度。

现有证据支持：近期整盘增长主要没有体现为当前private可见目录增长，需平台侧检查同一设备的其他项目/用户目录、不可见进程及文件系统状态。尚未定位具体写入者或某个突然增大的目录，不把这一推断写成已找到责任来源。

## 版本、命令与证据

盘点开始本地Q1基线dabec5919926e58e47ecc2ecc68569bce47fc38e；保存前fetch并快进到e3135df0d9e1d63f772ca6efd1f8931491bbb920，保留09:21每日研究记录。此阶段没有部署服务器代码或修改训练终态。

只读复核示例，在huanyu执行：

~~~bash
df -B1 -T /opt/data/private
find /opt/data/private -mindepth 1 -maxdepth 1 -printf '%y\t%p\t%l\n'
findmnt -J -R /opt/data/private
timeout 240s ionice -c 3 nice -n 15 du -x -B1 --max-depth=1 /opt/data/private
timeout 180s ionice -c 3 nice -n 15 du -x -B1 --max-depth=2 /opt/data/private/yuanqi_ops /opt/data/private/hechengzhu /opt/data/private/YuanQi_Q1
~~~

完整内存探针及原生stdout保存在[执行日志](../logs/session_records/q1_private_storage_inventory_20261025.log)。[汇总收据](../data/manifests/q1_private_storage_inventory_20261025.json)为Mac按原生输出整理的JSON，明确不是服务器生成文件。实际完成两次du总量比较、三个相关工作树文件元数据遍历、挂载核验、有限SHA抽样；无模型反序列化、IK、物理积分或optimizer更新。

## 保存与下一步

本轮服务器新增/删除/移动文件均0。AMASS、PHUMA、SMPL-X、官方基线、当前Q1依赖及R02D退出权重保持；不恢复任何旧停止实验。

最值得进一步做清理审计的是历史A3 checkpoint及重复的旧运行环境，先按既有终态、输入依赖、SHA/硬链接/进程FD检查规范确定保留项；本轮没有把目录大小当成可删空间。整盘持续增长的来源仍须平台宿主机侧核查。

本阶段README、日志、汇总、索引和当前状态按Q1审查/payload audit后提交推送，独立核远端，再同步用户恢复镜像；同次提交号从Git历史查询。科学目标保持存储阻塞，恢复100GiB可用空间后才进行新P15足地修正和R02D退出权重的独立动力学配对评测。
