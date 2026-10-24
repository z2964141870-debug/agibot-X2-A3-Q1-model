# PHUMA独立扩库与训练资源隔离

记录日期：2026-10-24，北京时间。阶段：SM-Q1-P06。状态：B015真实32片转换/独立19门完成，9片合格/31.08秒，原生退出0、worker已回收/OOM14；R02继续，900轮。新参考未混入R02，完整扩展库尚未组装。

## 目标与范围

继续现有PHUMA资源，朝30分钟、再1小时Q1合格训练参考推进。R02在已固定340片/1227.10秒库训练，距30分钟还差572.90秒；不修改其469冻结输入、来源、split或原终态。新候选是已于18:35选定的额外1024个不同train来源，512移动/512其他、SHA631cd06bd0272e279ef587abead63e8849357d34e234ee3c7dfa74af60703a20，尚未转换；development/test原名单不变。

用户已将BONES获取交师兄，不继续下载或凭据处理。AMASS原始库与16维SMPL-X到位，但本阶段不混入PHUMA固定实验。

## 关键决定与完成内容

23:20:42重新核实原guard825182/controller825183/trainer825189的实际/proc命令及活跃状态，R02到525轮/33600交互、WM159组/actor109次，GPU40摄氏度/OOM14，cgroup28545449984字节。此为真实进程核验，不凭状态文件盲目重启。现有批处理峰值可能占数GiB，因此先小批32，且额外数据worker先于原训练的内存线停止。

新增`script/supermimic_q1_sources/guard_phuma_batch.py`只包裹已发布`prepare_expansion_batch.py`。独立输出目录保存guard.json及batch/，不修改原转换器、19门、优化器或模型槽后端。nice10、调用时显式单线程/无CUDA；共享cgroup内存上限28.75GiB，比原trainer的30GiB低1.25GiB，每0.25秒在存储核验前后检查内存及OOM计数。触线只终止并回收本worker进程组，不向训练发信号；沿用已验证runtime_guard的异常、信号和清理路径。

磁盘预算仍为100GiB余量、独立批次最多12GiB、原ZIP所在源目录250GiB，软链接不跟随、硬链接按inode去重。额外worker读取原候选固定切片，必须全为train；输出排他创建，失败保留，不在原目录重启。900秒无新参考文件进度停止保护有效。0.25秒检查不能作为零资源扰动保证，因此以小批资源实测决定是否继续，不同时堆多个转换worker。

本地3项控制通过：共享内存等于上限时拒绝启动且不产生worker文件；新增OOM及存储扫描后内存上涨被拒绝；成功时保留存储与内存证据。没有重复已通过的完整物理后端/梯度验证，也未运行新训练或新几何检查。

## 版本、命令与证据

前一发布`1c6735d864532f8e7ad81ff302541405d53129f2`，Q1。新代码、测试和本README审核推送、独立核验后部署，不改变R02注册训练版本b6d3108。

拟首批命令，真实调用前核路径、代码SHA、内存/OOM及输出不存在：

```bash
cd /opt/data/private/YuanQi_Q1
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  script/supermimic_q1_sources/guard_phuma_batch.py \
  --root "$PWD" \
  --selection "$PWD/data/supermimic_for_q1/PHUMA_P04_additional_selection_20261024.json" \
  --vendor "$PWD/data/references/phuma_source_2faf15b5" \
  --output "$PWD/data/supermimic_for_q1/PHUMA_P06_B015_20261024" \
  --offset 0 --count 32 --expected-oom-kill 14 --maximum-memory-gib 28.75
```

本地验证：`python3 -m unittest discover -s script/supermimic_q1_sources -p test_guard_phuma_batch.py -v`。原转换与完整源SHA/输入隔离/19门继续由既有批处理执行；不把本地控制当真实扩库完成。

## 结果、保存与下一步

当前仅完成保护入口，0新Q1参考/optimizer；原R02继续有限预算及配对评测。下一发布部署、小批真实执行，检查本worker的原生退出/guard清理、原训练身份与进度、OOM和合格秒数；通过后按固定名单继续互不重叠的独立批次。新参考必须完整跨批时轴/来源隔离重算后才登记新库，不改变正在训练的R02。失败同样立即登记，保留原目录；数据/数组全部LOCAL_ONLY。

## B015真实完成

发布`0ca9a975905ece75807d29c52eefc89b79423f5e`已显式推送Q1并独立核远端；guard入口SHA`29bd2509cffccad91752e7c8119ab40262261fdba8d6ae098f9f0eb356cdf3ea`、测试SHAa989f93a…两端一致，服务器3项控制通过，固定额外名单SHA631cd06b…一致。上方命令已实际执行，23:30:23至23:31:45，guard原生退出0。

32个固定train候选(offset0/count32)，转换无异常；原19门接纳9片/31.08秒，其中other8、移动1/2.06秒。转换/独立检查计时73.76秒，guard全程82.28秒。仍主要被支撑比例20、30mm间隙16、自碰撞11、转换初门10、残差5拒绝，计数重叠。新片与原库的合格训练时长算术为1258.18秒（20.97分钟），尚未作完整跨部分合库/时轴重算，不宣称30分钟达成。

worker826112独立进程组，guard826110；实际检查时见/proc活跃、8个中间NPZ，采样共享内存30448566272字节，低于28.75GiB上限30870077440。结束后guard明确worker_reaped=true/return_code0，OOM14，内存28714233856字节、批次实际分配19795968字节、盘余1965634822144字节。没有给训练发信号。完成后原trainer825189仍RUNNING并到900轮/57600交互、WM309组/actor259次、GPU40摄氏度/OOM14。

实际产物`data/supermimic_for_q1/PHUMA_P06_B015_20261024/batch/`，stdout`logs/PHUMA_P06_B015_20261024.log`，guard stdout另`logs/PHUMA_P06_B015_20261024_guard.log`。completion为2195字节，SHA`ba4d19cc325f705032a96cad43c0ba4701b5422c56575d6f71005bb4ea47a87d`；[小收据](../data/manifests/supermimic_q1_phuma_P06_B015_20261024.json)及[guard收据](../data/manifests/supermimic_q1_phuma_P06_B015_guard_20261024.json)。原manifest/reference_check的大小/SHA均在completion中，未重做本批几何。

下一独立B016从offset32/count128开始；仍单数据worker、同一来源/门/资源提前停止线，与B015不重叠。若内存保护停止，保留失败目录，不放宽保护线或动原R02。
