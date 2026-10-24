# PHUMA独立扩库与训练资源隔离

记录日期：2026-10-24至25日，北京时间。阶段：SM-Q1-P06。状态：B015/B016/B017B/B018完整覆盖额外名单[0,672)，新增167片/593.92秒；与原库算术507训练片/1821.02秒（30.35分钟），尚未完整合库。B017失败保留、不计其未完成文件；原R02已FAILED，R02C独立续训进行中，新参考不混入本轮固定输入。剩余[672,1024)的352候选待B019。

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

本节为入口发布时的安排；真实执行结果逐批记在下方。原R02继续有限预算及配对评测。新参考必须完整跨批时轴/来源隔离重算后才登记新库，不改变正在训练的R02。失败同样立即登记，保留原目录；数据/数组全部LOCAL_ONLY。

## B015真实完成

发布`0ca9a975905ece75807d29c52eefc89b79423f5e`已显式推送Q1并独立核远端；guard入口SHA`29bd2509cffccad91752e7c8119ab40262261fdba8d6ae098f9f0eb356cdf3ea`、测试SHAa989f93a…两端一致，服务器3项控制通过，固定额外名单SHA631cd06b…一致。上方命令已实际执行，23:30:23至23:31:45，guard原生退出0。

32个固定train候选(offset0/count32)，转换无异常；原19门接纳9片/31.08秒，其中other8、移动1/2.06秒。转换/独立检查计时73.76秒，guard全程82.28秒。仍主要被支撑比例20、30mm间隙16、自碰撞11、转换初门10、残差5拒绝，计数重叠。新片与原库的合格训练时长算术为1258.18秒（20.97分钟），尚未作完整跨部分合库/时轴重算，不宣称30分钟达成。

worker826112独立进程组，guard826110；实际检查时见/proc活跃、8个中间NPZ，采样共享内存30448566272字节，低于28.75GiB上限30870077440。结束后guard明确worker_reaped=true/return_code0，OOM14，内存28714233856字节、批次实际分配19795968字节、盘余1965634822144字节。没有给训练发信号。完成后原trainer825189仍RUNNING并到900轮/57600交互、WM309组/actor259次、GPU40摄氏度/OOM14。

实际产物`data/supermimic_for_q1/PHUMA_P06_B015_20261024/batch/`，stdout`logs/PHUMA_P06_B015_20261024.log`，guard stdout另`logs/PHUMA_P06_B015_20261024_guard.log`。completion为2195字节，SHA`ba4d19cc325f705032a96cad43c0ba4701b5422c56575d6f71005bb4ea47a87d`；[小收据](../data/manifests/supermimic_q1_phuma_P06_B015_20261024.json)及[guard收据](../data/manifests/supermimic_q1_phuma_P06_B015_guard_20261024.json)。原manifest/reference_check的大小/SHA均在completion中，未重做本批几何。

下一独立B016从offset32/count128开始；仍单数据worker、同一来源/门/资源提前停止线，与B015不重叠。若内存保护停止，保留失败目录，不放宽保护线或动原R02。

## B016真实完成

沿用同一已发布guard及固定名单，实际offset32/count128；23:39:15至23:43:40，guard原生退出0。worker826504及guard826502在23:46:44独立检查时均已不存在，guard记录worker_reaped=true/return_code0。原exec会话44417已收集退出0，未重复启动。

128候选中31片通过原19门/110.64秒，其中other29片、移动2片/7.18秒；0转换异常，转换/核验256.60秒。主要重叠拒绝计数为支撑15mm比例79、自碰撞50、支撑间隙43、转换初门22，门槛未调整。加B015及原库，算术训练380片/1368.82秒（22.81分钟），移动65片/231.00秒；尚未做完整合库/跨批保存时轴重算，不宣称30分钟已达。

批次末实际分配79691776字节，盘余1965308944384字节，共享内存28818477056字节/OOM14；运行中采样30655176704字节，低于30870077440字节线。原训练23:46:44仍RUNNING到1275轮/81600交互、WM459组/actor409次、GPU39摄氏度/OOM14；1000轮中间权重205348199字节已存在，尚未做张量检查。

产物`data/supermimic_for_q1/PHUMA_P06_B016_20261024/batch/`，worker日志`logs/PHUMA_P06_B016_20261024.log`、guard stdout为同名`_guard.log`。completion2255字节/SHA`baceb73e71573236920981a08c3f9621506274793596ae57ac055135ae0a2ad7`；guard1940字节/SHA`95e4fcbb0189c7ce201313b6f37d41a3df92cbf3d5e90c3f598be173aeebfa40`。已取回[完成小收据](../data/manifests/supermimic_q1_phuma_P06_B016_20261024.json)与[guard小收据](../data/manifests/supermimic_q1_phuma_P06_B016_guard_20261024.json)。

下一先在无转换worker的间隙执行R02第1000轮CPU只读张量/Adam检查，再以新B017目录处理offset160起的下一切片。不改原R02的469冻结文件或输入，不恢复BONES获取。

## B017内存保护终止

R02第1000轮16项CPU保存检查于23:51完成并已发布01de432，见[训练阶段](README_SUPERMIMIC_Q1_EXISTING_RESOURCES_20261024.md)。随后B017独立offset160/count128，23:54:00启动；23:54:38共享内存30930849792字节超过数据worker线30870077440字节，guard按既定规则只终止本worker组827074，回收return_code=-15。guard827073结束、原exec11186已收集退出1；原输出保留，未覆盖或重启。

本批仅产生7份NPZ及对应诊断，未形成完整manifest、reference_check或completion，0新增合格参考，不计入380片/22.81分钟的已完成批次算术。触线前最近存储检查分配4313088字节、盘余1965633273856字节。OOM始终14，不能把提前保护记成实际OOM；没有向原训练发信号。

23:55:52独立检查数据worker已不存在，原trainer仍RUNNING至1525轮/97600交互、WM559组/actor509次、GPU39摄氏度/OOM14。23:57:17共享内存29021405184字节；rss18726244352、cache9494237184（其中shmem6219571200）。这些是共享cgroup统计，不能全部归因于本数据worker，也不足以认定某一转换函数泄漏。

[原样guard收据](../data/manifests/supermimic_q1_phuma_P06_B017_guard_20261024.json)为2007字节/SHA`b1d2c3a259520eb3c970730881e51acff6ad5f136424f9793d85e69cf66c83ea`。日志`logs/PHUMA_P06_B017_20261024_guard.log`与`logs/PHUMA_P06_B017_20261024.log`，失败目录`data/supermimic_for_q1/PHUMA_P06_B017_20261024/`均保留LOCAL_ONLY。

下一先暂停并行PHUMA转换；不放宽线、不清全机页缓存、不停止原训练，也不以缩小count盲目重试相同入口。待原训练释放资源后，以新目录从原固定offset160重新做完整转换/核验，不能跳过本次未完成的128来源。当前可推进AMASS轻量来源选择/数值协议准备，尚未Q1准入，不改变R02输入。

## 10月25日03:02 B017B恢复

原R02已内存保护FAILED，其三个原终态和训练日志已保存；六个10月21日失败G1残留经精确核验回收，容器内存34→12.76GB，见[资源恢复](README_SUPERMIMIC_Q1_RESOURCE_RECOVERY_20261025.md)。这是资源状态改变后的新独立批次，不在B017失败目录重启，也未缩小切片/放宽门。

03:02:09以同一已发布guard/转换器启动`PHUMA_P06_B017B_20261025`，offset160/count128原固定来源完整重做，CUDA禁用、各BLAS单线程、nice10、28.75GiB/OOM14及100GiB余量保护保持。guard834572/worker834573是本次观测，恢复时须重新读/proc及guard；日志`logs/PHUMA_P06_B017B_20261025.log`，状态`data/supermimic_for_q1/PHUMA_P06_B017B_20261025/guard.json`。首个现场内存13996662784字节，盘余1695072436224字节，无其他训练/转换worker。真实完成收据尚待，不预计新增合格秒数；原469冻结与R02输入不变。

03:05:57真实完成，guard原生退出0、worker_reaped=true/return_code0；128候选中33片合格/123.70秒（移动3/10.58秒），0转换异常，原19门未变。主要重叠拒绝为支撑90、间隙53、自碰撞38。转换/核验220.04秒。末内存13689217024字节/OOM14、批次分配80150528字节、盘余1676117712896字节。原B017失败目录及7个未完成候选保留，未计两次。

[完成小收据](../data/manifests/supermimic_q1_phuma_P06_B017B_20261025.json)和[guard原件](../data/manifests/supermimic_q1_phuma_P06_B017B_guard_20261025.json)已保存。额外已完成B015/B016/B017B合计73片/265.42秒；与原340训练片算术为413片/1492.52秒（24.88分钟），移动68片/241.58秒。尚未完整合库，30分钟目标未达；这些新片不混入继承原340输入的R02C续训。后续固定切片从offset288开始，仍单worker/同门/原速。

## 03:15 B018运行检查点

在B017B完整结束且新R02C初始12门继承审计通过后，03:15:43启动独立`PHUMA_P06_B018_20261025`，仍来自原固定1024来源，offset288/count384，覆盖[288,672)而不重复[0,288)。资源回收后共享余量超过14GB，转换逐片串行，扩大的是一次候选切片长度，原速/19门/资源提前停止线/来源与split不改；不能预先计入合格时长或保证30分钟。

[原样启动收据](../data/manifests/supermimic_q1_phuma_P06_B018_launch_20261025.json)，guard835179/start_ticks3616134874，worker835180；boot与R02C同。03:16实际guard RUNNING、已有输出文件，最新内存18684727296字节/OOM14、盘余1619400486912字节，数据worker独立组且nice10/单线程/无CUDA。状态`data/supermimic_for_q1/PHUMA_P06_B018_20261025/guard.json`，stdout`logs/PHUMA_P06_B018_20261025.log`及`_guard.log`。失败或停止仅回收本数据worker；不向R02C训练发信号、不自动重试。完成后先取completion/guard及检查原进程回收，再完整组库/重算来源隔离；未来切片从offset672起。

## 03:25 B018真实完成

03:25:43 guard记录WORKER_COMPLETED、worker_reaped=true/return_code0；03:26:44独立核验worker835180已不存在，guard835179为Z/RSS0，不再运行。completion绑定的selection/scan_check/retarget_manifest/reference_check四项大小及SHA全同，全部94个合格记录均保留原19门。两份原样小收据已取回并核两端SHA：[completion](../data/manifests/supermimic_q1_phuma_P06_B018_20261025.json)为2276字节/SHA`618ff9fa86ed58e46394e078f947ffd451ef8c410198fc62f5f1f65faa069d1f`，[guard](../data/manifests/supermimic_q1_phuma_P06_B018_guard_20261025.json)为1943字节/SHA`108e841a00f41fdd423fa78cc63fb4f430bd6ca7cd4f5cf1b855075987007bca`。

384候选中94片合格/328.50秒，其中移动13片/44.54秒，其他81片；0转换异常，转换及独立检查592.45秒。主要重叠拒绝为支撑260、最大间隙158、自碰撞143，阈值未调整。数据末内存17163304960字节/OOM14、批次分配235995136字节、盘余1571769532416字节。源端文件、NPZ与IK数组仍LOCAL_ONLY，Mac只接收4219字节完成记录。

新增完整批次合计167片/593.92秒，加基础340片为算术507片/1821.02秒（30.35分钟），其中移动81片/286.12秒。此为跨批算术，完整组库、全时轴重算及新增HumanML别名审计尚未执行，不能称完整30分钟库已验收，也没有新增策略效果证明。03:25:33 R02C仍RUNNING至575新轮/累计5575，WM179组/actor129次，OOM14。

下一以独立B019处理offset672/count352，仍单数据worker、同一固定名单、原19门和28.75GiB提前停止线。已核固定名单SHA`631cd06bd0272e279ef587abead63e8849357d34e234ee3c7dfa74af60703a20`、余352行全部train、目标目录尚不存在、转换器及guard两端SHA一致。先保存发布本阶段，再启动；完成全1024候选后用B005至B014与B015/B016/B017B/B018/B019组成新库，保留原R02C的340训练输入及496冻结。

## 03:30 B019启动检查点

B018记录发布`8a55bad14d01207a79c203e67990a7313e5fdcbf`已推送并独立核远端Q1。03:30:24在核同boot、R02C三个实际start_ticks、无活跃转换worker、代码和名单SHA、内存/OOM/磁盘后，通过Popen独立session启动最后352候选(offset672/count352)，未改已发布转换器或任何训练输入。guard835770/start_ticks3616223022，启动共享内存17162645504字节、盘余1547257995264字节/OOM14。

[启动原样收据](../data/manifests/supermimic_q1_phuma_P06_B019_launch_20261025.json)。独立输出`data/supermimic_for_q1/PHUMA_P06_B019_20261025/`，guard状态为该目录guard.json，日志`logs/PHUMA_P06_B019_20261025.log`及`_guard.log`；stdin关闭，无自动重试。恢复时读真实/proc、guard及completion，不能据旧PID认定仍运行；未生成完成收据前不增加合格计数。

## 移动标签与来源边界

本阶段的`locomotion`继承原扫描代码，只表示源最大水平位移至少0.25m，不是步行语义或真实支撑步态认证。基础test三段的源文件名分别包含`Side_Stretch_Ab_Squeeze_L`、`Side_Reach_To_Butt_Kicks`、`Side_Step_Arm_Swing`，属于fitness命名空间；名称及位移阈值不能证明它们是独立直线走路测试。后续仍按预登记门逐片报告实际根位移/误差和完整率，步行能力另需来源及实际轨迹核验，不把标签或在线奖励当移动策略成功。
