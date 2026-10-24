# PHUMA驱动的SuperMimic for Q1

记录日期：2026-10-24，北京时间。阶段：SM-Q1-PHUMA-P01/P02/P03/R01/D01/D02。状态：PHUMA R01固定3000轮训练、11项保存审计、18片初始/最终完整动力学评测及两次独立重算、三划分比较全部完成，7阶段原生退出0。最终16/18片完整执行，0/18达到预登记关节/根部跟随门，尚非可用全身跟随策略。

最新：14:36:50原流水线完成，14:38:44独立复核7个原生退出0事件/日志身份、121冻结文件、三划分配对算术和原进程组无活跃成员。训练实际3000轮/48000控制转移、97.6787分钟，未追加预算。初始13/18完整、最终16/18；部分误差改善但移动不足，详见末节。[V01图表](README_SUPERMIMIC_Q1_PHUMA_PLOTS_20261024.md)已实际退出0、36组重算及两图目视/身份核验通过，下一独立D02C诊断；旧A3来源动作/权重不接入。

## 目标与范围

用户要求停止当前整轮、重新开始、停止使用目前A3来源动作，改用PHUMA数据集。继续SuperMimic世界模型训练路径。新实验采用随机初始化的world model/actor/critic，不加载旧A3来源参考训练得到的Q1权重；保留旧实验，不重启R04V2或A3。

工作根为huanyu `/opt/data/private/YuanQi_Q1`，代码副本 `/tmp/yuanqi-supermimic-q1-20261023`、Git分支Q1。新源码置于 `script/supermimic_q1_phuma/`，不改旧冻结目录或历史终态。大型源包、官方模型、转换结果和权重保存在服务器data，Mac仅保留代码/报告/小清单。

## 已完成与关键发现

11:26:30旧R04V2总控/worker均STOPPED_SIGNAL，A932轮/14912控制转移，B未启动；11:27:32核验无存活组成员、三份权重SHA及冻结条件通过，详见[停止记录](README_SUPERMIMIC_Q1_HORIZON_PAIR_20261024.md)。

PHUMA原包 `/opt/data/private/yuanqi_ops/data/phuma/data.zip` 实测3,416,370,206字节，SHA256 `7bd38634f1ec075af907a2bce10616b083a5414885674ce6630664124e22e495`。ZIP有146152个NPY、展开4,290,810,000字节；G1与H1-2各73076。两个机器人版本不是两套独立采集。新主线优先G1源模型到Q1的重定向，不直接复制G1关节角，也不使用历史PHUMA→X2/A3派生数据。

抽查G1/H1-2各一份，字段为root_trans、root_ori、dof_pos、fps；均为30Hz、有限数组，G1为48帧/29关节，H1-2为35帧/27关节。初查使用仅允许NumPy数组构造器的受限反序列化，不执行包内脚本。此抽查不证明全库格式或质量。

官方源码固定 `DAVIAN-Robotics/PHUMA` 提交 `2faf15b52a3498bfcc3352ecb443dc0b5cd642ac`。已读取官方G1配置、MJCF与预处理入口：root_ori使用xyzw，读取到MuJoCo须转wxyz；G1官方MJCF启用23铰链，导出NPY在前19关节后和末尾各插入3个零腕部列，形成29列。须核验零列并恢复23关节顺序，不能把29列直接塞进该模型。官方源包不含LAFAN1/LocoMuJoCo，本轮先使用已在ZIP中的动作，不宣称已有额外两库。

## 输入准备与验证顺序

新增 `acquire_source.py`：只下载固定官方元数据、划分文件、G1 XML与其引用网格；逐文件核对Git blob身份/大小并登记SHA256，不执行官方安装或下载脚本。源代码许可证已读取为Apache-2.0；原始动作来源与外部资产不据此自动宣称可重新分发，仍留忽略目录。

获取代码 `3f5fbb2050d8063f3537423decfcb32f7c793be8` 已发布并核验Q1远端。11:43:02–11:45:21获取原生退出0，46文件/22,780,139字节；独立逐文件复核大小/SHA及MuJoCo编译通过，模型nq30/nv29/nu23，实际铰链顺序与恢复23列约定一致。源收据服务器 `data/references/phuma_source_2faf15b5/source_receipt.json`、Git小收据 `data/manifests/supermimic_q1_phuma_source_20261024.json`，11,838字节/SHA `533242e77229686dce1d19ac8d4f659c39ca711d45bb7488900b2b47fbe68986`；两份执行记录取回后大小/SHA与服务器一致。0新物理积分/optimizer更新。

服务器请求Hugging Face元数据失败（网络地址不可分配），尚未独立比对官方LFS对象身份；当前原包SHA是已在本机独立读取的固定输入身份，不能称为官方LFS匹配通过。

新增 `motion_io.py` / `scan_corpus.py` / `test_motion_io.py`：受限NumPy解码，验证30Hz、长度、有限值、单位xyzw和六零列。官方test涉及的整个来源组保留test，其余官方train组固定hash留10%development；移除chunk和clip后缀，候选每源组最多一片。固定seed2026102405，首批候选配额train80/development16/test16，分别覆盖移动/下蹲/上肢/其他；时长至少1.98s、根倾斜不超0.6rad、源关节速度P95不超4且最大不超12rad/s。该配额是输入候选，不能称为已合格Q1参考或全库训练。本地六项针对性控制全部通过，无跳过；服务器控制与全库扫描待执行。

11:57:30–11:58:43全库扫描原生退出0，服务器六项控制亦全部通过。73,076个G1 NPY均通过字段/30Hz/单位四元数/六补零列/有限值检查，0格式拒绝；15类别、35,023来源组。官方文件级train/test在5,213个来源组中混合，本轮将这些组整体留test。最终train49,548/development5,438/test17,567/unlisted523片；112候选足额选出（80/16/16），每个来源组仅一片，尚不是Q1合格参考。OOM计数保持14，0物理积分/optimizer更新。

选择清单 `data/manifests/supermimic_q1_phuma_selection_20261024.json` 为114,601字节/SHA `cf22cc86981f088cf4c9ffbe7a5f5b2afbee12a6948295a3fe5acaeb13e0cc11`，执行日志 `logs/session_records/supermimic_q1_phuma_scan_20261024.log` 为5,328字节/SHA `600ceb3a8ba255f08d25946b1f47294858d5eccffa2ee7a2e1cb308305a2a6bf`，取回一致。完整58,852,874字节JSONL留服务器 `data/supermimic_for_q1/PHUMA_P01_scan_20261024/all_chunks.jsonl`，SHA `512df6cd955fe7c7a8b90330e8a429e951f8a6b6237606f2bbc0afed468281d8`。

选后复核发现fitness存在 `_clip_17` 后缀，P01仅处理 `_clip17`，因此P01分组不是最终训练名单。保留该扫描终态/索引；修正为两种clip后缀均移除，在新 `PHUMA_P02_scan_20261024` 重新选择，再独立复核全索引、分组计数和候选原字节。分组只按命名来源茎，尚非subject-disjoint或全库语义去重证明。

新增独立 `retarget.py`、两份配置、`verify_reference.py` 与三项重采样控制，代码已完成、尚待两端数值验证和12候选小批运行。复用只读Mink/quadprog通用求解及Q1导出，输入仅PHUMA G1 FK，不读取A3动作；按中立位落地/腿长比例约0.5082缩放，水平原点归零而保留位移，30→50Hz仅重采样、不慢放。独立验证器显式检查PHUMA 30Hz、Q1 FK/速度导出/源位移，并追加90%帧支撑脚15mm内、最大支撑间隙30mm、自碰撞穿透不超3mm的固定几何门。尚未实测参考是否通过，不把运动学门作为动力学跟随证明。

12:10:51–12:13:43 P02分组扫描/独立重算/12片重定向/独立检查四进程均退出0；九项服务器控制通过。P02共23,225来源组，4,184组官方混合整体留test；train33,440/development3,864/test35,249/unlisted523片，112候选仍80/16/16，独立重读原字节/统计/全索引/分组零交集通过。收据 `data/manifests/supermimic_q1_phuma_P02_scan_check_20261024.json`；P02选择113,320字节/SHA `b26fa28ea35d06321369bc8f50fa487d399182f0711ac3c2f6389340afd8935e`。

P02首批12片独立参考合格0，虽时间/FK/限位/速度全部通过，11片支撑比例或间隙失败、5片自碰撞超过3mm；不是动力学或训练通过。完整小结果 `data/manifests/supermimic_q1_phuma_P02_pilot_check_20261024.json`，服务器原件 `data/supermimic_for_q1/PHUMA_P02_pilot_20261024/` 保留。进一步逐帧核查：每片首帧关节离中立位均被截在0.04rad；下蹲第0帧脚底-142.85mm，后段-63.58至-8.04mm，导致整段固定抬高142.85mm。证据支持首帧从中立位限速起步造成部分失真，仍不能解释所有动态支撑/碰撞问题。

下一独立P03只增加静态首帧Mink初始化，不改真实帧时间/2rad/s速度门、不添加参考帧、保持原质量门；源协议V2。先数值控制与同12片检查，再决定批量转换。PHUMA训练入口及固定3000轮后匹配完整评测流水线已实现，尚未发布/验证/登记/启动。

P03实际12:21:15–12:23:05三进程原生退出0、10项数值控制通过。首帧初始化后12片中1训练上肢片通过全部独立门（P02为0）；下蹲原固定抬高142.85mm等异常减少，但其余仍未通过支撑/碰撞，不足以启动三划分训练。P03原件 `data/supermimic_for_q1/PHUMA_P03_pilot_20261024/`，独立收据30,192字节/SHA `370ecb9a6c928b605b738334e9e3a71cfa3c353702fedae0e4c4d686e42f9d98`；代码 `7eae5a90a35f50e832fdaff7182d34e48238920f` 已发布核验。

下一用相同首帧算法转换固定112候选并独立筛选。验证器只需位置/速度/接触几何，改用MuJoCo fwdPosition/fwdVelocity以避免无关约束加速度求解；新增八种Q1状态与完整mj_forward逐数组/接触精确一致的控制，待服务器实跑。新增四项训练输入合同控制，拒绝旧A3协议、pilot、跨划分同源和已变参考；首轮计划3000轮/16环境/batch128/H32，原生MuJoCo在线交互和可微世界模型BPTT，三网络随机初始化，完成后匹配全片评测；900秒无进度停止、温度/OOM保护、无自动重启/追加。尚未登记/启动训练。

P03批量实际12:30:16–12:43:31三进程均原生退出0，15项控制全过，包括八种状态位置/姿态/几何/速度/接触与完整forward精确一致及四项训练隔离阴性控制。112片全部转换无异常；转换器初门通过56/15/11片，独立19门进一步筛选为11/4/3片，共18片合格、94片保留拒绝；训练输入仅11片，不是全73,076片训练。服务器 `data/supermimic_for_q1/PHUMA_P03_references_20261024/`；manifest 369,238字节/SHA `ec7f808eba1e08b955fe16b392b306ded75dc1def3b957dbe51352cda3bf3f34`，独立收据275,066字节/SHA `04b28919b2cc748fcf5e69c0ebd8844ce0fc4159197e497433a54b1795b891ae`。代码 `9ec6e0264fc74ee10c5d42f8508c2d6c42dfb3ac` 推送首次因SSH代理断连失败，重试推送及独立远端核验后通过。三份原生日志及执行收据已留logs；0物理积分/optimizer。

合格训练片实际总时长41.30s，包含2移动/1蹲/4上肢/4其他，移动参考最大水平离起点0.90044m/0.28814m；development为1移动/1上肢/2其他，test为1上肢/2其他。该首轮是小规模PHUMA训练闭环，test没有移动片，不能据此声称独立移动测试泛化。不同数据不能直接与旧A3来源参考按总回报作公平改进结论；本轮采用同18片初始/最终完整配对比较。

## 版本、命令与产物

服务器Python `/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`；运行前重新核验资源和其他作业。源码与小清单先审核/推送Q1并独立核对，再部署/验证身份。获取入口：

```bash
$PY script/supermimic_q1_phuma/acquire_source.py --output data/references/phuma_source_2faf15b5
```

上述获取和以下全库扫描均已完成；完整数据/模型文件不进Git：

```bash
$PY script/supermimic_q1_phuma/scan_corpus.py \
  --archive /opt/data/private/yuanqi_ops/data/phuma/data.zip \
  --vendor data/references/phuma_source_2faf15b5 \
  --output data/supermimic_for_q1/PHUMA_P01_scan_20261024
```

官方来源为PHUMA GitHub、Hugging Face数据卡及固定提交的 `src/curation/preprocess_g1_folder.py` / `asset/humanoid_model/g1/custom.xml`。全库扫描/重定向及新训练的实际结果在执行后补记，当前不预写通过。

## 保存与下一步

首轮已登记并启动 `data/supermimic_for_q1/PHUMA_R01_B128_3000_20261024`；只使用P03独立检查合格的train片段，全部合格development/test仅评测，原A3来源动作与旧Q1权重均不参与初始化。训练源保持可微世界模型+actor BPTT/critic，H32/B128、16环境、WM batch64/每组10更新，seed2026102406。匹配评测除原完整率/回报/关节门外另报根位置RMSE、参考/策略最大水平位移和腿/腰/臂误差，按三种真实划分分别汇总；跟随门为完整无早停、关节RMSE不超0.15rad、根位置RMSE不超0.10m。该门是仿真研究门，不代表真机验收。

登记和启动分开执行，以下是本轮实际使用的复现入口；既有实验已启动，不能再次登记或启动同一路径。须先有合格train/development/test及已核验的发布代码：

```bash
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
CODE_COMMIT=ed4217384e7abfc8a9cea0b23b95592d4a3e19b0
$PY script/supermimic_q1_phuma/pipeline.py register \
  --experiment data/supermimic_for_q1/PHUMA_R01_B128_3000_20261024 \
  --root /opt/data/private/YuanQi_Q1 \
  --motions data/supermimic_for_q1/PHUMA_P03_references_20261024 \
  --reference-receipt data/supermimic_for_q1/PHUMA_P03_references_20261024/independent_check.json \
  --scan-check data/supermimic_for_q1/PHUMA_P02_scan_20261024/independent_check.json \
  --backend-receipt data/supermimic_for_q1/P07_backend_20261023/receipt.json \
  --steps 3000 --seed 2026102406 --code-commit "$CODE_COMMIT"
$PY -u script/supermimic_q1_phuma/pipeline.py run \
  --experiment data/supermimic_for_q1/PHUMA_R01_B128_3000_20261024
```

控制器依次训练→保存审计→初始全片评测/独立重算→最终全片评测/独立重算→三划分配对汇总；达到预算后不自动追加/恢复。代码/配置/输入绑定于spec，阶段前再次逐文件核验；stdout/stderr统一写 `logs/PHUMA_R01_B128_3000_20261024/`。spec、controller_status、run/status、权重与评测数组在实验data目录；三个网络/Adam及冷启动检查由独立保存审计确认，尚未执行的阶段不预写通过。

批量验证期间源ZIP顺序哈希的只读页缓存使cgroup usage升至30,429,323,264字节；仅对PHUMA ZIP和两份全索引发出POSIX_FADV_DONTNEED后为26,922,516,480字节，实际释放3,506,806,784字节读缓存。数据字节保持，OOM仍14、GPU39摄氏度/空闲17,788MiB。该动作不是删除模型或网盘备份，全部资产仍LOCAL_ONLY。

停止、源获取、分组/重定向代码及全部批量小结果已保存并发布，`3d6a9a2c0b44ee9832bd84df9995014525b9d21d` 推送/独立Q1远端核验通过。服务器新增两项汇总控制原生退出0：关节0误差但根不移动时根RMSE0.7071m且拒绝成功；早停前缀0误差仍按0.5完整率拒绝成功。合计17项控制全部通过。登记入口将Q1原始24网格/3模型与既有P01固定输入身份再比对并冻结，不把旧编译收据当作新训练效果。训练发布 `ed4217384e7abfc8a9cea0b23b95592d4a3e19b0` 已推送并独立核验Q1远端，实际启动与实时核查如下。所有大型工件LOCAL_ONLY；旧R03/R04V2与D05B不作为本轮输入或通过证明。

## R01实际启动与运行核查

12:56:33.895已派发独立后台controller，12:56:36进入train阶段。controller PID/PGID799647、start_ticks3610979946，worker PID799650、start_ticks3610980180，boot_id `d994e71f-bae0-46ba-a5f7-6c2163771c6c`；这些是核查时身份，恢复必须重读状态及/proc确认，不能只按旧PID操作。spec共121个冻结源码/配置/模型网格/参考输入，29,763字节/SHA `66bd6b397592decee46ff772450d2e3f182d2e358e99e64af82a761d99a9058c`。

13:04:22独立只读核查16项全部通过：spec与launch SHA相同、121冻结文件未变、controller启动身份/worker父子关系/存活状态一致、状态新鲜、PHUMA-only与无父权重随机初始化、实际只训练11片且划分互斥、发布身份/3000轮预算匹配、已有真实WM和actor更新、日志数值有限、OOM无新增、内存低于30GiB保护线、控制器处于训练阶段。worker已400/3000轮、6400控制转移、109组WM/59次actor更新，在线计时425.82秒。该快照只证明训练已发生，不是匹配效果结论。

同次GPU为RTX3090、39摄氏度、使用7477MiB/空闲16659MiB；cgroup使用30,656,466,944字节，额度34,359,738,368字节，OOM仍14。内存保护为32,212,254,720字节（30GiB）；训练未改保护阈值或其他作业，冻结源码保持不动。初始step0为68,435,755字节/SHA `9e4d66cb26a1eab515745715f850bc93edb445bec53becba7dee978c6270abfc`，本次仅核身份，完整张量/Adam保存审计留在训练完成后的流水线。

小证据已取回，四份原始JSON和训练日志13,024字节前缀均独立与服务器大小/SHA一致：

- [spec](../data/manifests/supermimic_q1_phuma_R01_spec_20261024.json)、[launch](../data/manifests/supermimic_q1_phuma_R01_launch_20261024.json)、[PHUMA绑定](../data/manifests/supermimic_q1_phuma_R01_binding_20261024.json)、[训练准入](../data/manifests/supermimic_q1_phuma_R01_admission_20261024.json)。
- [启动与学习实核](../data/manifests/supermimic_q1_phuma_R01_launch_verification_20261024.json)、[原生日志前缀](../logs/session_records/supermimic_q1_phuma_R01_training_prefix_20261024.log)、[本次只读核查源码](../logs/session_records/supermimic_q1_phuma_R01_verification_source_20261024.txt)。

后续按原3000轮预算继续，controller自动执行保存审计、同18片初始/最终完整动力学评测、独立轨迹重算和train/development/test配对比较。尚无final或最终跟随成功结论；test不含移动片，保留该泛化限制。下一恢复先读本节，再读取实验 `controller_status.json`、`run/status.json` 和日志；如果保护停止则保留终态并定位，不自动重启或追加预算。训练代码已发布核验，本次启动记录随本阶段文档提交，发布身份以Q1 Git历史和独立远端核验为准。

## D01部分保存与数据瓶颈

13:17:09第500轮CPU只读审计原生退出0、15项全过：三网络参数均变化、Adam为1490/99/1980且与149组WM/99次actor一致，初始空Adam、保存有限/manifest/121冻结文件及无新增OOM通过。权重仅留服务器，完整预算审计仍待原流水线。新增固定112候选筛选汇总/独立重算一致：支撑比例85片、间隙47片、自碰撞44片失败（原因重叠），18合格/94拒绝不变；训练移动仅2/32、test移动0/6。原训练未改门或输入，13:21仍原进程875轮，GPU40摄氏度/OOM14。详见[D01核查与数据诊断](README_SUPERMIMIC_Q1_PHUMA_DIAGNOSIS_20261024.md)；这些结果支持后续检查重定向合格率，不是最终学习效果或新增候选放行。

## D02诊断延后与恢复

两端四项刚体几何控制通过，但八训练片真实分析尚未完成：D02因模块路径下函数名不符退出1，修正发布`248c2cf26ae179be50e989486e0763bd0adc2d77`经推送/独立核验和真实依赖导入后，D02B因模型加载后的内存余量保护退出1。均未创建输出目录或形成样本结果，未改变当前训练。13:52:53复核原进程仍运行，1775/3000轮，cgroup30,843,056,128字节、OOM14。详见[D02失败与资源保护](README_SUPERMIMIC_Q1_PHUMA_SUPPORT_MAPPING_20261024.md)及其小收据/真实日志。

不放宽30GiB训练保护线，训练期间不再并行尝试该诊断。下一先完成原3000轮流水线和18片完整配对，再重新核验资源，以独立D02C输出路径恢复足底归因；不把已有输入筛选计数当作具体源/映射/IK归因通过，亦不将训练日志回报当作最终跟随效果。

## R01完整训练与保存检查

14:34:30训练阶段原生退出0，状态`TRAINING_FINISHED_PENDING_EVALUATION`、error=null。实际3000采集轮、48000控制转移、1149组WM更新和1099次actor更新，三个网络从随机初始化开始，原A3参考/旧权重未接入。训练循环5860.723658秒。120条记录的GPU最高40摄氏度、cgroup最高31,576,260,608字节，低于原30GiB保护线32,212,254,720字节，OOM计数始终14；这是采样证据，不是连续峰值证明。

完整CPU保存审计原生退出0，11项门全部通过：完整预算、step0/初始空Adam、训练更新计数、Adam步数、三网络参数变化、有限张量/日志、源码未变、输入未变、指标到达预算和无新增OOM。实际Adam为WM11490、actor1099、critic21980，参数状态14/20/6；网络最大参数变化分别0.365294/0.103644/1.680740。审计读取并核验8份checkpoint身份，不再重复本轮已完成张量审计。

final位于服务器`data/supermimic_for_q1/PHUMA_R01_B128_3000_20261024/run/final.pt`，205,225,043字节/SHA`7db966ff922e067073331d922512fa37c146783a78231a7d93f3538f23ae5c18`。原始训练收据2176字节/SHA`bc310ff18d390df1ee2ceda7599a8cdea7929ecdc298f9579f75798a0a6dcea2`；8份权重均只留服务器，LOCAL_ONLY。

14:36:06实核原训练worker不存在，原controller仍运行并进入`evaluate_final`。训练、保存审计、初始18片完整评测和初始独立轨迹重算四阶段均原生退出0；最终评测/重算/三划分比较还未取得完成证据，不提前写成成功。接下来先等原流水线终态、读取并核对18片配对结果，再执行已发布V01图表和D02C诊断；不自动续训或追加预算。

## R01完整配对结果

14:36:50原controller终态为`PHUMA_TRAINING_AND_MATCHED_EVALUATION_COMPLETED`、error=null。14:38:44独立读取7个阶段事件均退出0并逐份核对日志大小/SHA；121冻结文件未变，8份小收据/比较身份、初始/最终评测与匹配权重身份、18唯一来源组和真实三划分名单一致。用标准库fsum独立重算全部三划分均值、完整率、失败数和跟随门数，与原比较一致。该核查不重复已完成张量审计；V01还将从原始数组独立重算根部/关节误差。

| 划分 | 片数 | 提前失败 初始→最终 | 平均关节误差 初始→最终(rad) | 平均根部误差 初始→最终(m) | 跟随门 最终 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 11 | 3→2 | 0.287358→0.274203 | 0.089137→0.064417 | 0/11 |
| development | 4 | 2→0 | 0.288644→0.263389 | 0.158502→0.084466 | 0/4 |
| test | 3 | 0→0 | 0.272425→0.240971 | 0.034695→0.032219 | 0/3 |

关节/根部误差在提前失败片段中仅覆盖执行前缀，不把上述train/development均值变化当作同长度完整轨迹的公平误差改善。train平均完整率反而0.896392→0.877126，原因包括上肢`bd4d0aff3163d8f7`从160/172帧缩短到76/172帧；失败数减少不能掩盖该退化。test三片在两次评测中均完整，因此0.272425→0.240971rad是在相同完整分母上的改善；test没有移动片，不能推广为移动测试泛化。

三段合格移动均未达到跟随门，最终具体为：

| 片段 | 最终执行/完整参考帧 | 关节RMSE(rad) | 根部RMSE(m) | 实际/参考最大水平离起点距离(m) |
| --- | ---: | ---: | ---: | ---: |
| train `1bc87468bec5a472` | 38/184，提前失败 | 0.302248 | 0.133572（前缀） | 0.351007/0.900436（实际为失败前缀） |
| train `26832240824d6963` | 199/199 | 0.191712 | 0.192044 | 0.100590/0.288144 |
| development `dac69a84c1f08190` | 199/199 | 0.335506 | 0.263104 | 0.066904/0.373315 |

第一片38控制步约0.76秒即失败，位移不能当作成功走路。开发移动初始根部RMSE0.262304m、最终0.263104m，基本未改善；腿部误差0.256288→0.190480rad，但身体移动仍明显不足。不能仅因回报上升而认为已复现可用策略，更不能把PHUMA和旧A3参考的不同分母直接比较成数据切换带来改善。

原权重、原数组及日志均保留。小证据：[完整核验](../data/manifests/supermimic_q1_phuma_R01_completion_verification_20261024.json)、[比较](../data/manifests/supermimic_q1_phuma_R01_comparison_20261024.json)、[训练收据](../data/manifests/supermimic_q1_phuma_R01_training_receipt_20261024.json)、[完整保存审计](../data/manifests/supermimic_q1_phuma_R01_training_audit_20261024.json)和对应初始/最终评测/独立重算JSON；原生日志留`logs/session_records/supermimic_q1_phuma_R01_*_20261024.log`，读取源码留同目录`completion_verification_source`。Mac未拉取权重或原始大型数据，全部大工件LOCAL_ONLY。

14:38:44原进程组无活跃成员，GPU38摄氏度/使用6348MiB、cgroup27,361,579,008字节、OOM仍14。下一先执行[V01图表](README_SUPERMIMIC_Q1_PHUMA_PLOTS_20261024.md)并目视真实轨迹，再以独立D02C核查输入支撑映射；D02C只解释筛选瓶颈，不冒充当前策略移动失败的因果诊断。选择下一训练方向时以train/development证据为依据，保留当前终态和匹配基线，不盲目追加预算。长期SuperMimic for Q1目标仍未达。
