# PHUMA到Q1分批扩量

记录日期：2026-10-24，北京时间。阶段：SM-Q1-PHUMA-P04。状态：截至B009，829个不同候选的转换/独立19门及209个实际NPZ跨批时轴重算完成；train196/708.38秒（移动16/57.14秒）、dev6/21.26秒、test7/27.56秒。全部本轮句柄原生退出0，17:47实核无转换/汇总进程。30分钟/1小时扩量与新训练尚未完成。四路内存及模块导入失败保留。

## 目标与范围

继续用户的SuperMimic for Q1世界模型目标，纠正R01仅11片/41.30秒的参考覆盖。AMASS原库在手但缺SMPL-X男女模型，完整BONES-SEED包尚缺，见[三源盘点](README_SUPERMIMIC_Q1_SOURCE_EXPANSION_20261024.md)。在等待这些输入时，直接处理现有PHUMA原料，不把缺少BONES当成所有工作停止的理由。

现有完整P02索引已有26.44小时温和训练候选。这是包内标称时长，不是独立或Q1合格时长。本轮先固定新候选名单、生成和独立核验实际Q1参考，再按实际合格秒数累计。工程目标仍为30分钟、再1小时；未达到时不以候选量代替。

## 实现与选择依据

- 新`select_expansion.py`直接读取已核验的73076行索引，不重扫原始ZIP。沿用P02官方来源组隔离：官方test组不进入训练，开发集仍按原固定种子保留。
- 整体排除之前112候选的命名来源组，避免重复转换和重复计算。原R01的11训练片、输入及终态保留；后续嵌套小库/扩量库可按原身份引用合格历史参考。
- 新名单每个命名来源组只选一个chunk，且每个原文件SHA只出现一次。全索引相同字节跨划分时，优先留test、再development、再train，即使高优先级那一片不满足温和运动门，也不将副本放回训练。此规则不是语义或subject去重证明。
- 新选择种子2026102407，候选预算train最多2048（移动1024/下蹲256/上肢256/其他512），development/test各32。池不足时登记实际数，不伪造补足。明确楼梯、梯子、椅子、攀爬、坐姿、爬行名称暂排除，保留精确正则。
- `prepare_expansion_batch.py`按固定offset/count生成一个独立批次，复用原始字节/划分重算、V2重定向及独立19门检查。每批独立输出selection、scan_check、references、reference_check及completion；失败保留部分产物，不覆盖重跑。
- 对项目大ZIP的完整SHA仍实际计算，但按读取区间释放文件页缓存，避免反复哈希把共享容器顶到原30GiB保护。未清空系统缓存，不读取或操作其他项目认证材料。
- 首批继续保留原19门，实际测量新候选通过率和时间开销。论文式持续悬空过滤是后续独立协议候选，本次没有删掉贴地门或把几何合格当动力学/策略合格。

## 平台与执行前证据

当前主机huanyu，项目根`/opt/data/private/YuanQi_Q1`。本轮只读实核GPU RTX3090，39摄氏度、空闲17788MiB、利用率0%；cgroup内存28,712,398,848/34,359,738,368字节、OOM14。CPU cgroup额度2400000/100000，即24核。原Q1训练进程均已终止；保留两个可视化HTTP服务与其他项目已有进程，不以这些快照代替稍后的实时启动检查。

代码副本`/tmp/yuanqi-supermimic-q1-20261023`，分支Q1，本阶段前HEAD`ce1f958c62956feb98348692b468e5652b23067c`。新增代码本地四项控制通过：旧候选和同录制组不重用；相同字节跨划分按test/dev优先且不因不合格回流；划分内内容去重；输入顺序改变名单仍确定且明确交互名称排除。语法检查通过。服务器数值执行待下节补充。

## 可复现入口

以下命令须在项目根运行，使用当前重新核验的OOM基准与新输出目录。首次真实执行完成后登记原生日志与大小/SHA。

```bash
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
$PY script/supermimic_q1_phuma/select_expansion.py \
  --base-selection data/supermimic_for_q1/PHUMA_P02_scan_20261024/selection.json \
  --output data/supermimic_for_q1/PHUMA_P04_selection_20261024.json

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  $PY -u script/supermimic_q1_phuma/prepare_expansion_batch.py \
  --root /opt/data/private/YuanQi_Q1 \
  --selection data/supermimic_for_q1/PHUMA_P04_selection_20261024.json \
  --vendor data/references/phuma_source_2faf15b5 \
  --output data/supermimic_for_q1/PHUMA_P04_B001_20261024 \
  --offset 0 --count 32 --expected-oom-kill 14
```

同一输出只执行一次；后续批次用不同编号和不重叠的固定slice。运行时日志存`logs/PHUMA_P04_Bxxx_20261024.log`。未产生completion不能登记批次完成。不会自动启动原R01、旧A3或旧STOPPED实验。

## 实际结果与边界

新代码提交`01890c10f46d9a20f8837c229cf978027129f083`已显式推送Q1并独立核验远端。5份复用核心源码及3份新脚本两端SHA一致；4项服务器来源隔离控制通过。

选择器实际退出0，产生`data/supermimic_for_q1/PHUMA_P04_selection_20261024.json`，1,803,556字节/SHA`c4dbc3054ed1a99c67a05667e1dc085af187e802d9740f331036b619baa88f4f`。实际train1800（移动1024/下蹲8/上肢256/其他512），development29，test32，共1861个不同来源组、1861份不同原字节。训练候选标称6558.3333秒（109.31分钟），开发99.6秒、测试115.3333秒，均未成为Q1合格时长。下蹲命名组不足预算，不用其他类别伪称补足。

完整索引存在6组跨源划分的同字节身份，已按test/dev优先隔离；不是6组已证明的语义重复。旧112候选来源组共覆盖725行、全部排除；另明确交互名称221行排除，统计见selection的expansion.proof。

北京时间16:49:11启动B001–B004，分别固定offset0/32/64/96、count32。四路独立来源重算通过，但均在`retarget.convert`的逐片resource_check触发`PHUMA scan memory protection`，原生退出1；未生成完整retarget manifest、独立reference_check或completion，不能登记参考合格。日志及部分产物留原编号，不覆盖重跑。退出后cgroup内存27,665,571,840字节、OOM仍14；需单路新编号继续，原30GiB保护不变。

本轮已有新增运动学计算但尚无新增合格结论，0新动力学积分、0optimizer更新。失败不证明PHUMA数据质量差，也不作为不可训练证据。

B001–B004每批均只保存首片NPZ与IK trace，随后触发内存保护；无完整manifest，因此不能用补校验模式接入。各目录和日志保留。单路B005在新编号重新处理固定offset0/count32，已完整转换32片并写入retarget manifest；随后因`retarget`修改sys.path，使普通`import verify_reference`误绑定到通用Q1脚本，调用不存在的verify而退出1。不是Q1参考检查失败。退出后内存27,282,817,024字节、OOM14。

修复采用按绝对文件位置加载PHUMA校验器，并增加受限`--verify-existing`入口：必须有原完整转换manifest、原scan_check与精确相同selection；不得已有reference_check/completion。只有缺失的独立验证和汇总会执行，0新增IK。原失败日志不覆盖；新日志用B005_verify。新单路批次继续从新offset开始。

修复发布`450210e9a2f6dea6ed0acb1f7aa7fc51e1e1c793`已独立核验远端；新版wrapper SHA`4068da689c9c5d5cb0e809d5540452e4c5fbe97367ba44288ee91ae10c2e0655`两端一致。对B001的补验阴性控制实际因缺manifest退出1，无新增转换/输出。B005补验实际退出0，32片中7片合格：开发6片/21.26秒（移动1片/3.96秒），测试1片移动/3.96秒；该批没有训练片。拒绝门重叠计数：15mm支撑23、自碰撞12、30mm间隙11、转换器7、IK残差1、动作覆盖1，不把它们相加当拒绝片数。补验25.769秒、OOM14，0新IK。

B005 retarget manifest：112,357字节/SHA`814f24deff6f95153ad3c864c0a4b6ecfc2a39ae4c2862fb4177d4129709aeea`；独立reference_check：80,471字节/SHA`6f5573781d7e4cedbe61403fd0f29e731fd04afe5da9c311f22a94ae7e2829bf`。日志分别`logs/PHUMA_P04_B005_20261024.log`（原失败）与`logs/PHUMA_P04_B005_verify_20261024.log`（补验成功）。

B006使用同一新版wrapper、offset61/count256，即全部是新训练候选，与B005的0–31不重叠；32–60的29个测试候选尚未转换。当前活跃执行句柄在本会话为88888，服务器日志`logs/PHUMA_P04_B006_20261024.log`，输出`data/supermimic_for_q1/PHUMA_P04_B006_20261024`。最后已输出160个转换器片段结果；内存29,215,936,512字节、OOM14、GPU39摄氏度/0%利用率。仍须等待原生完成和独立19门，不能据日志转换器初门计为合格训练参考。

新增`summarize_expansion.py`将只接收有completion的批次，核对完整名单slice、原字节/来源组跨批唯一性、检查收据与实际参考SHA，并从保存的`time_s`与帧数重新计算50Hz时长。旧112来源组一旦回流即拒绝，重复列同批次也应拒绝；该汇总尚待实际执行。

## 首批最终核实

北京时间17:15:28完成最终汇总，原生退出0。B006实际完成256训练候选，66合格，424.445秒；B007补offset32/count29的剩余测试候选，6合格，67.827秒，原生退出0。二者无转换异常，OOM14保持。B005/B006/B007覆盖完整selection前缀0–316，互不重叠；B001–B004不计作完整批次。

| 划分 | 完成候选 | 独立19门合格 | 合格参考秒数 | 合格移动片/秒数 |
| --- | ---: | ---: | ---: | ---: |
| train | 256 | 66 | 243.76 | 6 / 23.76 |
| development | 29 | 6 | 21.26 | 1 / 3.96 |
| test | 32 | 7 | 27.56 | 3 / 11.72 |
| 合计 | 317 | 79 | 292.58 | 10 / 39.44 |

训练66片包括移动6、下蹲1、上肢24、其他35。243.76秒为4.0627分钟，是R01原41.30秒的5.90倍，仍远未达到30分钟。这里没有把原11训练片再次加进来，也没有把6558秒候选时长算作已合格。新的test含3个移动来源组，修补旧test无移动的评测覆盖缺口；还未执行策略测试，不是泛化成功证据。

最终重算直接检查79个实际NPZ的SHA、50Hz时间轴和29列qpos维度，14,708帧；总时间按各片`(frames-1)/50`分别求和。317个候选来源组/原字节身份均唯一，重复列B005的阴性控制实际拒绝且未创建汇总输出。两端4来源隔离控制、缺manifest拒绝控制及重复批次控制均真实执行。完整小清单[实际参考与跨批重算](../data/manifests/supermimic_q1_phuma_P04_progress_20261024.json)，55,410字节/SHA`bc26a6a490d1d9aa36d08ad90d6369311f6b2db69ab4da20de6fc380e39896ac`，与服务器原件一致。

另读两个既有selection补核：完整1861新名单与旧112候选之间，来源组交集0、原字节SHA交集0。此项支持后续按身份嵌套原训练小库，不代替跨数据源或语义/subject去重。后续筛选协议与训练选型仅使用train/development，test保留最终效果，不根据测试效果调门。

317片的拒绝门重叠计数为支撑15mm201、自碰撞117、30mm间隙111、转换器68、IK残差24、动作覆盖4、固定地面原点平移1。训练移动候选83片仅6片合格（7.23%），因此下一要优先保证移动有效覆盖，不能只扩静止上肢片数或把所有拒绝都当原始数据差。

17:17收尾实核：cgroup内存27,469,729,792字节、OOM14；GPU39摄氏度、空闲17788MiB、利用率0%；无prepare_expansion_batch进程。全部已打开执行句柄均原生终止，0新增动力学积分/optimizer。本阶段[执行记录](../data/manifests/supermimic_q1_phuma_P04_execution_20261024.json)固定各原生日志大小/SHA、失败与补验退出码、代码发布及下一个offset。没有新模型权重或新训练效果。

最终汇总的可复现命令如下，原输出已存在，不覆盖重跑；如需独立复核必须用新输出名，不重做已完成IK或物理运行。

```bash
$PY script/supermimic_q1_phuma/summarize_expansion.py \
  --selection data/supermimic_for_q1/PHUMA_P04_selection_20261024.json \
  --batches data/supermimic_for_q1/PHUMA_P04_B005_20261024 \
            data/supermimic_for_q1/PHUMA_P04_B006_20261024 \
            data/supermimic_for_q1/PHUMA_P04_B007_20261024 \
  --output data/supermimic_for_q1/PHUMA_P04_progress_20261024.json \
  --expected-oom-kill 14
```

## 保存与下一步

代码/README/两个入口按Q1进行diff、payload审核、提交、显式推送并核对远端；服务器部署前核验新增路径不存在及旧引用源码SHA。大型参考、IK轨迹与日志原件保留服务器，LOCAL_ONLY，Mac只保存小清单。

首批收尾时的下一步为offset317、B008单路继续；实际执行见下节。训练移动合格率仅7.23%，因此后续也应基于论文持续悬空语义选定独立Q1筛选协议并做匹配动力学验证，不能追改原19门。先扩大真实有效覆盖、绑定固定开发和含移动独立测试，再登记新的世界模型/H32训练与交互预算。原R01及失败终态保留；30分钟/1小时参考与可用Q1运控目标均未完成。

## B008完成与B009执行检查点

B008固定offset317/count256，原生退出0，用时411.584秒，转换异常0、OOM14。独立19门接纳62个训练参考/217.60秒，其中移动5片/18.36秒、上肢32片、其他25片。与已完成首批作算术累计为573个不同候选、训练128片/461.36秒、移动11片/42.12秒；本段累计尚未冒充新的跨批时轴重算结果。

B008输出`data/supermimic_for_q1/PHUMA_P04_B008_20261024/`，日志`logs/PHUMA_P04_B008_20261024.log`。retarget manifest为835,221字节/SHA`eccde8e8618bfbb696b825c9ab407e76d2cf0f2b5b895b324e26d8e79ca4b6f6`；reference_check为628,035字节/SHA`82a11b14a22d335ce6cb473379ada44cc3444bebd52f21bd7475ac338f02382a`。仍使用已发布wrapper SHA`4068da689c9c5d5cb0e809d5540452e4c5fbe97367ba44288ee91ae10c2e0655`。

B009固定offset573/count256，输出`data/supermimic_for_q1/PHUMA_P04_B009_20261024/`，日志`logs/PHUMA_P04_B009_20261024.log`，原会话执行句柄52394。北京时间17:41:42实核仍在独立参考检查，completion尚未产生，内存29,476,315,136字节、OOM14。继续等待原句柄，不重启、不把转换器初门或日志片数登记为新增合格结论。成功结束后下一个未转换offset为829，必须用新编号B010。

同轮仅查询MuJoCo对象内存元数据，0新增物理步：模型`nbuffer=125623784`字节，数据`nbuffer=1070032`、`narena=14680064`，27 bodies/71 geoms/24 meshes/257726 mesh vertices。当前后端每环境深拷贝模型，64份模型缓冲区本身约8.04GB，尚不包含状态、arena和网络；现有共享容器余量不支持直接将16环境改为64或128。尚未实现模型槽复用、改变仿真后端、扩环境或启动新训练，后续如采用须独立核对数值等价与资源。

## 截至B009的实际汇总

B009最终原生退出0，用时413.412秒，转换异常0、OOM14；256候选中68片通过独立19门，均为训练参考，247.02秒，其中移动5片/15.02秒、上肢29片、其他34片。retarget manifest为836,333字节/SHA`d2d9e1095f77daa0f9893d26efc2185662276fd1687cfdab7c81a155e56338ee`；reference_check为628,400字节/SHA`1d7aa274ac5ed6fdcd232ef0a3c360c0fd38526a9bee80f44e87b1de18252bbc`。

北京时间17:44:56新汇总原生退出0，核对B005至B009五个完整批次、829个不同来源组/原字节身份及209个合格NPZ的38,069帧。沿用原汇总程序，没有重做IK或动力学。新输出使用独立名称，不覆盖首批317片汇总。

| 划分 | 完成候选 | 独立19门合格 | 合格参考秒数 | 合格移动片/秒数 |
| --- | ---: | ---: | ---: | ---: |
| train | 768 | 196 | 708.38 | 16 / 57.14 |
| development | 29 | 6 | 21.26 | 1 / 3.96 |
| test | 32 | 7 | 27.56 | 3 / 11.72 |
| 合计 | 829 | 209 | 757.20 | 20 / 72.82 |

708.38秒为11.8063分钟，约为原R01训练参考41.30秒的17.15倍。train另含下蹲1、上肢85、其他94片。训练移动候选254片中16片合格（6.30%）；数据扩量有实际进展，但有效移动覆盖仍很薄，不能把总片数增加当作已经训练出更好的移动策略。

新小清单[截至B009的实际时轴汇总](../data/manifests/supermimic_q1_phuma_P04_progress_B009_20261024.json)为139,678字节/SHA`1c48bc9084aeb882194109264f1b5692e6424d42bbf1a6441b41160eb6741046`；[本轮执行与资源证据](../data/manifests/supermimic_q1_phuma_P04_B009_execution_20261024.json)登记B008/B009 completion、原生日志、退出码、模型缓冲区查询及缺少的AMASS/BONES输入。旧首批清单和失败记录保留。

```bash
$PY script/supermimic_q1_phuma/summarize_expansion.py \
  --selection data/supermimic_for_q1/PHUMA_P04_selection_20261024.json \
  --batches data/supermimic_for_q1/PHUMA_P04_B005_20261024 \
            data/supermimic_for_q1/PHUMA_P04_B006_20261024 \
            data/supermimic_for_q1/PHUMA_P04_B007_20261024 \
            data/supermimic_for_q1/PHUMA_P04_B008_20261024 \
            data/supermimic_for_q1/PHUMA_P04_B009_20261024 \
  --output data/supermimic_for_q1/PHUMA_P04_progress_B009_20261024.json \
  --expected-oom-kill 14
```

以上输出已存在，不覆盖重跑。17:45:39资源实核内存27,766,009,856字节、OOM14，GPU39摄氏度/空闲17788MiB/利用率0%；17:47:23按实际进程argv复核无转换或汇总进程，原执行句柄全部已终止。0新物理步/optimizer更新，没有新训练权重或策略效果。

恢复从完整名单offset829开始，下一新批号B010/count256；尚余1032个训练候选。按单路、独立输出继续，不能把已完成批次重转。新合格参考尚未组装为新训练库，30分钟/1小时门槛未达；下一先补齐移动覆盖、组装可追溯训练/开发/测试库，再绑定独立世界模型/H32训练预算。AMASS原库已在手但缺男女SMPL-X模型，完整BONES G1包/metadata仍未到位，均已告知用户。大型原件仍LOCAL_ONLY；本轮报告和小清单按Q1分支审核、提交、推送并核对远端，同次提交号以Git历史为准。

## B010继续扩量

B010固定offset829/count256，于17:56:28核主机/boot、无训练/转换、GPU38摄氏度及OOM14后单路启动。原句柄81765已原生退出0，用时457.533秒；18:05:34检查完整completion存在，256候选中65训练参考通过，233.00秒，其中移动9片/28.48秒、上肢1、其他55。转换异常0、OOM14。

completion为2,228字节/SHA`8c37909f6143eab03a44eb0e2b7395dad6bec29edd9258406f45c206eb4f6422`；retarget manifest为833,613字节/SHA`01999952ae60bb5c0310e3df2eac2b496c88e80616476f7ec8e6c35a80f17383`；reference_check为628,445字节/SHA`4adb8020ae14ae96299c9d70e11c6723faf359fcb290e599d2d525e794930d3c`。输出和日志沿用独立B010编号，未覆盖已有批次。

已完成批次算术累计1085候选、训练261片/941.38秒（移动25片/85.62秒）；完整跨批时轴汇总留待剩余批次完成，不把此累计说成新汇总。下一未转换offset1085、B011/count256，尚余776训练候选。18:05:34内存25,954,168,832字节，OOM14。0新训练/动力学，正式组库入口及后端准备见[P05](README_SUPERMIMIC_Q1_EXPANDED_RUN_20261024.md)。

B011固定offset1085/count256，原句柄76516原生退出0，用时409.743秒，转换异常0/OOM14。18:15:26核完整completion，接纳train50/179.90秒，其中移动9/34.28秒、其他41。retarget manifest为831,909字节/SHA`99b295dff9d5c77fabc301cf16d94c6f7655317fbbbf110e91451f9fe996e967`，reference_check为628,204字节/SHA`9f9f423453722ff6e740b4ce15a371117fe21f6a86741ff4b0d731e2c652736f`。批次算术累计1341候选/train311/1121.28秒（移动34/119.90秒），尚余520训练候选，下一新offset1341/B012/count256。资源内存26,152,128,512字节、OOM14。

同次从既有selection核得可选命名来源组train移动7664、上肢390、其他7082、下蹲8；P04配额不是全库容量。新增来源/更多候选仍要独立登记、去重和检查，不因原料数量充足而把它们加到合格时长。

B012固定offset1341/count256，原句柄8737退出0，用时406.659秒，转换异常0/OOM14。18:25:01核完整completion：train18/64.94秒，其中移动17/60.98秒、其他1。retarget manifest为836,315字节/SHA`590e6db55eec595c6030fc5a2278bc1b6f9cfb473f80540027fb5bfba18b0a6a`；reference_check为632,547字节/SHA`4e1394099b3f33a64c8ee41d9a4018ccbf102278bd5d506aab73d697c0887851`。已完成批次算术累计1597候选/train329/1186.22秒（移动51/180.88秒）；余264训练候选，下一新offset1597/B013/count256，再B014/offset1853/count8。资源内存26,314,252,288字节/OOM14，0新训练。

B013固定offset1597/count256，原句柄18134原生退出0，用时427.086秒，转换异常0/OOM14。18:34:43核完整completion，train11/40.88秒，均为移动。retarget manifest为836,697字节/SHA`00feba3e30f0ebd17b07739b2ac8da0fbae848e215accc9a292e2506dbed84cd`；reference_check为632,829字节/SHA`68b9c6a08f0010e8876a69c923c02df353aff0ac747bd72c3fb90830731b06bb`。批次算术累计1853候选/train340/1227.10秒（移动62/221.76秒），只余原名单最后8候选。资源内存26,482,847,744字节/OOM14，下一新B014/offset1853/count8。

## 原固定名单完成与新增训练候选

B014固定offset1853/count8，原句柄15953原生退出0，用时33.098秒，8候选均未过原19门，转换异常0/OOM14。retarget manifest SHA`10fcdbd4c024bffc5ba7e6e4e592cebd88a95d2165db0144b0b72273197e9bc4`；reference_check SHA`f5a056139f0dc36139c934db7137ad61f818cae4963c1f84e58e21ef51cb23a0`。原1861候选由B005至B014完整覆盖，失败B001至B004不纳入。

各批算术累计train340片/1227.10秒（移动62片/221.76秒）、development6片/21.26秒、test7片/27.56秒；这是20.4517分钟训练参考，尚未完成全1861候选的独立跨批重算或真实组库，不能将算术累计称作已重算。

新增选择器原句柄49666退出0，固定512移动及512其他训练来源，共1024组/1024份不同原字节。完整排除旧112及原P04的1973个来源组，不新增或改变开发/测试；训练候选标称3706.1667秒，尚未转换或合格。名单`data/supermimic_for_q1/PHUMA_P04_additional_selection_20261024.json`为983,994字节/SHA`631cd06bd0272e279ef587abead63e8849357d34e234ee3c7dfa74af60703a20`。

18:40:50恢复实核huanyu boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`、无相关计算进程，内存26,446,974,976/34,359,738,368字节，OOM14；GPU38摄氏度/空闲17788MiB。下一先以完整B005至B014进行基础组库，再实际验证模型槽后端；新增名单从独立B015/offset0/count256开始单路转换，不覆盖已完成产物。0新训练，所有大型工件LOCAL_ONLY。
