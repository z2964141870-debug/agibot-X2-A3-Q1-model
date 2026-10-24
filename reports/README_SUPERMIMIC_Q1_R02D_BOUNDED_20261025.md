# PHUMA R02D动作域修正匹配试验

记录日期：2026-10-25，北京时间。状态：04:56:56独立启动，初始继承12门及05:00:34实际运行10门核验通过，固定7000新增轮进行中；配对评测尚待。用户要求继续已有资源；BONES由师兄解决，不再处理其下载与凭据。

## 目标与依据

复现SuperMimic可微世界模型、想象BPTT与critic路线。[R02C](README_SUPERMIMIC_Q1_R02_PARTIAL_CONTINUATION_20261025.md)新增1810轮时数值FAILED；[S02](README_SUPERMIMIC_Q1_ACTION_DOMAIN_20261025.md)已确认真实clean动作clip20、想象动作无界的接口差异，并完成服务器5项数值控制、保存1500的15门审计、真实replay H32/BPTT的7门检查。正常探针没有重现崩溃，不能宣称唯一根因或效果改善。

本次从原R02已审计step5000开始，沿用340片PHUMA训练、dev6/test7、seed20261025、64环境/4模型槽、H32、actor128、WM64每组10批，固定新增7000轮/448000转移。保留三网络、三Adam、两采样器及学习率offset5000；replay/episode/RNG重新开始。原R02/R02C FAILED、原A3与STOPPED实验不重启。

主要变量为真实与想象统一clip20；同时增加三个Adam写入前拒绝非有限梯度的停止保护。有限Adam更新不变，不能将两个改变描述成完全单变量因果证明。P08B-V2的612片/36.52分钟新库独立保留，暂不与动作域变量混合；AMASS尚无通过Q1准入的参考。

## 实现与保护

新增`script/supermimic_q1_stability/{run_contract,train_bounded,pipeline_bounded}.py`及`bounded_R02D_20261025.json`，均位于原496冻结文件外。注册只接受三份真实S02原件及其生产者/输入SHA，重新核原5000父资格、R02C固定数据和设置、全部冻结及当前资源；`check`不创建新实验或执行模型/优化器。

训练薄适配器复用原partial/extension循环及4槽后端。真实实例必须为BoundedPolicy，三个Adam各自实际注册指定step-pre-hook，加载父状态后再次确认；声明、入口身份与实际hook证据写入新admission。原网络键和形状保持，有限真实闭环动作仍由仿真裁到相同范围。原评测入口复用相同参数布局与真实动作裁剪，继续完成353片初始/最终MuJoCo配对、两次轨迹重算和三划分比较。

100GiB盘余、12GiB新实验、250GiB来源目录预算；训练30GiB内存/86摄氏度/新OOM变化停止，900秒无进展停止，无自动重启。每500轮保存，训练结束不追加预算。本轮恢复时现场huanyu/原boot核验为GPU14MiB/28摄氏度、内存12.26GB、OOM14、盘余1.153TB，无相关训练进程；共享挂载其他写入仍须现场监控，不把盘余下降全部归因于本任务。

## 版本、命令与产物

工作副本`/tmp/yuanqi-supermimic-q1-20261023`，分支Q1；训练根`huanyu:/opt/data/private/YuanQi_Q1`。先审查、payload audit、提交推送并独立核远端，再逐SHA部署及只读预检，之后新目录独立登记/guard启动。COMMIT须使用实际已发布全号：

```bash
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
ROOT=/opt/data/private/YuanQi_Q1
EXP="$ROOT/data/supermimic_for_q1/PHUMA_R02D_bounded_partial5000_extend7000_20261025"
"$PY" script/supermimic_q1_stability/pipeline_bounded.py check \
  --root "$ROOT" --experiment "$EXP" --code-commit COMMIT --seed 20261025
"$PY" script/supermimic_q1_stability/pipeline_bounded.py register \
  --root "$ROOT" --experiment "$EXP" --code-commit COMMIT --seed 20261025
```

外层runtime_guard命令与真实boot/PID/start_ticks保存新目录launch.json；stdout为同名`logs/`，状态为`storage_guard.json`、`controller_status.json`、`run/status.json`。登记与启动前的资源/输入通过证据不能替代真实训练结果。

## 当前结果、保存与下一步

目前已独立登记/启动，初始继承及首批更新核验完成，详情见下方真实记录。继续固定预算、保存并核数值/资源，随后执行原七阶段动力学配对；尚无新增移动效果结论。任一环节失败即保留真实终态，不修改父记录或放宽门槛。

阶段源码/配置/报告及后续小收据保存Q1；实际发布全号由Git历史及独立远端比对提供。大型权重、轨迹及数据继续LOCAL_ONLY，不声称网盘备份。目标仍未达：训练与数值稳定都不能代替实际MuJoCo移动跟随效果。

## 04:52服务器预检完成

源码发布`39397087493088605e080494d7728ff458d69939`已推送并独立核远端Q1；四份新文件部署前确认不存在，部署后逐SHA一致。新三入口服务器语法和真实`train_bounded.py --help`导入原生退出0；随后只读登记检查原生退出0，三份S02原件/生产者/引用输入、原5000父、R02C全部496冻结与340/6/7划分匹配，0新增动力学/优化器、没有创建新实验目录。

[原样登记预检收据](../data/manifests/supermimic_q1_R02D_registration_check_20261025.json)8003字节/SHA`d0e87ff6efc4662b28677c3f640746d8ea65e62942dc7faad6cd343c295c7aea`；服务器原件位于`data/supermimic_for_q1/PHUMA_R02D_registration_check_20261025.json`。stdout为`logs/PHUMA_R02D_{compile,import_help,registration_check}_20261025.log`。预检04:52:23资源为GPU27摄氏度/余24123MiB、内存12611538944字节/OOM14、盘余1117348257792字节。

已将d7565cc至该源码提交的26份已发布代码/报告/小证据机械同步用户恢复镜像，共320949字节；复制前逐文件比对旧基线或目标、没有未知修改，复制后逐字节一致。未复制未跟踪BONES下载脚本或大型工件。下一完成本次验证记录发布，再正式登记和启动新R02D；此处预检通过不是训练完成。

## 04:56独立登记和启动

预检记录发布`5229ea25ddb7df385105cad5f4e27e4a2f6a5c1c`已推送并独立核远端，恢复镜像四份追加文件也已逐字节同步。正式register原生退出0，最终516冻结，spec160813字节/SHA`fb42496059a517c5e2fa40934dc25ca3b13690e0400964bdb23965e58277c25d`；原partial登记输出只是构造中间模板，实际运行仅接受追加完成动作声明的新spec。新[原样spec](../data/manifests/supermimic_q1_R02D_spec_20261025.json)及[launch](../data/manifests/supermimic_q1_R02D_launch_20261025.json)已取回。

04:56:56.897由Popen独立session、stdin关闭启动runtime_guard；guard839210/start_ticks3616742247，controller839211/start_ticks3616742257，trainer839214/start_ticks3616742577，boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`。启动前实际盘余1093830193152字节、内存12334571520/OOM14/GPU27摄氏度，原516冻结逐SHA保持，无其他相关训练。完整命令保存在launch原件；没有自动重启或放宽预算。

恢复核验应读取本目录三个状态文件，重新按`/proc/<pid>/stat`的start_ticks、boot、非Z状态及cmdline匹配实际脚本，不凭上述旧PID认定仍运行。guard负责其独立controller进程组及子进程退出清理，日志保存在`logs/PHUMA_R02D_bounded_partial5000_extend7000_20261025/`。

## 初始继承与05:00实际更新验证

独立CPU/mmap初始检查原生退出0、12门全过，八树逐张量/容器完全相同；新局部计数0、offset5000、新seed及发布版本一致，全部516冻结无变化，无CUDA上下文。初始保存205325095字节/SHA`3a1bc5dff4966f2754a97e11c5d283b283733bea12b15d4a92a57961e7ba5b38`；字节与父不同由新manifest/计数产生，不是参数/Adam重置。[原样继承审计](../data/manifests/supermimic_q1_R02D_initial_retention_20261025.json)1543字节/SHA`3239fe57bc011e19a7ffd2cd1596d88be664eed715250cdbe31ab958b3436cdd`，stdout同目录`initial_retention.log`。

05:00:34[独立启动核验](../data/manifests/supermimic_q1_R02D_startup_20261025.json)10门全过，SHA`cb6bca4b7bbee9865b7dd82be1b9a6adeeca32cf5b35ef2bb404d20d8482f721`。三个实际/proc进程身份/组及脚本匹配、全部516冻结保持、实际BoundedPolicy与三个Adam指定hook在父状态加载后仍绑定，admission和发布入口一致。读到的05:00:02进度为275新增轮/17600转移、累计5275轮、WM59组/actor9次；actor梯度0.131855、critic loss0.0100401、WM loss0.196068及其余日志指标有限。检查本身0新增物理/optimizer。

核验健康GPU39摄氏度/余22909MiB、内存15038775296/OOM14、盘余1074733170688字节，新实验分配约205.65MB。该通过只证明正确初始化、入口保护与首批数值更新，不证明已避开此前1810崩溃或学会移动。继续原7000新增轮，不混612片，固定结束后原7阶段完整配对；后续需要保存/Adam计数及真实动力学轨迹证据，不能由在线均值代替。当前主目标仍未达，所有大型工件LOCAL_ONLY。

## 中间保存审计入口

新增独立`script/supermimic_q1_stability/audit_bounded_progress.py`，不在原516冻结内；CPU/mmap读immutable的500倍数保存，核对三网络、三Adam继承计数、两采样器有限性、原manifest/340/6/7及全部516冻结、完整有限日志前缀。沿用28.75GiB/OOM14/100GiB保护，0CUDA/物理/optimizer。该检查不更改运行预算、终态或参考，不能以审计通过称移动有效。发布部署后实际审计1500保存，执行结果待。

## 中间审计A元数据断言失败

审计入口发布`2b828a0d7afceb8fa902d667c7e3d749e44de072`已推送并独立核Q1，逐SHA部署。05:46:28审计1500原生退出2：十三门中十二门通过，只有动作声明整对象相等断言失败。实际训练manifest在注册声明上增加真实`runtime_optimizer_guards`三项和`runtime_policy_class`，与train_bounded源码一致；审计错误地要求无这些运行证据。不是训练数值失败，原训练继续1600新轮，不更改其状态。

A原件保留`intermediate1500_audit.json`，保存1500为205325095字节/SHA`2b006ce1ac0fb5cdebc4d2b69aba87cf3624715e5027b598b44dc0d5065282e4`。三网络/三Adam/两采样器有限，Adam计数19490→24980、1899→2398、37980→47960；516冻结不变，manifest与initial/admission完全相同、50Hz采样对应完整日志前缀，无CUDA/物理/optimizer。不能把A称全部通过。

修正审计按源码严格要求注册声明加上指定BoundedPolicy类和三个各一项的真实hook声明，不删除运行字段或放宽其内容。使用新`intermediate1500_audit_B.json`补检同一immutable保存，原A失败保留；执行结果待。

## 保存1500审计B及05:55活跃核验

修正发布`132af69bd50fd64a478e2aee20f4c381c34874d5`已独立核Q1远端、部署SHA一致。05:48:44新B审计原生退出0，十三门全过；同一保存1500、原spec/initial/admission及全部516冻结未变。实际BoundedPolicy/三个各一项的指定hook声明完整匹配，三网络已更新且三Adam计数正确，保存张量与采样器有限、日志前缀完整，无CUDA/物理/optimizer。[原样B收据](../data/manifests/supermimic_q1_R02D_intermediate1500_audit_B_20261025.json)4889字节/SHA`f443e4f78499fead84a2e24dda7442179f056bdfb549365413583124febb364a`；原A失败保留。

05:55:27[真实进度与身份](../data/manifests/supermimic_q1_R02D_progress_after_P11_20261025.json)重新核原boot/ticks/cmdline/进程组及三实际非Z进程，新增1825/7000轮、116800转移、累计6825；WM679组/actor629次、用时3476.78秒。actor梯度0.08102、critic loss0.013107、WM loss0.204920均有限；已越过原R02C1810失败位置，但不能推出固定预算必然完成、唯一因果或移动改善。

当次GPU40摄氏度/1228MiB、共享内存13735399424字节/OOM14、共享盘余784996777984字节。R02D分配821764096字节、P10/B/P11共9867264字节；共享盘余下降不全部归本任务，100GiB/12GiB/30GiB等原停止线不变。没有新训练/积分/optimizer由此次检查引入，唯一R02D继续原预算和七阶段配对。P10/B补充walking零准入及P11限速不足结论均已记录，不能由训练数值替代实际步行验收。

## 06:42保存3000独立审计通过

复用未改的已发布auditor，先确认实时3050新增轮且step3000文件完整，再独立CPU/mmap审计。06:42:32原生退出0，十三门全过；三网络已更新、三Adam/两采样器有限、计数19490至30980、1899至2998、37980至59960准确，原340/6/7、7000预算和全部516冻结保持。0CUDA/积分/策略/optimizer，不能作为跟随改善证据。

保存3000为205325095字节/SHA`edf972e0406b356a991cb8888e7baaad5fb092dbc20f60525f32d62ba879de06`；[原样审计](../data/manifests/supermimic_q1_R02D_intermediate3000_audit_20261025.json)SHA`a7acda2f6b9330f1c92edf03fd0d0df751dfab785fc1cdee62a40f168ad62197`。服务器原件实验根`intermediate3000_audit.json`，stdout同名logs目录`intermediate3000_audit.log`。结束内存14,931,582,976字节/OOM14，实验分配1,437,843,456字节、盘余541,350,809,600字节。共享盘下降继续监控，100GiB停止线不变；未删未知文件。

## 07:06继续原预算

[原生现场记录](../logs/session_records/supermimic_q1_R02D_after_P12C_20261025.log)重新核三实际非Z进程的原boot/start_ticks/组/cmdline及三个状态RUNNING，新增3700/7000轮、236800转移、累计8700轮；WM1429组/actor1379次，最新actor梯度0.12496、critic loss0.01623、WM loss0.27923有限。07:06共享内存13815267328字节/OOM14、GPU39摄氏度/1228MiB，盘余418098790400字节；共享盘其他写入持续，100GiB停止线保持，不删除未知资产。

并行的[P12C离线train参考诊断](README_SUPERMIMIC_Q1_WALKING_CONTACT_20261025.md)24组和独立重算完成，无新增合格参考，没有执行策略/积分或optimizer，也未改本轮340/6/7输入、7000新轮和终态规则。本轮仍待训练终态及既定七阶段353片真实动力学配对；数值有限及诊断可复算均不能替代实际移动效果。

## 07:20保存4000独立审计通过

实时已4075新轮后，复用未改已发布auditor对immutable step4000作CPU/mmap检查，07:20:49原生退出0，十三门全过。三网络、三Adam、两采样器有限；Adam计数34980/3398/67960正确，原340/6/7、7000新轮及全部516冻结保持。0CUDA/积分/策略/optimizer，尚无最终动力学效果。

保存4000为205325095字节/SHA`cdf5e259fe9c73dea58e8a86e9330750f1280f3cac73c0a21a2deb4506100f8c`；[原样审计](../data/manifests/supermimic_q1_R02D_intermediate4000_audit_20261025.json)SHA`625fb3ab2c62d463d2c1389af3b1a49e3af165991badf9fbde36223d95df9b40`。原件实验根`intermediate4000_audit.json`，stdout同名logs目录`intermediate4000_audit.log`。结束共享内存14929911808字节/OOM14、实验分配1848569856字节、盘余340030201856字节；100GiB停止线及原预算保持。
