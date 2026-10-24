# Q1从B128追加固定训练预算

记录日期：2026-10-23；2026-10-24补最终结果，北京时间。阶段：SM-Q1-R03。状态：固定预算训练、保存审计、完整配对评测及重算均完成；关节改善，但移动/跟随目标未达。
## 2026-10-24最终结果

本轮训练于北京时间10月24日06:08:22正常退出，08项流水线阶段全部原生退出0，06:11:04生成终态`CONTINUATION_PIPELINE_COMPLETED`。10月24日09:28:12独立读取冻结计划、全部结果/审计与事件文件，61项冻结条件通过，重新生成的summary与保存文件精确相等；每个阶段恰有一个退出0且结果大小/SHA对应的终态事件。这里只完成原固定预算，没有重启或追加。

### 训练与保存

- 原生训练循环24471.823941秒，约6小时47分52秒；追加12000采集轮/192000控制转移，累计15000轮/240000转移。
- 新增4749组WM与4699次actor更新；真实Adam计数WM58980、actor5798、critic115960，增量分别47490/4699/93980。
- 已保存的续训审计13项全部通过：8棵父状态树精确继承、固定预算/计数、三网络变化、有限数值、完整日志、无新增OOM等。本次复核读取保存审计，不重复重型checkpoint张量审计。
- final为205225747字节，SHA-256 `10d0838663b5d24215f493ce0738228597452e899b1a378df6d6b453b7e10ea7`。
- 09:28时原controller为僵尸Z、worker不存在；末次评测/分析子进程亦已结束。GPU38摄氏度、空闲17788MiB，容器25031303168字节、OOM14未增加。静态PID仅说明此次核验时的状态。

### 完整动力学评测

step0与final均重新执行三段完整动力学闭环，共各3487控制步，无提前失败。两份独立评测审计均verified=true，最终标量最大重算误差2.56e-8；审计通过不代表跟随门通过。另独立读取R02B终态，逐片确认R03初始评测的7个标量、参考身份、轨迹大小/SHA与父模型评测完全一致。

| 片段 | 完整步数 | 全身RMSE初始→最终(rad) | 腿部初始→最终(rad) | 上肢初始→最终(rad) | 归一化回报初始→最终 |
| --- | ---: | --- | --- | --- | --- |
| 走路 | 1652 | 0.363888→0.188527 | 0.178007→0.176143 | 0.559664→0.203159 | 0.720030→0.854335 |
| 浅蹲 | 1157 | 0.386939→0.201056 | 0.251199→0.189322 | 0.560773→0.214948 | 0.801088→0.953486 |
| 出拳(复用开发) | 678 | 0.311051→0.262525 | 0.110641→0.117430 | 0.489753→0.394347 | 0.750140→0.824271 |

三片全身误差分别下降48.19%、48.04%、15.60%，说明额外训练产生了实际关节跟随收益；单次固定续训带有fresh replay等限制，不把它表述成多种子或预算单因素的普遍因果结论。三片腰部误差均变大，出拳腿部也稍变差，不能仅凭全身均值宣称全身各部位都改善。

**仍未学会沿参考路径行走。** 走路参考最大水平离起点距离1.126480m，策略从0.030115m变为0.029120m；参考累计水平路径长2.439085m，策略路径长0.111093→0.615219m，但始终局限在约3cm范围。路径长增加主要反映局部活动，不能当成向前行走。全根位置RMSE 0.629545→0.645016m，走路腿部误差几乎不变，全身改善主要来自上肢。参考首尾净水平位移0.118145m，策略最终净位移0.009627m；净位移和最大离起点距离须区分。

原有三项可用门为true/true/**false**：完整不早停、平均完整片回报≥0.5通过，但各片全身RMSE≤0.15rad仍全部未达标。现有门本身也不能代替实际移动验收。出拳收据中的legacy `split=heldout`保持原文件，实际是被反复查看的开发数据，不能宣称独立未见测试；没有真机验证。

### 本次证据、恢复与保存

完整小证据为`data/manifests/supermimic_q1_R03_completion_verification_20261024.json`，包含12份原结果文档、大小/SHA、8项原生事件、冻结/summary重算和父评测一致性；读取程序与原生退出记录见`logs/session_records/supermimic_q1_R03_completion_read_20261024.log`。模型、NPZ及后续视频留服务器`LOCAL_ONLY`，没有大型Mac下载或网盘验证。

恢复观察时原18864转发已失效；新建同一路线本地SSH转发后读取到已完成流水线。观察断线没有被当成训练失败，没有恢复/重启原实验。认证仅临时交互使用，不写文件或Git。

09:52已完成[V02/R03实际回放](README_SUPERMIMIC_Q1_VISUAL_R03_20261024.md)：三段826/579/339帧、ffprobe/身份、九帧目视与浏览器播放通过；新入口18865，原18863仍为R01。画面支持走路仍近原地的结论。下一核查参考、奖励与世界模型动作响应后选择独立改进，不自动追加R03预算。`D05/D05B`根高候选依然无训练资格。A3保持停止，目标未完成。

本次记录依四目录规范落盘，提交号由Git历史提供；执行payload审计、显式push Q1并比对远端后再报告同步完成。

## 以下为启动与中期历史记录


## 目标与范围

继续用户授权的SuperMimic for Q1世界模型路线，检验增加学习预算能否得到实际移动与全身动作跟随。保留[R02完整对照](README_SUPERMIMIC_Q1_BATCH_TRAINING_20261023.md)终态，在独立`data/supermimic_for_q1/R03_B128_extend12000_20261023/`继承第二组最终模型，固定追加12000轮。原hp3090/A3不恢复，不操作机器人。

## 选择及依据

父臂`R02_cachefixed_pair_20261023/R02B_batch128`：三片完整、腿部误差更低，浅蹲和复用开发出拳漂移较小；A走路仅42.373%后失败。B上肢/全身误差较大，走路仍近原地、根水平误差0.629024m，不能称已得到可用策略或batch128在所有指标胜出。D03相同前缀比较及23:07真实父状态/18阶段/比较重算均已核验；单种子与开发数据复用限制保留。

新增预算12000轮/192000新控制转移，累计15000轮/240000训练控制转移，只有有限追加预算，没有自动增量、无限循环或失败恢复。预计本阶段4749组WM/4699次actor，实际以保存审计为准。B原6110.817秒/3000轮，追加预算预计约7至8小时；这是估计，不是到期停止条件或完成证据。

B最后两个在线窗口真实关节误差尚未改善，模型k8误差仍下降；增加预算只是待检验假设，不能排除模型偏差、控制/参考动态可行性等因素。结束后以相同完整动力学评测及独立重算判断，不按在线回报宣布成功。

## 固定配置与初始化

继承父WM/actor/critic、三组Adam动量与计数、return-bin采样统计；保持16环境、batch128、WM64×10、k8/H32、学习率原cosine累计偏移3000、PD50/2、模型25%力矩、`Q1_FOOT_PRIORITY_V2`与相同走路/浅蹲训练素材。初始/final评测仍含出拳复用开发片段。

原检查点没有完整replay/仿真进程状态，本阶段显式新seed20261024、新replay/episode，并保留128采集轮和50组模型预热。step0在新episode重置之前保存父网络/Adam/bin状态；这不是逐步无缝续跑。只把模型内推演反传用于策略学习，不加入教师、行为克隆或A3权重。

## 版本、平台与命令

代码副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`。服务器`huanyu:/opt/data/private/YuanQi_Q1`，Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。核心冻结`6b43cc0`；续训代码发布`d5f9476125f2c3c7ed2d7c1e6ac898e1f69d6429`及61项release已核验，后续D03分析代码位于独立目录，不改变这份发布清单。

父final205221971字节/SHA`b2990bc9af730127e5bda6d88fe1f3b214980e061e3badade555eedd02c51095`；release服务器`logs/supermimic_q1_extension_release_E03_20261023.json`，11363字节/SHA`c141ff85ecde79ac83bcde4edda7fe326e2f22cf5c8cf5932da718eb19386454`。当前主机boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，已重查RTX3090 24576MiB/驱动570.153.02；23:07容器26910662656字节、空闲17788MiB、GPU39摄氏度、OOM14。旧训练无活跃进程；启动仍由登记入口重查输入、运行时、源码和资源。

父选择/预算记录已发布`0cd14de5a07cf33eab5a59a3f3b8226654bff57f`，独立核对Q1远端一致后，下列登记/执行已实际各调用一次；当前正在运行，不能重复现有输出：

```bash
$PY script/supermimic_q1_extension/pipeline.py register \
  --root "$PWD" --release logs/supermimic_q1_extension_release_E03_20261023.json \
  --pair data/supermimic_for_q1/R02_cachefixed_pair_20261023 \
  --parent-run data/supermimic_for_q1/R02_cachefixed_pair_20261023/R02B_batch128 \
  --output data/supermimic_for_q1/R03_B128_extend12000_20261023 \
  --steps 12000 --seed 20261024 \
  --reason 'B completes all three clips with lower leg error and less stationary-clip drift; extend budget to test learning while preserving the audited control and model contract.'
$PY -u script/supermimic_q1_extension/pipeline.py execute \
  --plan data/supermimic_for_q1/R03_B128_extend12000_20261023/plan.json
```

实际执行使用脱离SSH的`subprocess.Popen`，stdin为DEVNULL、stdout/stderr到该独立实验的controller.log；原生控制器依次执行训练、续训保存审计及step0/final三片整段物理评测/重算/分肢诊断。GPU86摄氏度、可用显存2GiB、容器30GiB、新OOM等保护继续；不改变其他六个Python进程。

## 结果、保存与下一步

真实计划37578字节/SHA`fa92a82c7bb0e4cd8a3ba7e23636f00cbae6a72dd145bf28922b70155fe7ce9d`，Git小副本`data/manifests/supermimic_q1_R03_plan_20261023.json`。23:20:07由脱离SSH的独立session启动controller762428/start3606081278，随后训练worker762431/start3606081380，同boot；原生启动记录`native_launch.json`，小副本`supermimic_q1_R03_launch_20261023.json`。日志`logs/supermimic_q1_R03_B128_extend12000_20261023/{controller,train}.log`，输出run/admission、step0与status已实际生成。

23:22:40独立CPU读取父final与本轮step0，三个网络、三组完整Adam和bin_return/bin_memory共8项树逐张量精确一致；step0本地计数全0、累计3000，学习率偏移3000，初始Adam计数11490/1099/21980未清零，配置合同和admission/初始manifest一致。step0为205233831字节/SHA`eca79942bef1530d8aa679a8f02f38540046991271450220483efadaef61e006`。这是实际Q1父模型的继承正例，0新增物理/推理/optimizer；不能替代未来完整保存审计或跟随效果。

实际核验中原controller/worker均非僵尸、boot/PID/start_ticks/命令匹配，全部冻结源码/输入/运行时通过。训练快照250/12000轮、4000新交互、49组WM/0本阶段actor，累计3250轮；当时仍处原50组模型预热。GPU40摄氏度/空闲16671MiB、容器31511310336字节（含独立CPU读检查点开销）、OOM仍14。收据`launch_verification.json`3088字节/SHA`7d34f4541a89575066e5c1145be9f6e4df5b73b1fc0177c223a7a9fb0d704a8c`，小副本`data/manifests/supermimic_q1_R03_initial_retention_20261023.json`；admission26749字节/SHA`a94e3f9dfc5d4fe80a92b132cc6dc7b25c3068275b586dc0f1dcbfadca098339`也保存小副本。未将在线关节误差或模型损失当作最终成绩。

结束后同时核对0.15rad等既有门、腿部与水平根轨迹、实际/参考位移和画面，不把流程成功或站住当成移动跟随。此新阶段尚未完成，不能调用只适用于final的audit_run把启动误记为全流程成功。

代码/计划清单/小证据按Q1分支同步GitHub，先审查diff与payload；大型模型与轨迹仅服务器LOCAL_ONLY。恢复先读本记录与新目录status/started/日志，重新核对进程身份；无自动恢复，不重复注册或启动。

## 23:41第500轮保存状态核验

23:41:35独立CPU进程以`torch.load(map_location="cpu", mmap=True, weights_only=False)`实际读取step0与`run/step_000500.pt`，未创建网络实例、调用推理或推进物理/optimizer。第500轮模型205233831字节/SHA`8a91c026d43c5bb8048823969d60f58c905748108ae3d3ba2c3e3bfd1cd447d9`；读取前后大小/SHA相同，manifest与admission/step0一致。绑定真实plan、native_launch及23:22父状态继承收据，14项检查全部通过。

使用独立整数公式`(500-128+1)*2//5=149`计算WM组数，扣除50组预热得到99次actor更新；checkpoint本地500轮/8000新交互、累计3500轮及实际更新计数精确一致。

| Adam状态 | 继承初始计数 | 本阶段新增 | 第500轮实际计数 | 参数状态数 |
| --- | --- | --- | --- | --- |
| WM | 11490 | 1490 | 12980 | 14 |
| policy | 1099 | 99 | 1198 | 20 |
| critic | 21980 | 1980 | 23960 | 6 |

每个参数的Adam状态键、参数映射及计数逐一核对；三个网络的键/shape/dtype相同、数值有限且相对step0均已变化，最大绝对变化WM0.0623695、policy0.0161643、critic0.2215345。完整优化器/采样统计有限；前500轮共20条指标按25轮间隔齐全且有限，OOM计数仍14。源码、输入和运行时重新核验通过，未修改冻结的两个训练源码目录。

当时实时训练800/12000轮、12800新交互、269组WM/219次actor，累计3800轮；原controller762428/start3606081278与worker762431/start3606081380同boot且非僵尸，实际argv分别与native_launch/登记训练阶段精确匹配。GPU39至40摄氏度/空闲16671MiB，CPU核验后容器31738470400字节，低于30GiB保护线32212254720字节；诊断期间未新增OOM，未停止或重启训练。临时SSH重连的首个观察会话无响应，取消该观察会话后同路线重连成功；未把观察失败当作训练终止。

服务器证据`checkpoint500_verification.json`6517字节/SHA`f7822f15d58564b25f8bb197a05a64836039f557886527512f936ee3c8b68720`，Git小副本`data/manifests/supermimic_q1_R03_checkpoint500_20261023.json`。只取回小JSON，checkpoint仍留服务器LOCAL_ONLY。这项核验确认真实继承后的训练更新，不能证明跟随效果；仍等待原12000轮训练、完整保存审计、初始/final三片动力学评测与独立重算，不追加预算或用在线回报宣布完成。

## 2026-10-24中期在线日志诊断

[D06只读趋势诊断](README_SUPERMIMIC_Q1_BUDGET_TREND_20261024.md)已完成：读取R02B前3000轮及R03前5750轮的350条完整日志，原始前缀SHA独立核对一致。R03早期751–1500与后期5001–5750各30条单步采样，平均关节误差0.376099→0.250112rad，在线奖励0.070683→0.083835；150条选定标量记录的45个均值由独立算法重算，最大差1.11e-16。参考相位、状态与策略分布变化，不能视为整片改善、移动通过或预算的匹配因果证明；最终完整评测仍待完成。

02:38:46原controller/worker的boot/PID/start_ticks匹配、均非僵尸；追加5975/12000轮、95600新交互、累计8975轮，2339组WM/2289次actor。冻结源码/输入/运行时通过，GPU40摄氏度、空闲16671MiB、容器31070232576字节、OOM14，无终态收据。没有新增诊断物理/推理/NN更新，原R03按既定预算继续，D05B候选未接入。
