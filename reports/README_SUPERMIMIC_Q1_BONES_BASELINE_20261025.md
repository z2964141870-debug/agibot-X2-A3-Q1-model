# BONES到Q1的首版SuperMimic复现基线

记录日期：2026-10-25，北京时间。阶段：BONES-G02 / PF02 / RF02。状态：PF02的1024段原速参考库已完成，RF02训练进行中，初始/最终动力学比较尚未执行。

## 目标与用户决定

用户要求重新定目标并继续执行，优先复刻SuperMimic，不再追加代理制定的数据质量门槛。新目标是利用现有1024个冻结train来源形成原速Q1参考库，完成64环境、12000采集轮/768000真实转移的独立训练，交付同一固定参考上的初始/最终MuJoCo动力学比较及回放视频。1024和12000是本次可执行数据/计算预算，不是质量准入阈值，也不代表原论文全库或算力规模。

用户最新限制最多一个子agent；主代理处理数据、训练、记录与部署。评测/回放子任务已完成；后续唯一只读方法审阅也已结束，无子agent继续运行，旧多个子任务不再调度。

## 数据与方法

复用已完成PF01的128段/791.32秒，不重新转换或复制其数组。冻结名单前1024条全部train，后256条为development128/test128；已按offset128至896、每批128完成余896个来源转换，再通过软链接合库。各片只检查可读、Q1维度/身体顺序、原速50Hz时间轴与有限数值；原几何flags保留为诊断，统一2rad/s及旧19/22门不使用。

训练沿用RF01已跑通的可微世界模型、8步拟合、H32想象BPTT与critic，64环境/4模型槽、actor B128、WM B64每组10批。三网络重新随机初始化，固定12000采集轮，每1000轮保存中间模型，保留初始/最终。Q1形态、CPU MuJoCo与单3090参数是工程适配，不能称官方G1逐行复刻。

初始/最终使用同一库按clip名排序的前8段train，固定选择后才运行策略，统计实际误差、回报、完成比例、提前终止和根移动，不设置新的“必须达标才继续”门。该比较衡量训练内覆盖，不代表未见动作泛化。回放来自实际自由根动力学轨迹，不能用运动学播放冒充策略表现。

## 平台、代码与资源

唯一实体根`huanyu:/opt/data/private/ZHY`，四目录和Q1分支保持。Python为`data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python`。原boot仍d994e71f-bae0-46ba-a5f7-6c2163771c6c、OOM14；本次恢复18864跳板转发后实测盘余2093725360128字节、约2.094TB，容器内存23512776704字节。GPU在本轮首次核验27摄氏度/14MiB空闲任务显存，无现行训练进程；恢复时须重新读实时状态。

代码添加于`script/supermimic_q1_reproduction/`；保持原PF01/RF01四个源码不改，只新增多批合库、串行流程及新IO评测入口。`merge_library.py`复核原批IO协议和数组身份，仅建立新链接，保留旧false flags；合库不增加几何判定。`pipeline_baseline.py`依次准备7批、合库、12000轮训练、保存核对、初始/最终真实动力学比较与同一片回放。状态持续保存当前阶段、PID/boot/start_ticks、命令、日志和返回码；失败不自动重启，停止只通知自己的子进程，由原guard回收训练组。支持`--detach`，SSH断开不终止长任务。

合库2项服务器控制通过：旧false标记保留且只建数组链接；重复批在建输出前被识别。流程异常控制在Mac/服务器均通过：首阶段退出7时真实记录FAILED，后续阶段未启动、没有重试。4份主代理新增源码部署前后SHA逐一一致，合库源码7a3ea8e610e9de726df0be98cfa2b2402adffcc375cd654d7c4b4955fa4b8c40，流程源码7641488c3929b4813262fdbdcce8c707fc59be7d1049a758a944ffa5241079cf。

评测新入口不再依赖旧qualified flags或效果门；绑定训练一致的接触资产、clip20策略与原迁移视图。服务器3项评测控制与语法检查通过。实际执行RF01前2片初始/最终CPU评测及指标重算均退出0，结果未见整体改善：

| 固定前两片均值 | 随机初始策略 | RF01最终策略 |
|---|---:|---:|
| 归一化整片回报 | 0.27296 | 0.14400 |
| 整片完成比例 | 62.03% | 32.49% |
| 提前终止 | 1/2 | 2/2 |
| 执行段关节RMSE | 0.44609rad | 0.44551rad |
| 根平面位移 | 0.24135m | 0.47935m |

第二片最终策略虽然移动约0.480m，但108/282步提前终止，不能称有效步行。第一片最终视频实际查看末帧：参考直立行走、策略倒地，清晰标注71/266步提前终止，无空白或文字重叠。视频`data/supermimic_for_q1/BONES_RF01_E01_20261025/final_first_video/comparison.mp4`，126120字节、H264/1280×548/25fps/36帧/1.44秒，SHA`2b935c38e8d9f4b2138edf14067f6bbb7416f3ef7c67f3d4fb70af11cb81177d`，像素标准差最小73.732。上述仅覆盖500轮小试的两片训练动作，不是12k轮或1024库结果，也不据此暂停扩库/训练。

五份原样收据已从服务器取回至`data/manifests/`，本次逐一复核SHA与服务器一致：

| 原样证据 | SHA-256 |
|---|---|
| [初始评测](../data/manifests/supermimic_q1_BONES_RF01_E01_initial_20261025.json) | `e09c1f92c95d460a6074e1979a565f0a4f0a3ee31d02595b8457b8e9d28ef54d` |
| [初始算术复算](../data/manifests/supermimic_q1_BONES_RF01_E01_initial_audit_20261025.json) | `2dc2963a7e72fb561d2922cc923cfa913a93c9c84abc70703d4725640506345c` |
| [最终评测](../data/manifests/supermimic_q1_BONES_RF01_E01_final_20261025.json) | `fab013277b4a0d163b55102e0c2b1f149ae24f0eb9745a01fd03eeec8cea841d` |
| [最终算术复算](../data/manifests/supermimic_q1_BONES_RF01_E01_final_audit_20261025.json) | `cba6e022fccd3a0eb25032bab140dc01dc9b8c0ec566d595db64da22a16e1e61` |
| [最终首片回放](../data/manifests/supermimic_q1_BONES_RF01_E01_render_20261025.json) | `c0bef1a49e6fbf610336c0ff11407297c118947853c9544fd8e86efccecd336e` |

E01真实输出根`data/supermimic_for_q1/BONES_RF01_E01_20261025/`；执行日志`logs/BONES_RF01_E01_{initial,final,initial_audit,final_audit,final_first_render}_20261025.log`。评测使用`evaluate_reproduction.py --root /opt/data/private/ZHY --motions /opt/data/private/ZHY/data/supermimic_for_q1/BONES_PF01_128_20261025/library/references --checkpoint <RF01/run/step_000000.pt或final.pt> --output <新评测目录> --device cpu --count 2`，再以`audit_reproduction_evaluation.py --root /opt/data/private/ZHY --evaluation <新评测目录> --output <新评测目录>/audit.json`复算；新目录要求用于防止覆盖原证据。

实际启动命令（13:38:56北京时间执行）：

```bash
ROOT=/opt/data/private/ZHY
PY=$ROOT/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -u "$ROOT/script/supermimic_q1_reproduction/pipeline_baseline.py" \
  --root "$ROOT" --intake "$ROOT/data/supermimic_for_q1/BONES_A02_received_20261025/intake/receipt.json" \
  --backend-receipt "$ROOT/data/supermimic_for_q1/PHUMA_P05_slots_V01_20261024/receipt.json" \
  --code-commit d66ac1cd21de1682ca6f069227c070ebf98da02f --expected-oom-kill 14 --detach
```

新增代码、测试、阶段报告与五份原样收据已发布`d66ac1cd21de1682ca6f069227c070ebf98da02f`，独立`git ls-remote origin refs/heads/Q1`与本地HEAD完全一致；17文件/1035808字节的暂存payload检查及diff检查通过。服务器全部12个reproduction源码SHA与发布前本地逐一一致；原PF01/RF01源码未改，旧下载器仍不执行、不提交、不删除。

实际启动身份：[原样launch](../data/manifests/supermimic_q1_BONES_G02_launch_20261025.json)，控制器PID861470/start_ticks3619874227、boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`；首次prepare包装进程PID861471/start_ticks3619874236、worker861474。启动后实际读取`/proc`确认控制器与包装进程存活且start_ticks匹配，状态为RUNNING/prepare_0128，不能仅用LAUNCHED收据推断运行。首批进度实读42/128、65.85秒，guard为RUNNING、OOM14，盘余2092641165312字节、共享内存24778801152字节；RF02尚未开始。

[原样plan](../data/manifests/supermimic_q1_BONES_G02_plan_20261025.json)和[初次状态快照](../data/manifests/supermimic_q1_BONES_G02_launch_status_20261025.json)已取回。快照是启动观测，不代表恢复时实时状态；须再读服务器`status.json`及当前阶段guard/progress、匹配boot/PID/start_ticks。控制器日志`logs/BONES_BASELINE_G02_20261025_controller.log`；转换实际worker日志`logs/BONES_PF02_0128_20261025.log`，包装进程日志见status事件。发生SSH超时先重读状态，不重新启动旧输出。

保留100GiB余盘、单实验12GiB和温度/内存/OOM保护，控制模型保存量；不全量解压原包、不复制23.5GB归档、不把模型拉到Mac。原PHUMA/A3停止或失败实验不重启、不改终态，大型资产均LOCAL_ONLY。

## PF02完整合库结果

14:02:39已读到7批全部退出0并进入merge，14:02:53实际进入train。逐批结果如下，每批均128/128可用、转换错误0、IO错误0，guard终态WORKER_COMPLETED；前四批完成worker的`/proc`路径已实读消失。

| offset | 帧数 | 原速秒数 | 批次分配字节 |
|---|---:|---:|---:|
| 128 | 40174 | 800.92 | 144306176 |
| 256 | 42766 | 852.76 | 153427968 |
| 384 | 40419 | 805.82 | 145235968 |
| 512 | 37534 | 748.12 | 135094272 |
| 640 | 43153 | 860.50 | 154804224 |
| 768 | 43640 | 870.24 | 156528640 |
| 896 | 39503 | 787.50 | 141955072 |

新7批总分配1031352320字节，复用PF01的128段，不重新复制原数组。合库实际1024/1024段、326883帧、6517.18秒=108.62分钟；1024个`.npz`均为指向原批数组的链接。独立读取`BONES_P01_metadata_20261024/selected_candidates.jsonl`前1024条，按`source_group`集合比较合库`motions[].bones_source.source_group`：期望/实际唯一来源均1024、缺项0、额外0、split全部train。source_hz集合仅120、target_hz仅50、time_scale仅1；未引入dev/test来源。

合库保留1024个原`qualified_for_training_pilot=false`，同时按新IO协议准入1024个，`local_geometry_gates_used_for_admission=false`；旧几何标签不是本次训练条件。[合库原样准入收据](../data/manifests/supermimic_q1_BONES_PF02_1024_admission_20261025.json)SHA`557fb8725a1ed605f703b33e53402105a58b704a32e06dd579d8ebff92c9693d`，绑定服务器原manifest及其SHA。各批原样收据和资源guard保存为`data/manifests/supermimic_q1_BONES_PF02_{offset:04d}_{admission,guard}_20261025.json`。14份批次终态、合库收据及训练manifest共16份静态文件已独立逐SHA比较服务器与本地，覆盖一致、无差异；另外保存的进行中状态均标为观察快照。

## RF02实际启动与恢复

训练事件于14:02:53启动，包装PID862406/start_ticks3620017857、worker862408/start_ticks3620018050；控制器861470/start_ticks3619874227、boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`保持。14:05:27实际读取三进程`/proc`均存在且状态S，训练status为RUNNING/250采集轮/16000真实转移、世界模型49组/actor0次；actor仍在50组模型预热阶段，不把它写成已完成BPTT。初始`step_000000.pt`实存69122731字节，未拉取模型到Mac。

[训练启动manifest](../data/manifests/supermimic_q1_BONES_RF02_admission_20261025.json)实读train_clips1024、steps12000、num_envs64、WM窗口8、H32、actor B128、WM B64每组10批、seed2026102513、每1000轮保存；三个网络从随机初始化开始，code_commit为d66ac1c。启动[控制器快照](../data/manifests/supermimic_q1_BONES_G02_training_start_status_20261025.json)及[worker进度快照](../data/manifests/supermimic_q1_BONES_RF02_start_status_20261025.json)已取回，均是当时观察，不代表之后实时状态。

当次状态采样GPU39摄氏度、容器内存27573526528字节/OOM14；guard观测盘余2091584765952字节、训练分配69902336字节、共享内存27692744704字节，未触发保护。完整12000轮、参数保存核对、固定8片真实初始/最终动力学比较与回放仍待完成，目标保持active；扩库成功与早期更新不代表跟随效果改善。

后续取回的[早期更新快照](../data/manifests/supermimic_q1_BONES_RF02_early_update_status_20261025.json)为350轮/22400真实转移、WM89组/actor39次，`policy_grad_norm=1.0185585022`、`critic_loss=4.1829929352`，H32想象BPTT及critic已实际进入更新，数值有限；训练计时321.26秒、GPU40摄氏度、容器内存27873009664字节/OOM14。另保存[启动阶段guard快照](../data/manifests/supermimic_q1_BONES_RF02_start_guard_20261025.json)，两者是原样观察，恢复仍以服务器实时状态为准；模型张量和Adam计数的保存核对在产生相应checkpoint后执行。

## 中间保存核对入口

新增`script/project_tools/audit_q1_checkpoint.py`，以独立CPU进程读取初始和指定保存，核对参数形状/真实变化/有限性、三Adam计数、训练manifest、保存前缀日志、输入与训练源码身份；只输出证据，0新动力积分/0新optimizer更新。它位于独立工具目录，当前RF02训练源码保持冻结。核对前保留28.75GiB共享内存余量边界，过程中29.5GiB及OOM14保护，不创建CUDA上下文。

14:17:19已用RF01已完成500轮作为真实控制，输出[原样控制收据](../data/manifests/supermimic_q1_BONES_RF01_checkpoint_tool_control_20261025.json)，状态CHECKPOINT_EVIDENCE_PASSED、13项检查全过。WM/policy/critic变化张量数14/20/6、最大差0.2534199357/0.0123939831/0.3602191806，三Adam1490/99/1980；与原RF01独立保存核对的计数和变化逐项一致。耗时3.22秒、最高采样共享内存26917109760字节/OOM14。该控制只验证核对入口，不是RF02中间保存结论。

工具SHA`ecdb504efd54eec6bf319585c474b70f1585b2a8d7a9e1c8697250f4a3653425`，控制收据SHA`e054b7e81f976ff6c45f6fcd2c7a7fc3a1ac630f39389e409495b7b447a6ae29`；本地和服务器已独立逐SHA一致。工具及控制发布`da8b7d443f350163bef878f02da7db4fa658536d`，独立核对远端Q1一致；RF02训练代码身份仍为d66ac1c。控制实际命令是以下命令中run改为`BONES_RF01_500_20261025/run`、checkpoint改`final.pt`、output改`BONES_BASELINE_G02_20261025/checkpoint_tool_RF01_control.json`。以下RF02命令已在1000保存实际出现后执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/data/private/ZHY/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python \
  /opt/data/private/ZHY/script/project_tools/audit_q1_checkpoint.py \
  --root /opt/data/private/ZHY \
  --run /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF02_12000_20261025/run \
  --checkpoint step_001000.pt \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF02_12000_20261025/checkpoint_1000_audit.json \
  --expected-oom-kill 14
```

## RF02第1000轮保存实证

14:30:18实读RUNNING/1000轮/64000真实转移、WM349组/actor299次，worker862408/start_ticks3620018050与原boot匹配，训练计时1613.13秒=26.89分钟；实际`step_001000.pt`已经通过原子保存出现。14:30:22独立CPU核对完成，[原样收据](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_1000_audit_20261025.json)状态CHECKPOINT_EVIDENCE_PASSED、13项全过，真实保存结果：

| 网络 / Adam | 变化张量数 | 最大绝对变化 | Adam实际步数 |
|---|---:|---:|---:|
| 世界模型 | 14 | 0.3303254545 | 3490 |
| actor | 20 | 0.0306657720 | 299 |
| critic | 6 | 0.8715044856 | 5980 |

三网络和optimizer张量有限，初始Adam为空，manifest/1024输入/训练源码身份保持，25至1000每25轮日志前缀完整；0新动力积分/0新更新/无CUDA上下文。核对耗时4.07秒，容器共享内存采样最高27398664192字节，OOM14保持。收据SHA`2bc311dfb6658848e7fce5f10a00aafcd6f8028434198fa55d3074257e73bc75`，已从服务器原样取回并独立逐SHA一致。

初始`step_000000.pt`69122731字节/SHA`262d007e7b6cd4fd2ae25eafd7ce5a6dfb9da2cf4656373d83798c6a68846ae0`；1000保存205920167字节/SHA`2078ea8d4c357fc0cf63d9a7c453fbbe5f1010917c998057dfccf0d9f6fa2aed`。权重只在服务器保留，LOCAL_ONLY，Mac/Git仅取小收据。

14:31:41再读训练已继续1025轮、guard RUNNING；训练目录分配275877888字节、盘余2091288834048字节、共享内存27088363520字节/OOM14。1000保存实证是中间进度，12000固定预算及之后的固定8片初始/最终真实动力学比较、回放仍未完成；没有以中间核对替代全目标，也没有新增效果达标门槛。

## RF02第2000轮自动保存

15:04:04只读观察到2000轮/128000真实转移、WM749组/actor699次，控制器/包装/worker三进程的boot与start_ticks均匹配，pipeline及guard均RUNNING。实际`run/step_002000.pt`为205920167字节，mtime为15:03:18，完整SHA`92f161d48347fdeed7e3df388395be1dc2e7539ee8f15ac94d71e5ff4d68eb57`；只登记文件身份，没有重复1000轮的CPU张量/Adam审计，也没有将身份登记写成参数审计通过。

15:04取回的[保存后原样状态](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_2000_status_20261025.json)已继续2025轮/129600转移、WM759组/actor709次，训练计时3649.68秒、GPU40摄氏度、容器内存27109081088字节/OOM14。训练目录实测分配481861632字节，2000轮观察时磁盘可用2089501077504字节。权重保留服务器LOCAL_ONLY，Mac仅取小状态文件。

该次有界观察进程正常结束，训练控制器独立继续；没有重启或追加另一轮训练。恢复时读服务器实时status/guard并核对身份，接着完成原定12000轮及既有评测/回放流程，完整目标仍未完成。

## RF02后续自动保存登记

15:36:58有界只读观察到3000轮/192000真实转移，三进程boot/start_ticks匹配，pipeline/guard RUNNING，GPU40摄氏度、内存26462679040字节/OOM14、盘余2086684401664字节。下表只登记实际文件身份，没有重复1000轮的CPU张量/Adam审计。

| 保存 | 字节数 | SHA-256 | 保存时WM组/actor次数 |
| --- | --- | --- | --- |
| `run/step_003000.pt` | 205920167 | `1b15a4246597fb92bda1422862f26b904abdfa5518d3075e0e6486badda91bae` | 1149 / 1099 |
| `run/step_004000.pt` | 205920167 | `08462bfd6dac6e33c6573def39d90db6d41f2c1c1ba11547a285904548588ae1` | 1549 / 1499 |
| `run/step_005000.pt` | 205920167 | `3d0c95a3b5573b2aa788100611d6e29eec46df58cacf70d7d574d4937934f39b` | 1949 / 1899 |
| `run/step_006000.pt` | 205920167 | `f4462ce221ea646b73ad9ad38782b6d647d50b02e43fac0c9ed578b4a7e73f0d` | 2349 / 2299 |
| `run/step_007000.pt` | 205920167 | `6cd192ee69347d84e073481368750250265f4cce311d0fb6e2e675b787291d33` | 2749 / 2699 |
| `run/step_008000.pt` | 205920167 | `a0162ba19ffbb6182fd39bc2930ef0b26b2be372d85f137fa3f803f8a044c976` | 3149 / 3099 |
| `run/step_009000.pt` | 205920167 | `3fb641bb6f2d7840c09064156e89a87847c9fd0552ea97b3a28f22e8c9446f44` | 3549 / 3499 |
| `run/step_010000.pt` | 205920167 | `34a571e7ac6eec767fdee9172a4d1dbc7db17cfd4addfd2b7ab0cb8df55d43fc` | 3949 / 3899 |

3000模型mtime为15:36:37，训练目录实际分配687841280字节。15:38取回的[原样保存后状态](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_3000_status_20261025.json)已继续3050轮/195200真实转移、WM1169组/actor1119次、RUNNING，训练计时5697.48秒、GPU40摄氏度、内存26439262208字节/OOM14。权重保留服务器LOCAL_ONLY，Mac只取小状态；本次观察会话正常结束，原训练控制器继续。

当前完成本轮采集预算的四分之一，尚未完成12000轮或固定8片动力学比较/回放，不以训练日志推断跟随效果。未更改训练配置、源码或准入条件。

16:09:54有界观察实际达到4000轮/256000转移、WM1549组/actor1499次，三训练进程身份匹配、RUNNING/OOM14；观察随后正常退出，未中断独立训练。4000保存实际完整SHA/字节数见上表及16:27[原样附件观测](../data/manifests/supermimic_q1_BONES_RF02_dashboard_attachment_20261025.json)，仍仅登记文件身份，不重复张量核对。该观测已继续4525轮/289600转移，四进程（含新增绘图回调）身份保持、68训练源码SHA无变化，训练分配893853696字节、盘余2077460029440字节。

用户新增[每轮六宫格图要求](README_SUPERMIMIC_Q1_TRAINING_DASHBOARD_20261025.md)已落实至独立工具和本轮唯一终后回调，16:25:54启动，日志`logs/BONES_RF02_training_dashboard_20261025.log`；它不更改本轮训练。4325轮预览已真实生成/目视核验，尚非最终曲线；回合级前三项历史未记录，不能重建，下一轮新入口补齐记录。终后仍要完成原定固定8片物理比较、真实回放与最终图检查，不能以绘图准备代替完整目标。

16:43:38有界观察实际达到5000轮/320000真实转移，WM1949组/actor1899次，训练及绘图回调四进程身份匹配，GPU40摄氏度/OOM14。5000保存mtime为16:42:55，文件大小与完整SHA已登记上表；本次只核文件身份和日志轮数，不读取权重张量，不重复1000轮的独立CPU参数/Adam审计。

16:45:36[原样5000保存后观测](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_5000_observation_20261025.json)已继续5075轮/324800转移、WM1979组/actor1929次，pipeline/guard/training均RUNNING，四进程boot/start_ticks匹配。训练计时9724.19秒，GPU40摄氏度、共享内存25298948096字节/OOM14；训练目录分配1099816960字节、盘余2076855734272字节。权重留服务器LOCAL_ONLY，Mac仅取小观测收据。有界观察正常结束，训练与唯一绘图回调独立继续，完整12000轮/固定8片评测及真实回放仍未完成。

17:16:21有界观察实际达到6000轮/384000真实转移、WM2349组/actor2299次，四进程boot/start_ticks匹配，GPU40摄氏度/OOM14，完成本次采集预算一半。6000保存mtime17:16:04，205920167字节及完整SHA如上；仅核文件身份和对应日志，没有重复参数/Adam审计或新增评测。

17:18:15[原样6000保存后观测](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_6000_observation_20261025.json)已继续6050轮/387200转移、WM2369组/actor2319次，pipeline/guard/training均RUNNING。四进程身份保持，内存25404534784字节/OOM14，训练目录分配1305796608字节、盘余2076433354752字节；nvidia-smi实际GPU为RTX3090，采样39摄氏度/1544MiB显存。`final.pt`实际尚不存在，绘图回调日志继续等待6025/6050轮。小收据取回并核SHA，权重留服务器LOCAL_ONLY；观察正常结束，原训练继续，完整12000/固定8片物理比较/真实回放仍待。

17:49:48有界观察达到7000轮/448000真实转移、WM2749组/actor2699次，四进程boot/start_ticks匹配，GPU41摄氏度/OOM14。保存mtime17:49:27，实际205920167字节、完整SHA已登记上表；本次仍仅核文件身份，不重复参数/Adam审计。

17:52:13[原样7000保存后观测](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_7000_observation_20261025.json)已继续7075轮/452800转移、WM2779组/actor2729次，pipeline/guard/training均RUNNING。重新逐SHA核68份训练源码，变化0，训练版本仍d66ac1c；四进程身份保持、GPU41摄氏度、内存25611862016字节/OOM14。训练分配1511780352字节、盘余2068079329280字节，`final.pt`仍不存在，唯一绘图回调仍等待。观测会话正常结束，未重启/改配置；完整12000轮、固定8片实际比较及回放未完成。

18:22:53有界观察达到8000轮/512000真实转移、WM3149组/actor3099次，完成本次采集预算三分之二；实际保存mtime18:22:50，205920167字节及完整SHA已登记上表。本次仅核文件身份及对应日志，没有重复参数/Adam审计。四进程boot/start_ticks匹配、GPU41摄氏度/OOM14，观察随后正常退出，原训练和绘图回调独立继续。

18:23:53[原样8000保存后观测](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_8000_observation_20261025.json)已继续8025轮/513600真实转移、WM3159组/actor3109次，pipeline/guard/training均RUNNING；内存25777135616字节/OOM14，训练目录分配1717760000字节、盘余2067066073088字节。`final.pt`仍不存在，唯一绘图回调等待8000轮。训练源码没有本次重哈希，最近68份逐SHA核验仍为17:52的7000观测；没有修改训练源码或配置。权重留服务器LOCAL_ONLY，Mac/Git只取小收据，完整12000轮、同8片实际比较、最终图和真实回放均未完成。

18:56:27有界只读观察达到9000轮/576000真实转移、WM3549组/actor3499次，完成本次预算75%；保存mtime18:56:02，实际205920167字节及完整SHA见上表。仅登记文件身份，不重复中间参数/Adam审计。四进程boot/start_ticks匹配、GPU41摄氏度/OOM14；观察正常退出，训练没有重启。

18:58:18[9000保存后观测收据](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_9000_observation_20261025.json)已继续9050轮/579200真实转移、WM3569组/actor3519次，pipeline/guard/training均RUNNING；重新逐SHA核对68份训练源码，变化0、code_commit仍d66ac1c。内存25902755840字节/OOM14，训练目录分配1923743744字节、盘余2066297643008字节，`final.pt`仍不存在，唯一绘图回调等待9050轮。没有修改训练配置、来源或评测选择；权重留服务器LOCAL_ONLY，完整12000轮、固定8片动力学比较、最终图及真实回放均未完成。

19:29:17有界只读观察达到10000轮/640000真实转移、WM3949组/actor3899次，完成本次预算83.33%；保存mtime19:29:06，205920167字节及完整SHA见上表。本次只登记文件身份，没有重复参数/Adam审计。四进程boot/start_ticks匹配，观察正常结束，训练与绘图回调独立继续。

19:33:05[10000保存后观测收据](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_10000_observation_20261025.json)已继续10100轮/646400真实转移、WM3989组/actor3939次，pipeline/guard/training均RUNNING。实测GPU0为NVIDIA GeForce RTX3090、41摄氏度、1544MiB显存；内存25714364416字节/OOM14，训练目录分配2129735680字节、盘余2065182425088字节。源码本次未重复重哈希，最近68份逐SHA核验为18:58的9000观测，训练版本保持d66ac1c；没有修改训练代码或配置。`final.pt`仍不存在，唯一绘图回调等待10100轮，权重保留服务器LOCAL_ONLY；完整目标未完成。

## RF02核心链路只读审阅

[RF02-M01审阅](README_SUPERMIMIC_Q1_RF02_METHOD_REVIEW_20261025.md)已完成：未发现需要立即中止本轮的8步拟合/H32梯度/critic或replay边界错误。固定学习核心来自第三方`CuboiLeo/SuperMimic-Reproduced@d9be385`，不是作者官方训练源码，保持来源表述准确。

确认一项reset采样偏差：500步人工截断的采样回报从零尾值反算，缺少上游的critic bootstrap或临近截断样本排除；1024段中86段超过500帧，最长1044，实际触发次数/效果未量化。它没有直接写错critic GAE或WM物理done标签，本轮保留冻结代码继续，留待实际评测后独立修正。累计采样记忆与上游EMA、初始bin回报差异亦已登记，不新增门槛。

[原样观察快照](../data/manifests/supermimic_q1_BONES_RF02_review_status_20261025.json)实读1625轮/104000转移、WM599组/actor549次、RUNNING；GPU40摄氏度、内存27.105GB/OOM14。唯一只读审阅子agent已结束。本轮完整训练和初始/最终比较仍未完成。

## RF02回放入口准备

### 固定八片的具体名单

17:03:32已在训练仍5600/12000轮时，把原plan的“合库按名称排序前8片”展开成具体名单，保存服务器`BONES_BASELINE_G02_20261025/evaluation_selection.json`，瘦身原样收据为[固定评测名单](../data/manifests/supermimic_q1_BONES_RF02_evaluation_selection_20261025.json)。没有根据策略表现选择或替换动作，没有新几何/效果门；初始和最终评测仍由原控制器执行，选择规则和训练代码不改。

| 固定参考 | 帧数 | 原速秒数 | 整条参考根平面位移m |
| --- | --- | --- | --- |
| `train_0052d4522747ef665036` | 244 | 4.86 | 1.96590 |
| `train_00660a3755173f532abb` | 474 | 9.46 | 1.89859 |
| `train_0081d048da3e3d090c08` | 192 | 3.82 | 1.19839 |
| `train_008ad0df4aced3b85225` | 228 | 4.54 | 2.00085 |
| `train_00aeab45fa5f744d3138` | 566 | 11.30 | 3.31925 |
| `train_00e98fc0d7a57764c762` | 253 | 5.04 | 1.28602 |
| `train_0122791b8661dc86bc7b` | 266 | 5.30 | 1.99687 |
| `train_0124097cf15003106ad9` | 186 | 3.70 | 0.85643 |

合计2409帧、`sum((帧数-1)/50)=48.02秒`，包含前走/侧走/转向等来源，全部train，不证明未见动作泛化。正式评测按每参考帧一次控制，最多2409控制/策略；566帧片不按训练的500步人工截断缩短。两段视频固定用第一片`train_0052d4522747ef665036`，与旧RF01展示的第一片不同，不能混用视频。

8个实际NPZ的SHA/大小均与合库manifest一致；收据绑定合库manifest、IO准入和训练manifest SHA。根轨迹按现有MotionLibrary从`body_pos_w[:, body_names.index("pelvis")]`读取，本次只有数组读取、0新动力积分/0更新。首次辅助提取曾误读独立`root_pos`键而KeyError，未产生输出、未改输入；已按实际布局纠正，不涉及训练故障。

服务器完整收据保留LOCAL_ONLY；瘦身文件`evaluation_selection_compact.json`8057字节/SHA`61585b3e0c2ab784d5c9eacf65509ec6d1e55ed74aaaeaea091c9993869bebed`仅保留必要来源ID/路径/动作名、数组身份和重算统计，不把完整自然语言/演员元数据重复放Git。评测结束后须核对两份实际receipt的clip列表、参考SHA及渲染clip均与此名单一致；目前没有新增策略效果结果。

新增[replay.html](../script/supermimic_q1_reproduction/replay.html)，沿用原Q1回放页的布局与原生视频控件，切换同一首段参考的initial/final实际动力学视频，并从既有`comparison.json`读取该段的完成比例、整片归一化回报、执行段关节RMSE、根位移及提前终止。页面字段已对照真实RF01评测收据和当前评测源码，JavaScript经Node语法检查通过；未使用RF01视频冒充RF02。

当前只完成页面源码准备，尚未部署为RF02可访问回放，也未完成浏览器桌面/移动端画面与控件核验。训练及两份真实RF02视频生成后，将页面复制到`data/supermimic_for_q1/BONES_BASELINE_G02_20261025/index.html`，恢复仅本机可访问的18863入口，再检查视频和指标。现有训练Python源码保持冻结；新增HTML不更改训练输入或控制器。

页面源码SHA`a6d863014adc4526cdae15199939e781ec947cc67d03de106c4d973e82f005b8`已与服务器逐SHA一致，仅复制到`script/supermimic_q1_reproduction/replay.html`，没有提前启动回放服务。15:20:39有界只读观察结束时为2500轮/160000交互、WM949组/actor899次，三进程boot/start_ticks匹配；guard/pipeline RUNNING、GPU39摄氏度、内存26709639168字节/OOM14、盘余2087439777792字节。原控制器继续运行。

15:21取回的[原样状态](../data/manifests/supermimic_q1_BONES_RF02_replay_preparation_status_20261025.json)已继续2525轮/161600交互、WM959组/actor909次，训练计时4649.31秒，内存26702372864字节/OOM14；这些训练指标不是固定参考动力学比较结果，尚无RF02跟随效果结论。

18:36只读确认Mac已有bundled Playwright及其Chromium可执行文件，无需下载浏览器；此项只证明文件存在，尚未以真实RF02页面验证浏览器运行、视频画面或桌面/移动布局。后续实际核验仍待本轮视频生成，不用工具存在代替回放通过。

## 目标交付进度

下表对应既定目标及用户追加的图表交付，状态以各自证据为准。

| 交付 | 当前证据 | 状态 |
| --- | --- | --- |
| 1024个冻结训练来源的原速Q1库，复用128并新增896，仅运行必要IO条件 | 本文PF02合库、[合库收据](../data/manifests/supermimic_q1_BONES_PF02_1024_admission_20261025.json)、[RF02训练输入](../data/manifests/supermimic_q1_BONES_RF02_admission_20261025.json) | 已完成；最终保存核对还要核输入身份 |
| 可微世界模型8步拟合、H32想象BPTT及critic实际更新 | 训练输入配置、[1000保存核对](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_1000_audit_20261025.json)、[方法审阅](README_SUPERMIMIC_Q1_RF02_METHOD_REVIEW_20261025.md) | 中间保存已验证；最终三网络及Adam计数待 |
| RTX3090、64环境、12000采集轮/768000转移及初始/中间/最终保存 | 本文保存表及10000观测，实际初始与1000至10000保存已存在 | 训练进行中；final.pt尚不存在 |
| 同一固定8条参考的初始/最终MuJoCo动力学比较，含误差/回报/提前终止/移动 | [提前保存的具体名单](../data/manifests/supermimic_q1_BONES_RF02_evaluation_selection_20261025.json)、既有pipeline评测与独立算术核对阶段 | 待训练结束；还要核两份实际名单/参考SHA及566帧整片覆盖 |
| 首片初始/最终真实物理轨迹视频与可访问回放 | 本文回放准备、[页面源码](../script/supermimic_q1_reproduction/replay.html)、pipeline两项render阶段 | 待实际视频、18863服务及桌面/移动播放核验 |
| 每轮六宫格PNG/PDF/CSV及来源收据 | [图表阶段](README_SUPERMIMIC_Q1_TRAINING_DASHBOARD_20261025.md)、已目视预览、唯一终后绘图回调 | 待最终图检查；本轮未记录三项回合统计须留缺 |
| 统一ZHY四目录、控制磁盘和内存 | [存储归拢](README_Q1_STORAGE_CONSOLIDATION_20261025.md)、既有guard、各次资源观测 | 当前满足；终后复核，权重LOCAL_ONLY |
| 阶段README/索引/当前状态及Q1分支记录 | 本文、报告索引、当前状态、各阶段Git提交与独立远端核验 | 阶段持续保存；整轮最终报告待 |

## 计划产物与恢复

- 原参考：`data/supermimic_for_q1/BONES_PF01_128_20261025/library`。
- 新7批：`data/supermimic_for_q1/BONES_PF02_{offset:04d}_20261025`，offset为128至896。
- 合库：`data/supermimic_for_q1/BONES_PF02_1024_20261025/library`。
- 训练：`data/supermimic_for_q1/BONES_RF02_12000_20261025/run`。
- 流程状态：`data/supermimic_for_q1/BONES_BASELINE_G02_20261025/status.json`，计划与进程身份同目录保存。
- 日志：统一`logs/BONES_*_20261025*.log`，模型/参考/视频只写data。

后续继续观测既有RF02至原定12000轮，再完成流程已有的最终保存核对、固定8片动力学比较和回放；actor/critic真实更新及1000保存核对均已有证据。每完成重要阶段更新本README/当前状态/索引并同步GitHub；训练/评测/完整目标未完成。代码、README及小收据按阶段同步用户镜像与服务器，服务器AGENTS已增补用户确认的最多一个子agent及每轮六宫格图要求，保留其余历史差异。旧可视化18863入口本次curl连接失败；待本轮真实视频生成后恢复可访问入口，不用旧视频冒充RF02。
