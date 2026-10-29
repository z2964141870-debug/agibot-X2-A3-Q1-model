# MorphMimic源码与最小复现准备

记录日期：2026-10-29，北京时间。当前阶段：MM05。原版单G1训练/ONNX/同片回放工程闭环已完成，稳定步态尚未复现。完整数据、本地六MP4及范围见[最终结果](README_MORPHMIMIC_G1_RESULTS_20261029.md)与[逐工件收据](../data/manifests/morphmimic_g1_results_20261029.json)。下方MM04/MM03/MM02是对应时点历史。

600轮实际WM/policy/critic6000/100/2000，同步iteration1249.131082秒；三片初末完成率均0/3。最终失败65/57/116帧，不把失败稍晚称为自然步态。六原速视频、六联系图/六格曲线和43小工件5600206字节Mac/25386大小SHA已验；最后停走渲染超时及单片补交保留。原训练14:08:55/补交14:30:37正常回收，进程退出/GPU15MiB无计算/OOM0，权重/ONNX留服务器。训练冻结仍c7f0499、原183源码不变，没有追加训练或改自动任务。

## MM04唯一工程训练与固定回放

2026-10-29 13:47:33北京时间实际启动原版G1训练。运行冻结`c7f049965d1841e5ee556637695ea2494fda5bb1`已push并独立核40位远端；10文件319374字节与Git blob、Mac主根、25386逐大小及SHA一致，启动前183原版blob未变。后续文档提交不改变这次运行来源。[发布收据](../data/manifests/morphmimic_deployment_publication_20261029.json)。

guard185658/start16649863、timeout185659和trainer185660/start16649878，原boot/GPU。64环境、WM/imag128、H32、原500轮WM-only及总600轮，原网络宽度/奖励/控制不变，compile none。外层3500秒全进程硬限和20秒清理，包含初始化，不扩时/重试。359轮现场实测677.895477秒原始iteration时间，WM3590/policy0/critic0；仍是预热，不是策略学习或自然步态证据。[启动与预算](../data/manifests/morphmimic_g1_launch_20261029.json)。

在观察回放结果前固定三片：`walk_forward_loop_001__A022`(1032帧)、`walk_forward_start_002__A023`(152帧)、`walk_forward_stop_005__A026`(253帧)，均来自原seed0 tiny。初始checkpoint0使用本次64环境训练保存，不混用16环境probe；最终600轮。用原版exporter做PyTorch/ONNX parity，再用同一资产/初态和`--interface motion`运行原版sim2sim。保留提前失败及短视频，不补hold帧、不换片；权重/ONNX留服务器。[固定计划](../script/morphmimic_repro/review_plan_20261029.json)。

当前工程小闭环不是作者4096环境/40000轮的性能复现，不是Q1模型或候选采纳。原监督及其他任务不改，无agent或旧goal。待原训练退出后才导出资产及回放。

## MM03原版部署执行与转发恢复

### 当前已通过的部署与预检

用户补齐容器认证后，原Windows及25386双master均通过检查，服务器于12:59北京时间核原boot `4d859008-4c83-4a6d-814d-cbb9dfe885c9`、GPU `GPU-26028785-9fb9-45af-e614-4cc53af15d08`，3090/24576MiB、驱动595.112.07，容器内存上限34292629504字节/OOM0。初始GPU15MiB无计算进程；磁盘可用约1.5TiB。没有重启桥接机、容器或旧训练。

普通Git clone无进展约3分钟后终止并保留不完整目录，改为官方GitHub API固定提交归档。主仓183个blob/140372650字节、GMR固定gitlink的1938个blob/1249216405字节逐Git blob SHA-1全部匹配。主仓归档SHA256 `fe877e4cc1600c6d7cb234e457784bc66af4421d4ced8da919f3cacfc9ea67a6`；GMR归档SHA256 `efefae3f719be5f950f274d0c6920f1b309b4e3ee6144c8bb54e1847144477c0`。这是可运行源码归档，不是完整递归Git历史；真机SDK子模块未安装，本阶段只做仿真。

独立环境`data/environments/morphmimic_original_20261029`按原`uv.lock`使用`uv sync --frozen --group export --no-dev`完成149包安装：Python3.11.18、Torch2.11.0+cu128、mjlab1.6.0、MuJoCo/MuJoCo-Warp3.11.0、Warp1.17.0、ONNX1.23.2、ORT1.31.0。CUDA实际张量计算通过。初次训练help/转换在import时因镜像缺Git报错，随后仅将Ubuntu Git包解压到`data/environments/morphmimic_git_tools`，显式设`GIT_PYTHON_GIT_EXECUTABLE`后help与转换通过；没有改系统包数据库、HP环境或上游算法文件。

直接复用服务器旧intake保留的原始G1 CSV，不重新传23.5GB归档，不使用Q1重定向结果。原版converter取按路径排序的前128条、workers4、metadata v004，得到860.38秒，原版name/feasibility过滤128/128通过；原版seed0 tiny为50条/363.72秒，29维关节/50Hz/数值有限逐文件通过。上游生成的`1h/7h/40h`文件此时也只有0.239小时，文件名不代表实际数据量。[逐源与NPZ SHA清单](../data/manifests/morphmimic_g1_data_20261029.json)。

13:24:15至13:25:44，16环境原版Trainer创建、40步实际采集640转移、checkpoint0保存通过，实际optimizer步数WM/policy/critic为0/0/0，优化器状态为空；worker正常退出0、guard回收，GPU回到15MiB/OOM0。receipt在实验根`probe_observation/progress.json`及`probe_guard.json`，原日志`logs/morphmimic_g1_probe_20261029.log`。这不证明学习或步态效果。

本项目仅新增`script/morphmimic_repro/`薄封装，调用原版Trainer、计优化器实际step和同步iteration时间，初始checkpoint0单独保存，不修改原版iteration/奖励/模型计算。唯一工程训练拟64环境、WM/imag128、H32、原500轮WM-only和总600轮；外层复用既有runtime_guard及`timeout3500秒/TERM/20秒清理`，全进程时限包含初始化，保守保证实训不超过3600秒。原始训练器其余默认保持；这不是论文4096环境/40000轮或三机器人复现。正式训练须在本阶段冻结发布并核三端文件后启动一次，不故障重试或自动扩时。

### 连接恢复过程与执行顺序

用户明确“先原版部署复现，服务器重新开个转发”，随后补齐Windows登录信息。本次授权覆盖此前不重建桥的限制，优先执行原版单G1复现，保留SuperMimic底座/PPO两线方向。独立训练遵守单轮最多3600实训秒，不沿用已过期夜间预算。

截至2026-10-29 12:57北京时间，原桥`yu@100.65.183.91`已建立`127.0.0.1:18873 -> 59.77.15.15:25386`，Windows控制master检查通过。第一次交互在认证期间Broken pipe；第二次显式password认证正常建立连接。随后25386 root的BatchMode认证返回Permission denied，说明转发已到达容器SSH，但尚未登录。已请求用户完成容器认证，密码不写命令参数、文件或Git。没有因断线操作容器或旧训练。[连接收据](../data/manifests/morphmimic_connection_20261029.json)。

后续顺序：

1. 完成25386登录，核实时boot/GPU/驱动、可用内存、磁盘和当前计算任务；旧boot/GPU仅用于比对。
2. 在`/opt/data/private/ZHY/script/third_party/morphmimic_aa40fe46`取得完整固定源码及所需子模块，环境单独放`data/environments/`，不更新既有HP环境。上游有`uv.lock`，优先按锁文件安装训练及export依赖；本次已核`.gitmodules`均为公开HTTPS，GMR固定gitlink为`25f56db79ac411258b377b6d4ccba0c8ff3372ac`。这些安装尚未执行。
3. 优先查看本项目已有G1原包/CSV/metadata并选择原版G1小子集，使用上游converter；不重新传输整包，不用Q1 NPZ替代。数据读取、50Hz时轴和G1维度须实际检查。
4. 原生零optimizer检查通过后，独立限时运行包含WM和actor/critic更新的小闭环，保留上游算法；64环境/600轮仅为工程配置候选，与论文4096环境/40000轮区分。
5. 用上游exporter核PyTorch/ONNX动作一致性，导出G1仿真资产并用上游`deploy.run`生成原速MP4，取回Mac核大小/SHA；另外交付训练曲线。工程闭环和论文性能分别报告。

上面的执行顺序中，连接、运行依赖、G1数据和零optimizer原生仿真现已完成；独立训练、ONNX和MP4仍待。下文MM02快照范围与连接阻塞仅为历史，不覆盖本段新现场。

## 用户要求与判断

用户要求继续SuperMimic底座与PPO适配两线工作，随后提供MorphMimic项目页，询问是否有代码、能否复现。结论是有公开实现，适合作为同一方法路线的独立对照。它保留可微世界模型、想象rollout和actor/critic，仓库明确说明是依据SuperMimic论文自行实现，不是SuperMimic原作者代码。

本次先查原版可复现性，不把既有Q1项目整体迁移，也不将其G1结果当作Q1已获改善。原两线脚底诊断及候选保留，见[本地时序分析](README_SUPERMIMIC_Q1_DUALTRACK_DIAGNOSIS_20261029.md)。没有新增Q1物理、策略推理、optimizer或训练，没有恢复旧作业、agent或goal，没有修改自动任务。

## 实际完成与未完成

| 层次 | 本次结果 |
| --- | --- |
| 代码是否公开 | GitHub API完整树可读，固定main提交aa40fe46a7fa995ba7a7354ad065a2c8ad64a302，与10月27日初评相同 |
| 源码快照 | 8个关键文件160435字节，逐一核字节数和Git blob SHA-1，另存SHA-256；不含完整仓库、机器人mesh和子模块 |
| 静态检查 | 5个Python文件AST解析通过；实际模块导入、依赖安装及训练CLI尚未运行 |
| 官方在线推理 | 已在应用浏览器加载官方demo，G1/X2 Ultra/T2三机器人Walk画面和运行状态可见，显示约0.7-0.8倍实时，页面已保留 |
| 训练checkpoint | 本次GitHub releases返回空列表，所查README没有成套PyTorch训练checkpoint下载入口；在线demo实际下载了显示49.1MB的推理策略，不能因此说没有任何公开模型资产 |
| 独立离线复现 | 尚未完成；官方网页运行不等同本机独立训练、部署回放或论文性能复现 |
| Q1接入 | 尚未实现；当前registry只有g1/x2/t2，不能把已有共享策略直接装到Q1 |

官方demo是第三方已有策略的在线推理/仿真，不是本项目训练结果；未将其权重另行保存为Q1本地checkpoint。浏览器低于实时不等于服务器训练速度或策略质量结论。

普通Git HTTPS查询约75秒连接失败，原始README和pyproject读取各在30秒超时；改用公开GitHub API读取固定树与blob成功，不需要认证。没有完整clone、安装大型环境或下载SEED全库。

## 依赖、规模与Q1差异

固定pyproject要求Python 3.11、mjlab 1.6.0、Torch>=2.8.0、Warp>=1.14且<1.18；Linux Torch使用cu128索引。当前Mac系统Python为3.9.6，本次仅做不导入依赖的源码检查。后续使用独立服务器环境，不能在现有HP环境里直接uv sync覆盖依赖。

上游README报告4096环境、1小时动作子集约28GB显存；全库约51-58GB/GPU，多机器人配置每机器人一张GPU。这是作者报告，不是本项目测量，也不能保证缩到64环境就必然适配现卡。先测单G1小数据/小批量，完整论文多形态规模另行评估。

`robots/spec.py`要求G1规范29槽位各自唯一匹配本体关节，缺失或重复均拒绝；Q1现有22控制关节不能满足该直接映射。需要独立设计缺失自由度表达或Q1专用维度，并核IMU、关节符号/零位、PD/effort、碰撞/足底和Q1参考。不能仅补零就宣称共享策略支持新机器人，也不能把Q1 NPZ作为G1动作库直接输入。

源代码Apache-2.0，机器人资产和子模块各保留自身许可。本次第三方原始文件仅存被忽略的data，不作为本项目源码重新发布。

## 最小复现方案与一个易误判处

首先完成原版单G1的环境创建、参考/动作契约及零optimizer检查，再执行包含真实actor和critic更新的小闭环，导出后用相同动作回放。工程连通后再比较训练稳定性与普通走效果，最后做Q1独立适配和同底座PPO对照。多形态共享与五点接口不放入首个小闭环，避免同时扩大变量。

已读`TrainCfg.wm_only_iters=500`，`Trainer`只在越过这一门后进入策略更新；入口注释的60轮短示例维持此默认时，只能证明采集/世界模型阶段运行，不能证明策略学习正常。准备的小闭环为64环境、WM/想象batch各128、H32、500轮WM预热、总600轮、compile none；这是待验证工程配置，非论文等规模复现。后续必须核实际policy/critic更新计数，而非只看进程正常结束。

以下命令仅为已检查字段的准备记录，**未执行且尚不具备直接启动条件**。先取得完整固定仓库/必要子模块、真实G1 tiny动作库、通过原生检查，另接独立实训时钟/退出保护。长期单轮最多3600实训秒作为上限，故障不自动重试/扩时；600轮不代表承诺必须耗尽一小时，也不能将初始化/编译凑作训练时间。

```sh
# 在独立服务器checkout及环境中；G1_TINY必须指向已核的G1动作子集。
uv run python -m morphmimic.scripts.train \
  --robot g1 --subset-yaml "$G1_TINY" --gpu 0 \
  --num-envs 64 --wm-batch 128 --imag-batch 128 \
  --horizon 32 --wm-only-iters 500 --iterations 600 --compile none \
  --log-every 10 --ckpt-every 100 \
  --log-dir /opt/data/private/ZHY/data/supermimic_for_q1/MORPHMIMIC_G1_SMOKE_20261029 \
  --run-name source_smoke
```

该命令尚无本项目独立时钟guard，不能从本记录自动开训。正常退出、有限loss和小样本回放均不能代替稳健步态验收。

底座稳定性可比较的具体源码项包括WM/critic LayerNorm、critic目标上界、WM状态坐标及归一化、合法想象起点；这些与本项目HP实现的准确差异及效果仍待独立对照。它们不是已经定位的Q1退化原因。PPO/残差适配继续保留，后续分别核底座变化与适配净增益，不能相加两份独立结果替代组合实测。

## 版本、产物与来源

Mac快照根：`/Users/yu/Documents/ChatGPT/元启/data/references/morphmimic_review_20261029`。

- `tree.json`：完整文件树，183个blob约140.37MB的仓库规模仅统计，未全部下载。
- `source/`：README、LICENSE、pyproject、constants、registry、Translator、训练入口和trainer共8文件。
- `source_receipt.json`：固定提交、8文件大小/Git blob/SHA-256、5个静态解析结果。
- `repro_plan.json`：实际默认参数、工程小闭环候选和执行前缺项。
- [小型核验清单](../data/manifests/morphmimic_repro_readiness_20261029.json)：版本、来源、官方在线演示、网络失败及范围边界。

一手入口及固定源码：

```text
https://lzyang2000.github.io/morphmimic/
https://github.com/lzyang2000/morphmimic
https://lzyang2000.github.io/morphmimic/demo/
https://github.com/lzyang2000/morphmimic/blob/aa40fe46a7fa995ba7a7354ad065a2c8ad64a302/README.md
https://github.com/lzyang2000/morphmimic/blob/aa40fe46a7fa995ba7a7354ad065a2c8ad64a302/pyproject.toml
https://github.com/lzyang2000/morphmimic/blob/aa40fe46a7fa995ba7a7354ad065a2c8ad64a302/src/morphmimic/robots/spec.py
https://github.com/lzyang2000/morphmimic/blob/aa40fe46a7fa995ba7a7354ad065a2c8ad64a302/src/morphmimic/train/trainer.py
```

## MM02历史连接阻塞

本轮此前唯一25386入口返回Connection refused，一次有界本地核查无18873监听且控制socket缺失，远端UNKNOWN；没有因为本次问题重复连接、改旧入口、读取认证或重启桥。当前无法在服务器取得完整环境、G1数据和实际GPU验证，离线训练复现仍待。

记录在权威Git Q1分支保存，Git推送/独立40位SHA及Mac镜像以实际发布收据为准；25386镜像明确待连接恢复，不能称三端一致。原监督ACTIVE每10分钟与其他任务保持，候选采纳null，未结算整体目标。

恢复先读本记录、[当前状态](README_CURRENT_STATE.md)及[双线诊断](README_SUPERMIMIC_Q1_DUALTRACK_DIAGNOSIS_20261029.md)。下一步是恢复授权入口后的独立原版G1环境和零更新检查，不直接改旧Q1冻结runtime或恢复RF15/RF16。
