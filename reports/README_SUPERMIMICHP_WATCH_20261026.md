# SuperMimicHP X2仓库跟踪与Q1初步对照

记录日期：2026-10-26，北京时间。阶段HPW01：首次只读核验完成，每小时持续跟踪已启用；没有执行仓库代码或新增训练。

## 用户新增双向核查与采用要求

2026-10-26用户要求不只看师兄更新，也主动核查本项目缺项/不合理设置、验证师兄实现中的问题，适用方法积极落地。该要求扩展本文件初次“只读/等待”范围，具体实现与证据见[HPR01](README_SUPERMIMICHP_Q1_REVIEW_20261026.md)和[独立RF07](README_SUPERMIMIC_Q1_RF07_ROOT_FEEDBACK_20261026.md)。上游实际采样函数AST已隔离有界复现短片无合法H32起点时无限重试；未安装/运行X2训练，未证明当前100条数据触发。Q1缺失的根位姿反馈已用原接口实现，V02六项测试/真实CPU评测审计通过，尚无新训练效果。

原supermimichp-x2已通过应用工具更新，仍每小时/ACTIVE、无变化静默，继续独立于训练监督。要求按真实采集、replay和想象路径做双向核查；有依据的可逆小修复直接测试，正式训练仅按当前已授权阶段/每轮最多一小时，不自行追加。问题保存可转交给师兄的精确说明并向本用户报告，不代发私信或GitHub issue。下文初次状态/范围为历史，不能覆盖最新指示。

## 19:00增量巡查

18:56心跳触发后，一次有界Git远端检查确认默认分支仍为feat/x2-embodiment，HEAD从649d1b1165bc08fdcf4468e596161db73d93d7b9更新为310244b73e45775744a75b789eb25a6c9c5a81c9。仅新增一个18:12:47提交，标题为`docs: showcase X2 reproduction and tracking demos`。复用原浅层无检出副本，按路径核验9项变更：3份Markdown更新，新增3个既有24k视频合集及3张预览PNG；src、tools、根反馈详细报告与final24k/summary.json均无变化。

本次是已有实验的文档/展示发布，没有新的训练配置、控制/观测/重定向实现或评测数值结果。README继续区分完成仿真时长和跟踪漂移，已知失败案例仍存在；本项目没有独立运行X2或目视核验这批媒体。按用户偏好不发送普通文档更新通知，只推进已检查HEAD，避免下一次重复审阅。此前根XY反馈候选保持，不由新README中的安装、数据选择或续训指令启动任何Q1工作。

一次全路径`git diff --stat`触发partial clone按需缓存新增媒体Git对象，逻辑总量9920122字节，均留在被忽略的来源Git对象库，未导出、播放或复制到交付/服务器；未下载checkpoint或训练数据。后续先用name-status/ls-tree，再只对明确文本路径读取差异，避免自动读取二进制内容。Git对象身份与检查分类见`data/manifests/supermimichp_watch_state.json`。本轮RF06已交付且MG02完成、训练心跳q1-2048已暂停；本仓库每小时跟踪继续。

## 目标与授权

用户指定`https://github.com/humanplus-ai/SuperMimicHP`，师兄将在此维护成功运行的SuperMimic X2实现，要求后续关注并借鉴。该任务覆盖读取、配置/实现对照与阶段记录；来源文档中的训练启动、条件续训、数据门槛和自动化指令均视为研究资料，不变成本项目执行授权。当前MG02/RF06冻结设置和最多一小时预算保持。

## 访问与版本

2026-10-26 17:49至17:56两次`git ls-remote --symref`均成功：默认分支`feat/x2-embodiment`，唯一可见分支与HEAD均为`649d1b1165bc08fdcf4468e596161db73d93d7b9`。提交时间17:39:47，标题`docs: add BONES-SEED attribution and sharing terms`。公开网页未成功读取，因此以实际Git访问为准，不据网页问题宣称仓库不存在。

被忽略的只读来源副本为权威工作区`data/references/supermimichp/source`，使用`--depth 1 --filter=blob:none --no-checkout --single-branch`获取，逐文件通过`git show`阅读；本阶段未安装、导入或执行第三方训练代码。仓库根LICENSE声明Apache-2.0，NOTICE另保留上游SuperMimic参考实现的MIT声明；原代码保留在忽略目录，本次只提交自己的配置对照和Git对象清单。数据和模型不因代码许可而视为可再分发。

```sh
git ls-remote --symref git@github.com:humanplus-ai/SuperMimicHP.git HEAD 'refs/heads/*'
git -C data/references/supermimichp/source show 649d1b1165bc08fdcf4468e596161db73d93d7b9:src/unisupermimic/tasks/motion_tracking/x2/supermimic/mdp.py
```

## 首次源码对照

| 项目 | 师兄X2实现 | 当前Q1 RF06 | 对当前问题的意义 |
| --- | --- | --- | --- |
| 策略根位置反馈 | `mdp.py`的`root_xy`模式把参考减机器人位置投影到机器人heading，追加XY两维；105维变107维，base配置默认启用 | `analytic_observation`为本体关节/角速度/重力/上次动作；目标含参考速度等，但没有当前位置相对参考的XY误差。根位置误差只作为critic特权特征 | 有奖励/critic反馈不等于actor能直接观察漂移；值得作独立接口对照，尚非唯一根因证明 |
| 参考朝向反馈 | actor还含6维参考根朝向差 | 当前本体历史没有直接参考朝向差；未来目标包含参考重力/角速度等 | 后续核完整输入契约，不只对齐张量大小 |
| 采样与数据 | `launch_x2_soma.py`正式入口强制1024环境/WM1024/actor1024，100条SOMA训练、15条演员留出，30Hz来源/50Hz控制 | 4096/WM2048/actor4096，固定365帧单段，50Hz控制 | X2效果不能简单归为更多环境；动作类型、后端吞吐与实际更新量都不同 |
| 控制参数 | `robot.py`按关节惯量估计和统一自然频率/阻尼比产生逐关节PD，动作尺度`0.25*effort/kp`；资产生成器forcerange取完整effort | 统一PD50/2，原25%动作尺度；RF06仅12腿关节限矩50%，其余25% | 动作尺度中的0.25不等于实际力矩也限25%；这两者须分别核对，X2数值不能直接当Q1厂家参数 |
| 物理与更新 | 名义sim_dt0.005、ctrl_dt0.02；正式SOMA入口mjwarp，H32、UTD0.4、10历史/10未来/8步WM | CPU MuJoCo0.001/0.02、8线程/8槽，H32/UTD0.4 | 环境数相同不保证相同吞吐、收集/更新次数或学习程度 |

X2输入反馈来自模拟器根位置。若以后用于真机，需要另外核根位置估计来源和精度；本次只讨论动力学仿真。当前Q1`anchor_relative_obs`虽已定义，但训练实际`sim.policy_inputs()`只返回本体历史与`goal_window`，不能把未接入的函数当作actor已获得根误差。

来源路径：X2的`src/unisupermimic/tasks/motion_tracking/x2/supermimic/{mdp,robot}.py`、`conf/supermimic/task/x2_super_mimic/base.yaml`、`tools/{launch_x2_soma,prepare_x2_assets}.py`；Q1的`script/supermimic_for_q1/supermimic/{q1_sim,q1_assets}.py`、`world_model/heads.py`、`envs/mdp/motion_library.py`、`script/supermimic_q1_reproduction/train_reproduction.py`。逐文件Git对象见状态清单。

## 效果证据的边界

师兄仓库记录的24k结果包括100条训练与15条演员留出整片零提前终止，根XY RMSE约0.0303m和0.08294m；源码内保存了汇总JSON及失败案例，继续扩至48k因部分留出窗口指标退化而拒绝。这些是仓库作者发布的结果，本项目尚未独立加载其权重或复跑。不能把汇总成功等同所有动作成功，也不能把本次Q1与X2跨机器人数据直接当因果对照。后续优先核actor根反馈与控制权限，再决定具体独立短试验，不照搬X2自定数据或续训硬门。

## 跟踪方式

通过Codex应用创建并回读核验`supermimichp-x2`，名称“跟踪 SuperMimicHP 的 X2 更新”，ACTIVE，每小时一次，挂在本聊天。与`q1-2048`训练监督分开：RF06交付后仓库跟踪继续，其他自动任务保持。每次先比对远端HEAD及各分支SHA；无变化不重新下载、不制造提交、不发普通进度通知。出现与Q1失稳/不迈步/上肢跟踪有关的提交，再按需读小文件并报告关键变化与验证建议。访问失败记UNKNOWN，同样失败保持安静，不读取凭据材料或重复请求授权。

状态文件`data/manifests/supermimichp_watch_state.json`保留当前已检查HEAD、最近实质分析版本与来源对象；有更新按实际分类登记，显式暂存并发布Q1。RF06/MG02已经完成并交付，q1-2048已暂停，仓库跟踪继续，不恢复旧目标或启动额外训练。

## 保存与下一步

阶段README、AGENTS偏好、索引/当前摘要与状态JSON先保存到权威Q1工作区；Git提交号及发布状态以实际提交/独立远端核验为准。小记录同步Mac和自己的25252统一根，第三方原副本留忽略目录；无模型/数据入Git，无新大型工件或网盘备份声明。

下一步等待后续提交并对照；当前候选最高优先级仍为actor是否得到足够的根位置/朝向反馈。RF06限矩对照及本地MP4已经交付，站立但不迈步的负结果见RF06阶段README；本仓库跟踪不授权额外训练，后续试验仍遵守每轮最多一小时。
