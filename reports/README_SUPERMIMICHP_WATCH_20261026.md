# SuperMimicHP X2仓库跟踪与Q1初步对照

记录日期：2026-10-26，北京时间。阶段HPW01：首次只读核验完成，每小时持续跟踪已启用；没有执行仓库代码或新增训练。

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

状态文件`data/manifests/supermimichp_watch_state.json`保留当前基线与来源对象；每次有实质变化更新同日README、索引与当前摘要，显式暂存并发布Q1。当前RF06继续由原监督检查boot/PID/start_ticks、实际训练时长及终后物理评测，仓库跟踪不触发热改或额外训练。

## 保存与下一步

阶段README、AGENTS偏好、索引/当前摘要与状态JSON先保存到权威Q1工作区；Git提交号及发布状态以实际提交/独立远端核验为准。小记录同步Mac和自己的25252统一根，第三方原副本留忽略目录；无模型/数据入Git，无新大型工件或网盘备份声明。

下一步等待后续提交并对照；当前候选最高优先级为actor是否得到足够的根位置/朝向反馈。RF06仍是已授权的腿部限矩单变量试验，先完成其真实结果和本地MP4交付，再依据证据安排后续工作。
