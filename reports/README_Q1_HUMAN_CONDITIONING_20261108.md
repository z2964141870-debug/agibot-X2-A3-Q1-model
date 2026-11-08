# Q1因果人体条件纠偏对照

日期：2026-11-08，北京时间。实验：F0_HUMAN_CONDITIONING_PAIR01。当前仅独立实现及六项本地CPU合同通过，真实新GPU训练未派发。原共享训练继续暂停，低延迟自然遥操目标尚未达到。

## 问题与可检验假设

[训练目标诊断](README_Q1_TRAIN_OBJECTIVE_RESULT_20261108.md)确认旧训练能力也退约5%，纯学生训练收益约72%但对应验证只约12.5%；四批局部梯度并不能证明失败原因。当前可训练纠偏分支只看930维机器人历史，人体当前与历史经冻结编码器形成基础latent。本轮检验让纠偏分支直接使用已因果可见的人体latent是否有帮助，而不是延长同一种逐状态拟合。

这项是本项目独立条件输入消融，不是GigaBrain复现，也没有证明缺少人体条件是原失败根因。仍以全身同目标跟随为任务；上肢当前目标加自主脚步不能替代它。

## 唯一实验变量

保留实际已经训练3000更新的机器人历史四张量分支。增加`Linear(latent,128)→ELU→Linear(128,latent)`人体分支，末层权重和偏置全零；两组同种子、同参数、同初始动作。新latent为`z+(1-abs(z))*tanh(robot_adapter(history)+human_adapter(real_z_or_zero))`。

- `zero_human`：新增分支输入与人体latent同形状的全零。
- `real_human`：新增分支输入冻结f0＋历史人体编码器当下已可用的latent。

两组名义参数容量一致，控制组零输入使部分权重梯度恒零，这是该对照的一部分，不能称有效信息容量一致。新增零输入分支仍可通过偏置学习常量，因此历史retained_point不能直接替代本轮控制组。

teacher、世界模型、人体编码器、共享decoder、所有归一化、动力学及时间锚点全部冻结；仅两个残差分支各四张量可训练。前向只接收原`f0[10,226]`和机器人历史930维，不接收教师未来目标、next_state或terminal。原输入构建与保存序列边界已审计，本轮复用；原始人体前处理仍为offline_unverified。

## 数据、曝光与选点

严格复用旧6725train/2358validation和纯学生406train/219validation，来源组split保持分离。旧6581、纯387训练四标签窗，40ms标签间隔、120ms跨度仍是保存状态上的open-loop拟合，没有新物理递推。

相同原seed2026110806、每批16旧窗＋16纯窗，128动作标签；每组最多新增3000Adam，学习率0.0001、梯度裁剪1、每500更新评估。两组相同point动作MSE＋20倍旧train初始输出保持，不同时增加increment项或改标签、损失权重、采样。训练摘要和来源曝光必须逐组精确一致，不能截断一臂凑预算。

保留原matched/pure各半验证风险和三个旧能力不退门：全旧、固定R1、固定mixed；包括update0。test与新留出事件未打开，不用验证梯度更新。选回初始须CPU核验实际base、机器人分支及人体零输出结构，再直接复用旧0/7闭环；只有真实改变且选中的策略才进入独立物理与原速视频验收。

## 初始化与可见性门

在任何真实更新之前，从旧train与纯学生train各取最多16行，检查两组CPU动作与各自原策略逐位同、GPU动作与原GPU策略逐位同、CPU/GPU最大差不超过2e-5。八残差张量做两次train-only autograd，梯度有限，冻结模块及模型SHA不变。该门在本地合成CPU已过，真实原运行时和真实CUDA门仍待。

六项本地检查覆盖真实分支输入、初始动作和名义容量、冻结范围、未来对象和source/split隔离、真实小型Adam与八计数、原保持目标、同批次曝光、选0及拒绝静默截断；1.044秒全过，CLI也过。不是新增真实Q1训练效果。

## 有界执行与成本

当前原台账4820/14400秒，全SETTLED无预留；r1/r2/r3为1614/1835/1371，总余9580、r2余2665秒。新对照最多600秒/128MiB，原reserve→worker回收→settle保持，不重置阶段或总预算。22GiB准入/24GiB退出、GPU18GiB80度、100GiB余盘/private500GB、原非阻塞GPU锁与自有子树清理不改，不清缓存。

10:46:17现场通过原Linux45014核验：原训练暂停/五PID退出，原锁和计算空、OOM0，总8267460608B/限34292629504B，GPU1MiB25度、余盘183396945920B。这是历史快照；实际部署及派发前重新核源、预算和资源。

新根：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_F0_HUMAN_CONDITIONING_PAIR01_20261108`。新六文件之外的旧执行来源不改，原输出/日志/派发收据/精确进程均无才唯一派发。必要连接由Agent自行按固定source/ssh入口和逐跳原生密码完成，不让用户代测；认证不入记录。

## 代码与复现

六新增文件在`script/supermimic_teleop`：`student_conditioning_controller.py`、`student_conditioning_worker.py`、`student_conditioning_models.py`、`student_conditioning_fit.py`、`student_conditioning_plan_20261108.json`和`test_student_conditioning.py`。先scoped发布源码和来源清单、独立40位远端及Mac核同，再原运行时合同和资源准入；实际执行版本随后登记。

本地检查命令，在该脚本目录运行：

```sh
/Users/yu/Documents/ChatGPT/元启/data/environments/q1-teleop-mps-20261106/bin/python -m unittest -v test_student_conditioning
```

原运行时controller先`--admit-only`，正式单派同命令移除此参数，外层`timeout --signal=TERM --kill-after=5s 610s`，内部最大600秒。manifest、SHA与evaluation-commit必须使用实际已发布版本，不用占位提交启动。

## done与todo

1. 独立实现、六本地CPU合同和CLI已过，原32k/旧六训练文件和实际数据未改。
2. 待源码/manifest发布、服务器六CPU合同、来源/资源准入以及真实初始化门后单派。
3. 训练结束交付两组六宫PNG/PDF/CSV、原始数值与成本，按真实选点决定闭环，不能据loss改善判自然遥操合格。完整软件与实测人体/硬件端到端目标继续保留。

[准备原记录](../logs/session_records/q1_human_conditioning_preparation_20261108.json)。服务器小记录和大索引镜像状态以publication为准；当前未同步处PENDING，采纳null。
