# Q1因果人体条件纠偏对照

日期：2026-11-08，北京时间。实验：F0_HUMAN_CONDITIONING_PAIR01。两组真实GPU训练已完成，各3000更新，最终均选新增update0；训练终态和清洁回收已收到，独立CPU权重审计提交后未收到回执，原曲线与六宫图交付待补。原共享训练继续暂停，低延迟自然遥操目标尚未达到。

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

在任何真实更新之前，从旧train与纯学生train各取最多16行，检查两组CPU动作与各自原策略逐位同、GPU动作与原GPU策略逐位同、CPU/GPU最大差不超过2e-5。八残差张量做两次train-only autograd，梯度有限，冻结模块及模型SHA不变。该门在本地及原Morph运行时合成CPU已过，真实CUDA门仍待。

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
2. 源码/manifest发布、服务器六CPU合同和来源/资源准入已过；先发布准入，再唯一单派，并在任何真实更新前过CUDA初始化门。
3. 训练结束交付两组六宫PNG/PDF/CSV、原始数值与成本，按真实选点决定闭环，不能据loss改善判自然遥操合格。完整软件与实测人体/硬件端到端目标继续保留。

[准备原记录](../logs/session_records/q1_human_conditioning_preparation_20261108.json)。服务器小记录和大索引镜像状态以publication为准；当前未同步处PENDING，采纳null。

## 服务器原运行时准入

源码`8a742af22085089b0b599dace72dc95e7f7e9f69`、六文件清单发布`dc55cd0f77021d4f71a331c100fd8bc83841d5e6`均已显式push origin Q1并独立40位远端核同，Mac源码十一文件917059B及清单1676B大小SHA同。18486B九成员纠正包SHA `e4fde03c6aad65f39f7c7df38191819dfa6902e19734faccafc4a58128b54b2b`已十五精确ACK部署，1229原来源前后同。首包因Mac附加的九个AppleDouble成员被原白名单拒绝，所有目标/receipt/archive确无写入后才纠正重传；没有放宽成员门，失败包本地保留。

原Morph运行时六CPU检查通过，外层进程5.803316秒，GPU模型调用和真实优化均0。10:54:42来源/原暂停/资源/成本与锁准入11.758276秒通过，1240保护文件、总8263753728B/限34292629504B、GPU1MiB25度、OOM0、余盘183396569088B；原4820秒全部SETTLED无预留。三原回执4823B、2093B包SHA `d3805cd6bda7cce9812a5ad2484660312fcd67f957a41fb186661b541346df7a`已Mac二进制和逐件核同，原文未改。先发布本准入再唯一600秒派发，不重部署或重复合同，原共享不恢复。

- [新增部署原件](../logs/session_records/q1_human_conditioning_deployed_20261108.json)
- [原运行时合同原件](../logs/session_records/q1_human_conditioning_contracts_20261108.json)
- [真实准入原件](../logs/session_records/q1_human_conditioning_admitted_20261108.json)

## 真实训练观察与待补证据

准入发布`1ef74ab955aa22951119b743dd848dccbd43d133`独立远端及Mac核同后，10:58:45唯一派发timeout392055/start41964408，controller392056/start41964409，worker392130。11:02:45原result记录`HUMAN_CONDITIONING_PAIR_TRAINED_PENDING_CLOSED_LOOP`、source PASSED、6000实际更新；两组各八个Adam计数均3000。初始真实CPU/GPU动作与原策略各自逐位同、最大跨设备差2.1457672119140625e-6低于2e-5，两次train-only梯度门先于任何真实更新通过。

两组最终均选新增update0，初始及选中验证风险均0.2210775182271997。update0是已有非零机器人历史selected3000策略加零输出人体分支。当前只确认没有被选中的可采用新策略；各500更新保存点的旧能力门与退步幅度须取原曲线后逐点报告，不能仅凭选0断言所有非零点均未过门。不能据本轮结果否定f0或证明缺少人体条件是失败根因。

guard为WORKER_COMPLETED，worker已回收且return0。216秒SETTLED，累计5036/14400秒，r1/r2/r3为1614/2051/1371，无预留；总余9364秒、r2余2449秒。11:03:40资源历史快照：总8282664960/限34292629504B、OOM0、GPU1MiB27度、余盘183381708800B，原暂停门通过。controller/timeout当前退出与锁释放仍需下一次实际观察，不能把初始PID描述或低显存当它们已退出的独立证据。

之后提交独立CPU审计：只读selected/last有限保存、冻结base/统计/teacher/decoder、机器人分支与原起点相等以及人体末层为零的全输入等价证明；未收到完成回执即Linux SSH被远端关闭，45014已回到Windows PowerShell。同一Windows原生第二跳重登返回`Connection refused`，没有到认证；第一跳仍在线，不把它描述为整桥接不可达，也不从失去观察推断训练失败。服务器是否已有`checkpoint_cpu_audit.json`或`q1_human_conditioning_completed_20261108.json`须先查，不直接重复审计；本地尚无该原件。

本地独立`deliver_conditioning_pair.py`入口CLI已过，实际保存数据绘图尚未执行。原始曲线和独立CPU证明取回后才生成两PNG/两PDF/两CSV与证据ZIP；当前无本轮新媒体。若选中策略确实与旧策略全输入等价，则复用原0/7闭环，不重复相同物理或视频。若证明不成立，保持问题并独立验收，不伪称复用。

实际观察记录：[训练终态、具体第二跳错误和待回执审计代码](../logs/session_records/q1_human_conditioning_training_observed_20261108.json)。这是已收到终端输出的记录，不冒充服务器result/dispatch的逐字节原件。下一先补证据和六宫图，再据完整曲线设计不同机制；不重派本轮训练、不延长相同残差拟合、不恢复共享训练或调用goal工具。服务器镜像PENDING，采纳null。
