# humanplus剩余PhysX验收HP06

记录：2026-10-18北京时间10:49:45–11:04:41。阶段A3_HUMANPLUS_PHYSX_20261018_HP06。终态STOPPED，原因RuntimeError:STAGE_DEADLINE_900_SECONDS。EULA启动配置修复有效，但没有十步完成收据，实际完成步数未知，不算PhysX验收通过。

## 目标与授权

用户在HP05停止后明确回复“允许”，授权修正Isaac Sim非交互启动配置并仅补做未完成PhysX验收。HP05的STOPPED及CPU/fused CUDA四事务、MuJoCo固定MJCF十步通过证据保留，均不重跑。新机仅humanplus@100.101.30.68、/home/humanplus/YUANQI_A3，Mac只小代码/记录。

## 实现与预算

新增script/a3/humanplus_physx_validation.py；只在本次worker环境提供OMNI_KIT_ACCEPT_EULA=YES，在IsaacLab导入之前生效，并使用无缓冲Python日志。本机isaacsim/kit/kit_app.py第19行读取该变量，此源码与IsaacLab/Python/新代码固定SHA；未修改候选x2-sonic-isaaclab包源码、配置或安装依赖。Isaac自身在环境内默认路径写了运行日志，这不等于完全没有环境目录写入。HP05源代码及原计划SHA不修改，新登记引用其独立审计及完成收据。

复用已验humanplus_supervisor，三次5秒启动门、8GiB启动/4GiB保留、CPU90/GPU85连续两次5秒、900秒总阶段期限、内核完整前缀/cgroup v2/OOM门不变。只创建CUDA空场景、固定dt0.01及10步，无机器人环境、策略模型、rollout、训练optimizer，不校验训练资产闭包。此次新的STOPPED仍不自动重试/扩预算。

## 版本、命令与产物

- 父提交3e76d87ae025f997afc3cbac7b202851158a7602；新机main干净核验后新增本阶段文件。
- Mac标准库AST语法检查通过。无监督器改动，不为累加次数重跑既有13控制、旧14遥测测试、4张量控制或MuJoCo。
- 新根执行/usr/bin/python3 -B -m script.a3.humanplus_physx_validation register，然后run；两者均已单次执行，禁止复跑。本README中的命令只供审计，不是恢复指令。
- 计划data/manifests/a3_humanplus_physx_20261018_HP06_plan.json，结果同前缀_result.json；登记/基准在data/experiments/a3_humanplus_physx_20261018_HP06/。
- 原始日志logs/a3_humanplus_physx_20261018_HP06/physx/{state.json,health.jsonl,console.log,result.json}，未完成时可能没有result.json；顶层result包含实际终态及文件SHA。

## 实测结果与边界

三次启动采样门通过，Kit在约3.6秒记录Simulation App Startup Complete，随后创建/physicsScene及device 0的PhysX tensors上下文，约3.953秒记录onResume。EULA EOF不再出现。console保持5,048字节，worker持续占用约一个CPU核，未产生十步完成收据。监督器预留4秒采样截止余量，实际896.670秒触发原900秒阶段期限，清理自有进程组后退出1；没有扩预算或第二次启动。

181份健康采样同boot afb6796b-3883-4df4-99ea-bc381c52ea13，CPU最高78°C/GPU44°C，可用最小20,334,821,376字节，最大采样间隔5.020秒，0FATAL/新增OOM。只读终态审计确认登记代码和环境源码SHA未变、停止后健康无新增问题、自有进程组无存活进程。没有证据证明硬件故障；也不能以正常健康指标推断PhysX通过。

worker把result.json写在sim.stop()/sim.clear()/app.close()之后，因而缺少收据不能证明实际0步，也不能证明已经完成10步。当前证据无法区分reset、步进或退出阶段卡住。0策略模型加载/策略rollout/训练optimizer，035仍baseline，无新效果；新匹配基线和短训练均未登记。

## 只读定位与后续方案

固定环境源码simulation_context.py第248–256行在非terminal模式订阅STOP事件；第1010–1030行回调在标志未禁用时循环render直到timeline重新playing。reset第512–531行仅在reset期间禁用此回调；SimulationApp源码第226行设ISAAC_LAUNCHED_FROM_TERMINAL=False。本worker在十步后显式sim.stop()，存在进入该循环的源码路径，这是待验证假设，未取得运行栈，不能记为已确认根因。Kit中也没有SimulationApp.close开始记录，但日志缺失仍不能确认实际卡点。没有attach、sudo、安装诊断器、删缓存、改驱动或IOMMU。

后续优先只读审查当前安装版本的官方有限步示例与退出调用合同。若明确解除本次停止约束后另登记纠错验收，应在启动/reset/每步/CUDA同步/退出前后保存独立进度收据，使十步证据与进程正常退出可分别审计，保留相同十步、900秒和健康阈值。禁止改HP06源码/计划/终态后复跑，也不重跑已通过CPU/CUDA和MuJoCo控制。

新增只读审计script/project_tools/humanplus_physx_audit.py已单次执行：/usr/bin/python3 -B -m script.project_tools.humanplus_physx_audit；审计文件同前缀_audit.json。原始health.jsonl为11,757,653字节，仅保留服务器logs，不纳Git。相关Kit日志397,352字节、SHA980e7fec9facec3634031b7d6998cb433daeaade21f634d0db3c5d5126e9d3cd；IsaacLab日志392字节、SHAa9e0a284f438a7a9611eca2d900f04041b13481e88402b2aeb3c5cd0700afac9，已复制到本阶段logs并逐一核验。控制台提到previous crash的历史dump，未读取/删除，也未归因于HP06。

## 保存与恢复

数据/原始日志LOCAL_ONLY，无已核验网盘备份。本页、索引/current state、计划/终态/独立审计纳入scoped提交；GitHubSSH既有publickey失败，未恢复写权限，不重复相同失败push，状态LOCAL_COMMITTED_UNPUSHED，提交号以Git历史为准。

zhy-a3保持PAUSED并保存HP06终态恢复入口，不自动复跑或转入035基线/训练。用户已授权的本次纠错已单次执行，迁移63/63及关键10/10完成；不再等待用户搬文件或重复索取已有认证。旧hp3090不计算、ZHY只保留迁移来源，旧停止实验不恢复。
