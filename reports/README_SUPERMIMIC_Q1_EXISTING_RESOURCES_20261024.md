# 现有资源继续推进SuperMimic for Q1

记录日期：2026-10-24，北京时间。阶段：SM-Q1-EXISTING-R02。状态：23:08独立PHUMA R02已登记并经存储guard启动，23:12:30核验300轮/19200交互、WM69组/actor19次，469冻结输入与活跃进程身份通过；尚无最终配对评测或移动效果。

## 目标与范围

用户停止代理处理BONES获取，由师兄解决，后续到位由用户通知。当前继续用现有资源训练Q1平地全身移动跟随，保持可微世界模型、想象BPTT和critic的方法复现；不恢复旧A3或STOPPED实验。

## 完成内容与关键决定

PHUMA基础库已完整重算：340段训练/1227.10秒，移动62段/221.76秒；development6段/21.26秒，含1移动/3.96秒；test7段/27.56秒，含3移动/11.72秒。来源、原字节、划分隔离及原19门保持。现成库不等待30分钟工程目标即可用于独立有限学习试验，不能称已复现论文规模或具有充分未见移动覆盖。

现有AMASS已索引17265标准SMPL-X动作/120Hz/16betas，标称56.34小时；男女身体模型已接收并通过有限样本16维FK。不再索要或下载这些资源。AMASS尚未Q1重定向/质检，暂不混入固定PHUMA实验；后续独立原速接入。

恢复已完成的PHUMA D03B原始收据，服务器/本地SHA一致，90组共同轨迹预测、0新物理步或optimizer。开发移动H32根速度误差为CV的2.56倍，而关节预测已改善；扩库不是唯一假设，也不把此预测比较当唯一训练失败原因。详见[D03B](README_SUPERMIMIC_Q1_PHUMA_MODEL_20261024.md)。

64环境/4槽MuJoCo后端此前逐状态、连续DR/噪声/reset和H32梯度验证通过，收据SHA5ed98f8c…重新核对，未重复完整验证。拟直接用已完成库，独立随机初始化R02、64环境/4线程/4槽、actor B128、WM B64每次10批、H32、12000采集轮/768000转移，学习率/奖励/优化器沿用原trainer。多项条件较R01改变，因此以同轮step0/final配对为证据，不归因于单一数据规模因素。

## 版本、命令与产物

权威Mac工作副本`/tmp/yuanqi-supermimic-q1-20261023`、Q1、恢复前HEAD`0bf5fc6e1463f9468e28749442bb659cbe135ac5`。服务器huanyu:/opt/data/private/YuanQi_Q1，Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。

现有入口为`script/supermimic_q1_expanded/expanded_pipeline.py register/run`，库`data/supermimic_for_q1/PHUMA_P05_base_library_20261024`、后端收据`PHUMA_P05_slots_V01_20261024/receipt.json`。新实验拟`PHUMA_R02_expanded64_H32_12000_20261024`；登记与启动后立即补实际命令、冻结清单、进程及日志。

预算配置`script/supermimic_q1_expanded/phuma_R02_20261024.json`，用现有`runtime_guard.py`包裹整条流水线：盘余至少100GiB、实验唯一分配最多12GiB、现有源库目录250GiB；每1000轮存checkpoint，最多12个中间点及初/末，预计权重不足3GiB。guard只统计自己的实验/库目录且不跟随链接，实验stdout在logs另记；不会下载或复制原始数据。旧trainer温度86摄氏度、内存30GiB、新增OOM停止和controller 900秒训练无进度保护保持。

登记固定全部353段step0/final实际无支撑MuJoCo评测及原轨迹审计，按三划分记录完成率、关节/根误差和实际水平位移；旧门为完整无提前失败、关节RMSE<=0.15rad、根位置RMSE<=0.10m。开发/测试不参与optimizer。尚无本轮效果，不把仅完整或站稳当移动成功。

## 实际验证与清理

22:52:59实核huanyu boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`、GPU38摄氏度/空闲17788MiB、无现行训练/转换进程，内存26837823488/34359738368字节、OOM14，数据盘可用1965787930624字节。首次命令发现服务器无rg，后续用grep读取进程，未安装新依赖。

BONES失败D01-A原目录和收据保留，0新参考/optimizer；不继续诊断失败。临时下载SSH会话23954退出255、服务器18866无监听，训练18864与回放18865保留。临时下载令牌已网页撤销，刷新显示无令牌，内存令牌与传输密钥变量已清空；凭据不入文件/Git。截图仅本机临时证据`/tmp/yuanqi-bones-token-revoked-20261024.png`，不随Git发布账户页面。

## 保存与下一步

本阶段更新规则、当前状态与报告索引，连同此前未记录的模型槽/D03B小收据审计后提交、推送Q1并独立核验。未提交下载准备脚本保留，未作为训练依赖。所有原模型、数据、数组和新模型均LOCAL_ONLY；不声称网盘备份。

下一先核代码身份并登记R02，经存储guard启动，核实际采集/世界模型与actor更新和保护状态后保存可恢复记录；若失败，留终态后另立实验，不覆盖重启。之后按开发效果选择下一独立方案，AMASS转换与BONES到位保持独立。

## R02登记与实际启动

恢复记录/配置版本`b6d310897b89d1b8664f6165867bf1256cf49208`已显式推送Q1，独立远端ref一致。7份关键入口/核心trainer/guard的本地与服务器SHA完全一致；既有后端和原输入再次按原注册器核验。存储guard本地8项控制通过，原物理后端验证未重复。

实际登记命令如下，已退出0。注册器输出先有旧模板的中间摘要，最终expanded spec才是权威：144781字节/SHA`1eae94ba37418b81140605de95170279f2dd37dd539565cb4e691d892023c742`，64环境/4槽、12000轮/768000交互、train340/dev6/test7、随机初始化。相同目录不得再次登记或启动。

```bash
cd /opt/data/private/YuanQi_Q1
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
"$PY" -u script/supermimic_q1_expanded/expanded_pipeline.py register \
  --root "$PWD" \
  --experiment "$PWD/data/supermimic_for_q1/PHUMA_R02_expanded64_H32_12000_20261024" \
  --library "$PWD/data/supermimic_for_q1/PHUMA_P05_base_library_20261024" \
  --backend-receipt "$PWD/data/supermimic_for_q1/PHUMA_P05_slots_V01_20261024/receipt.json" \
  --model-slots 4 --steps 12000 --seed 20261024 \
  --code-commit b6d310897b89d1b8664f6165867bf1256cf49208
```

23:08:03以独立session的Popen启动已发布runtime_guard，stdin关闭、stdout在`logs/PHUMA_R02_expanded64_H32_12000_20261024/guard.log`；guard再启动expanded pipeline，命令完整记录于`launch.json`。三个角色分别为guard825182/start_ticks3614648876、controller825183/start_ticks3614648887、trainer825189；boot仍d994e71f…。PID仅是本次证据，恢复必须比较boot/start_ticks/命令，不据旧PID盲目发信号。

实际状态路径均在新实验目录：`launch.json`、`spec.json`、`storage_guard.json`、`controller_status.json`、`run/status.json`。训练stdout为`logs/PHUMA_R02_expanded64_H32_12000_20261024/train.log`，流水线事件为同目录`pipeline.log`。guard每5秒重核100GiB/12GiB/250GiB预算，原900秒无训练进度停止有效；本轮无自动重启。

23:08:45观察trainer125轮/8000交互、0WM/actor（尚未到128轮预热起点），20.05秒、OOM14、GPU38摄氏度、cgroup29426081792字节；实验分配68894720字节、盘余1965714333696字节。CONTACT_PROTOCOL_PREFLIGHT与PHUMA_ONLY_RANDOM_INITIALIZATION_ADMITTED通过。此为真实采集证据，不是优化器已运行或移动成功。下一核验预热后实际网络更新。

原样小副本：[冻结spec](../data/manifests/supermimic_q1_phuma_R02_spec_20261024.json)、[启动收据](../data/manifests/supermimic_q1_phuma_R02_launch_20261024.json)。训练/评测大工件留服务器LOCAL_ONLY。

## 23:12实际更新与恢复检查点

独立轻量只读核验退出0，生成`startup_verification.json`（7714字节/SHA`67fd5dc4a9353003eada5196ef86b7782e239e5b9d6477cbd53595acd4bdedaa`）。[原样小副本](../data/manifests/supermimic_q1_phuma_R02_startup_20261024.json)保留当时controller/guard/指标、初始权重和日志前缀身份。

实际300/12000轮、19200交互，WM69组更新（每组10批）、actor19次；actor梯度范数0.06262、critic loss0.08216均有限，证明采集及世界模型/BPTT/critic更新路径已运行，未做新权重张量审计或固定片效果评测。全部469个冻结文件逐SHA相符，spec/config与启动收据相符，guard/controller/trainer活跃、boot/start_ticks/命令匹配。实际温度40摄氏度、cgroup29916872704字节、OOM14，约227.49秒训练；不据早期在线回报宣称学会移动。

当前停止/恢复必须先读本文件及新实验`controller_status.json`、`storage_guard.json`、`run/status.json`。无信号时原有限流水线继续到12000轮并自动做训练保存审计、全部353片初末评测/轨迹审计和比较；任一非零退出即停止后续阶段，失败不自动重启。若长时域根部仍差，依据development和同轮配对结果另立方案；test只作最终报告，不据其调参。当前还没有达到30分钟库或策略成功目标，AMASS准入为后续独立数据工作。

## 第1000轮检查准备

23:46:44原trainer实际1275轮/81600交互、WM459组/actor409次，GPU39摄氏度/OOM14；第1000轮权重已保存205348199字节。独立扩库B016已退出0且worker回收，细节见[P06](README_SUPERMIMIC_Q1_PHUMA_CONCURRENT_EXPANSION_20261024.md)。准备新独立只读入口`script/supermimic_q1_sources/audit_phuma_R02_checkpoint.py`，不改469冻结文件；本地语法检查通过，尚未实际运行。

本入口适配R02的1000/12000轮、64环境、b6d3108训练版本及469冻结文件；用独立2/5整数节奏重算WM349组、actor299次，并按已存manifest重算三优化器Adam次数。CPU/mmap、单线程、无新物理步/策略调用/optimizer；初始内存须低于28.75GiB，逐张量/冻结文件在29.5GiB前终止且OOM固定14，低于原30GiB训练线。只检查原step0与step1000，保存前后SHA，不读取尚未完成的metrics末行；此诊断不能代替最终动力学效果。

发布部署并确认无转换worker后执行，输出使用新`checkpoint1000_audit.json`排他创建；随后立即记录真实结果，不把准备记为通过。
