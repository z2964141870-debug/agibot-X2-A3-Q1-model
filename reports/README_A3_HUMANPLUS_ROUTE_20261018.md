# A3 humanplus 原生训练路由单变量

记录日期：2026-10-18。D03保存证据审查完成；下一独立实现、冻结和登记，当前尚未开始新科学试验。

## 目标与范围

持续改善官方035的衣服/SMPL完整动作，同时保留四官方能力。只在humanplus:/home/humanplus/YUANQI_A3计算；Mac仅小代码和记录。HP17、HP18已完整拒绝，旧权重已按S01删除，不加载、恢复或重建。

## 已完成的差异审查

新review_humanplus_route_evidence仅一次读取保存JSON及固定源码，72来源大小/SHA登记在data/manifests/a3_humanplus_route_design_20261018_D03.json。比较HP11/HP16/HP17/HP18完整十动作及逐四分之一窗口、共同跌倒前窗口、候选loss/teacher和实际路由。0张量反序列化、策略forward、仿真、optimizer；没有执行旧checked或旧审计。

D01实际a3_fast1084/3072=35.2865%，D02为1275/3072=41.5039%，其余G1原生参考驱动同一共享输出头；评测固定a3_fast。D02全部32提案首scale1接受，原硬teacher最高RMSE0.01708323，但完整闭环仍恶化。zzx共同未倒前RMSE仅+0.0548%，全程+22.6076%，后续跌倒/轨迹偏离放大了总指标；四官方与其他不倒衣服仍有误差退化，因此不能仅归咎跌倒后数据。不得把窗口选择当新验收门或宣称退化因果已找到。

本地既有OUTCOME_CURVE在10/13已指出路由差异但选择曝光量，未试路由对齐。已有MimicX/PriMo/D2R/CPO与LoRA研究不重筛、不下载框架。该选择是项目消融，不声明新颖性。

## 唯一变量与保留条件

从D02配方fresh035重新开始，只把manager_env.commands.motion.encoder_sample_probs.g1由1设0，a3_fast仍1、其他0。保留active_encoders原g1/a3_fast、原生G1配对coactivation及辅助loss；原顺序下a3_fast覆盖配对G1的动作token，不改vendor或手改token。需在真实训练内逐rollout留证a3_fast384/G1独占0，并保持实际cached token一致。部署所有动作仍同一个策略，不用任务/评测身份开关。

原两head key14877参数、Critic/空Adam、E45四段、seed0/16env24步、原reset/奖励/LR/KL/fullclip.1、D02四官方各96 preclip标签loss系数1、原E11全部384硬门RMSE<=.05/maxabs<=.5、固定1..1/128有限候选、8更新32提交3072交互、外层调用上限2440均保持。不扫LR/权重/batch，不增加预算。训练状态分布和token随原生路由变化是本变量的后果，不能声称与D02批次逐值相同。

## 实施与验证入口

下一独立route trainer/entry/verification/trial/audit，不能只替换trainer字符串而保留HP18硬绑定。保存0/2/4/8和完整两key/冻结/Adam整数ID矩状态/token/配置/调用收据；配置允许差异精确限定为g1采样概率与既有存储字段。新输入闭包只纳入实际仍在且必要的文件，不能递归继承已删旧权重。新编号按真实账本核对后冻结登记，当前D03不是执行合同。

运行保留实时三次5秒启动门（>=8GiB、无其他计算、GPU空闲、可靠遥测）、运行>=4GiB、CPU90/GPU85两次5秒、900秒/阶段、cgroup v2/全boot journal连续保护、独立Kit路径与正常关闭协议。任何科学STOPPED不续跑/扩预算/评早期，机器首异常保存恢复并暂停heartbeat，只终止自有组。

只有32提交与最终8独立保存验证完成才十项8769完整步，HP11基线不重放。原效果门：六衣服全部不倒/均RMSE改善>=1%，原不倒及四官方关节<=+2%、世界根腕<=+5%、官方无新增跌倒。通过仍是开发适配，需独立seed与真正未见会话。对齐路由可能无收益或损害通用性，不提前声称提升。

## 保存状态

README/索引/current state/小清单及审查脚本scoped审diff和audit_git_payload后本地提交。GitHub已知publickey写失败不重复push，LOCAL_COMMITTED_UNPUSHED；原始数组日志及必要模型LOCAL_ONLY，不称备份。035仍baseline，当前无新改进模型。

## D03实施与冻结完成

独立humanplus_route_trial/trainer/protocol/entry/verification和phase/stop审计完成；旧HP18源码/行为不变。原D02训练类仅增加真实采样mask检查及rollout计数断言，不覆盖loss/optimizer/接受规则。入口在创建16环境前核对完整路由配置，保存核验精确还原唯一g1概率差异后与HP18原配置比较；原G1/a3_fast两个网络和G1配对必须保留。实际完整运行仍待新登记。

8项新标准库控制Mac及新机各通过，覆盖存储字段/唯一采样概率允许、奖励LR及配对偷改拒绝、网络/路由变更拒绝、完整及部分预算/顺序、真实采样mask与缓存链继承、独立绑定、原最终效果与preclip边界不变。不是数值或仿真成功证明。

新freeze仅单次，319源码/456实际存在输入/10环境源码SHA及大小通过，输入PT仅035和E11/E12三份必要资产，无任何已删训练PT。配置、D02标签来源和D03差异记录纳入新闭包；固定外层调用上限2440，无新增forward。readiness为data/manifests/a3_humanplus_route_readiness_20261018.json。新执行编号HP19/R25已核对无占用，下一提交实际源码与冻结记录后独立register，再按实时资源门执行train。原基线8769步与旧科学阶段不重跑。

## HP19单次登记

实现提交05794c6后独立register完成，319源码/458输入/10环境来源固定。新目录data/experiments/a3_humanplus_route_20261018_HP19、data/training同名/R25_001_s0、logs同名。计划data/manifests/a3_humanplus_route_20261018_HP19_plan.json。登记资源采样不是启动门，下一supervisor三次5秒实时核验后单次train；0新训练尚未记成功。

```bash
/usr/bin/python3 -B -m script.a3.humanplus_route_trial register
/usr/bin/python3 -B -m script.a3.humanplus_route_trial run --stage train
/usr/bin/python3 -B -m script.project_tools.humanplus_route_audit --stage train
# 只在train与独立审计成功后依序verify-training/cloth/official/effect，每阶段单次审计保存。
```

## 18:32 真实训练与独立审计完成

HP19单次train完成8更新/32提案32提交/3072交互，全部8个rollout实际a3_fast384、G1独占0，原生配对mask为[1,0,0,1]。64候选含live检查，推导外层调用872/2440；fresh035对D02固定preclip标签maxabs2.0265579223632812e-6。原E11硬teacher最高RMSE0.017082905396819115/maxabs0.07597750425338745，不放宽阈值。

原生16环境cuda:0完成/CUDA同步、专属Kit日志实际设置/native PID/清回调/正常关闭退出0通过。train独立审计verified=true，60.162秒，13健康同boot、CPU最高88/GPU46°C、最低可用19223252992字节，0FATAL/新增OOM/自有组残留；未到温度停止门，旧快照不是未来启动证据。保存0/2/4/8尚待独立张量核验，0新效果回放，无035改善证据。下一仅新合同verify-training。

## 保存状态核验通过

单次verify-training及独立审计verified=true。0/2/4/8大小SHA、fresh035 Actor/Critic和空Adam、原两key变化/其他Actor参数与buffer冻结、Adam整数ID所有权/矩形状/有限性/4倍step计数、实际cached token/首概率、D02 loss及旧teacher收据均通过。完整manager_env/algo配置只允许原存储字段与登记g1采样概率差异，两个encoder及原配对保留；8次真实路由共3072个a3_fast。累计17健康/0新增异常/无残留，阶段间不作连续证明。下一仅最终8六衣服2994步完整回放，不评早期点。

## 六衣服完整效果完成

六段各499步共2994完整步已单次完成，cloth独立审计verified=true，5/6不倒、zzx仍跌倒。算术均RMSE0.1525250051006924rad，高于035的0.13932748084287783；当前候选不满足衣服门。累计28健康同boot、CPU最高88/GPU46°C、0FATAL/新增OOM/残留。原始preclip捕获按固定float64+clip边界与轨迹严格相等，所有观察/接触/状态/tick完整；不重放衣服。下一仅原预算剩余四官方5775步及十项汇总，不能用训练/部分动作改善冒称整体提升。
