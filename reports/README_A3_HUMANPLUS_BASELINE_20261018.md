# humanplus 035匹配基线入口

更新2026-10-18北京时间13:08：HP11终态COMPLETE_MATCHED_BASELINE，独立final audit verified=true，十项8769步已完成。四官方阶段55.160秒，右单腿637步/RMSE0.04577225、原地慢踏882步/0.04307360、前行1652步/0.06567668、原地旋转2604步/0.07607356rad，均未跌倒。六衣服2994步、5/6未倒、zzx5.66s首倒，全程六段算术均RMSE0.1393274808rad；按所有步/关节合并RMSE0.1613716665rad，二者不混用。0optimizer/训练，不宣称035改善或最终未见验收通过。

最终累计26健康，三阶段均三次5秒启动门，阶段内最大采样间隔5.015秒；同boot、CPU最高82/GPU46°C、可用最小20,130,357,248字节、0FATAL/新增OOM/after issues/自有组残留。阶段间未连续监督，之后每新阶段须重查。固定代码/环境源码/输入SHA未变，10次真实完整策略回放成功加载035；HP10缺失的模型诊断收据仍不补填。

对照合同data/manifests/a3_humanplus_baseline_reference_20261018_HP11_contract.json由独立标准库publisher读取已有final audit聚合，不重跑审计或回放。合同包含实际新根输入SHA、逐段完整/跌倒前窗口、根/身体/腕世界误差、原效果门及最终step8选择限制。完整轨迹/输入/动作/接触位于data/experiments/a3_humanplus_baseline_reference_20261018_HP11/{cloth,official}，原始日志位于logs同名目录；全部LOCAL_ONLY。HP11 register/run/audit/publish均已单次完成，禁止重跑/覆盖。

下一步独立登记humanplus固定8更新短训练及最终10项对照，显式重建新根路径/监督器，不调用旧finite_retention_trial的register/run。E45四段包括zzx作为DEVELOPMENT_TRAIN、fresh035 Actor+Critic/空Adam、seed0/16env24步、原reset/奖励/路由/cached实际token/teacher1/fullclip.1/KL名义LR不变；隔离原生Adam提案的8实际位移比例与完整384 teacher .05/.5、同batch固定Critic值Actor目标1e-7门不变。8更新/32提案提交/3072交互/256候选/1728额外forward，保存0/2/4/8，仅最终8完整10回放。HP05已验四事务不重跑，旧科学STOPPED不恢复、不扩预算、不评测早期点。

静态训练URDF资产齐全仍不等于完整训练任务/模型及16环境已验收。需在新登记内明确原生Kit退出的父进程收据、公开clear_instance回调清理、共享包只读、生成缓存/日志位置及同环境协议比较；不能把HP09空场景或HP11 MuJoCo成功冒充训练通过。当前0训练worker，zhy-a3恢复原30分钟主线节奏；真实新异常或最终模型目标达成后保存并暂停。记录scoped本地提交，GitHub写权限未恢复、不重复失败push。以下为本日逐阶段历史。

更新2026-10-18北京时间13:04：HP11六衣服完整基线完成，45.162秒，6×499=2994步全部保留。新机结果5/6不倒，zzx首跌倒5.66s；六段全程关节RMSE依次0.09151693、0.12590409、0.08765224、0.09162262、0.12099006、0.31827894rad（顺序见selection）。这是未更新035的新机开发基线，不能解释为训练改善，也不把旧主机差异当提升。

逐段输入1570维、动作29维、完整tick/关节/根/身体/接触捕获已检查，独立标准库从timeseries重算全程RMSE及第一跌倒时刻，cloth审计verified=true。累计validate+cloth共14健康同boot、CPU最高82/GPU46°C/可用最小20,130,357,248字节，无FATAL/新增OOM/after issues/自有组残留；阶段间不宣称连续监督。small result与cloth_audit已保存。下一原计划四官方5775步，0optimizer/训练，旧HP10不动。

更新2026-10-18北京时间13:03：独立HP11的reference-only校验已通过，15.162秒。新入口先创建并SHA核验六个明确别名，2项标准库路径/缺源/逃逸控制Mac与新机通过；不重复HP10模型诊断，不补旧收据。六段均300行/499策略步，CSV/姿态/时长误差<=1.34e-15，模型body FK误差<=1.66e-15m，重采样脚底最低-3.719mm，符合原5mm门。

HP11 validate独立审计verified=true，4健康同boot、CPU最高82/GPU44°C、可用最小22,208,077,824字节，无FATAL/新增OOM/after issues/自有组残留。0新模型诊断加载/策略forward/回放/optimizer。源码humanplus_baseline_reference.py及其测试/审计入口，旧HP10文件不改；只复用纯数值ABI与固定评测函数，使用已生成并SHA固定的solver桥接缓存，不改旧STOPPED工件。

HP11计划/结果/validate审计：data/manifests/a3_humanplus_baseline_reference_20261018_HP11_{plan,result,validate_audit}.json；新根data/experiments及logs同名前缀。单次登记后按validate→cloth→official顺序各执行一次，监督器/保护/8769步原预算不变；运行命令为/usr/bin/python3 -B -m script.a3.humanplus_baseline_reference run --stage cloth，待本阶段README/索引/状态与scoped提交保存后执行。旧HP10以下记录保留，不重跑。

记录日期2026-10-18。HP10状态STOPPED，尚未开始策略回放或训练。输入迁移与HP09运行控制已完成，不能把旧ZHY效果当新机基线。

## 目标与范围

在humanplus新根独立核验固定035、六衣服参考ABI及四官方完整匹配基线。只读共享环境/home/humanplus/anaconda3/envs/x2-sonic-isaaclab，不调用旧ZHY登记器/监督器/恢复流程，0optimizer。基线计划固定10动作/8769步，含跌倒后全部帧，不裁短或选择早期点。

## 已完成与版本

新增script/a3/humanplus_baseline.py、test_humanplus_baseline.py及script/project_tools/humanplus_baseline_audit.py。6启动/进程组/顺序/新根路径控制及3独立RMSE/跌倒/非有限值控制在Mac与新机各通过，共9项；不重跑HP05–HP09已有控制。

静态闭包通过：103文件20,107,958字节，覆盖两份MJCF、训练URDF及其引用网格、闭链solver库/头文件。唯一新实化为固定vendor的liba3_ankle_waist_solver.a，33,578字节，SHA ceced0794654c6534a1fd444dce123ad6dfdacb132725873edc63039e8c5be15。Git LFS单次取回并核SHA，vendor fe6868ba37034f89b912f0fb851bce19f120266d仍干净；未改共享环境。清单data/manifests/a3_humanplus_policy_assets_20261018.json。这是静态文件闭包，不是完整训练环境/任务实例化证明。

HP10登记父HEAD affa71928585da60f176a40d5def0197db17fc3c；实际代码/输入/环境SHA、主机及budget见data/manifests/a3_humanplus_baseline_20261018_HP10_plan.json。63迁入输入使用前再次SHA核验；035 PT SHA保持9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d。按历史合同只提取输入选择/协议，路径重建到新根，旧boot/cgroup/健康/效果不作为新机通过证明。

## HP10停止证据

北京时间12:54:21–12:54:37，单次validate在reference_contract读取validation/references/G_0113_zj_train__183926_zj_walk.csv时FileNotFoundError，入口漏建六个reference链接。此前已执行有限网络张量检查、A3Policy构建、640编码输入/930本体观测维度检查及CUDA同步；依据是异常栈位于这些语句之后。缺少最终模型/网络计数收据，不补填验收结果，不记完整模型验证通过。0策略forward、0回放、0optimizer。

独立标准库终态审计verified=false：4健康采样同boot afb6796b-3883-4df4-99ea-bc381c52ea13，CPU最高79/GPU44°C，可用最小22,235,348,992字节，0FATAL/新增OOM，停止后无自有组残留/after issues。不是已证实机器故障。原三次5秒启动、8GiB/4GiB、90/85连续两次、900秒及journal/v2门未改。

计划/结果/审计为data/manifests/a3_humanplus_baseline_20261018_HP10_{plan,result,validate_audit}.json；原始日志logs/a3_humanplus_baseline_20261018_HP10/validate/console.log与health/state，项目生成solver桥接缓存位于data/experiments同名目录/solver_bridge。HP10终态及注册源码冻结，禁止run/register/audit复跑或修改终态。

## 保存与下一步

原zhy-a3前台纠错期间暂停，其他自动任务不变。先保存本失败README/索引/current state及scoped本地提交，再独立修复参考入口并验证路径适配；不重复已执行的模型诊断，不重跑空场景控制。后续真实回放自身加载模型属于固定10回放预算，不将其当额外诊断。不恢复科学STOPPED或扩预算。

所有原始工件LOCAL_ONLY，GitHub SSH写权限未恢复，记录LOCAL_COMMITTED_UNPUSHED，不重复同一失败push。闭环基线尚未完成，无新035效果或提升证据。
