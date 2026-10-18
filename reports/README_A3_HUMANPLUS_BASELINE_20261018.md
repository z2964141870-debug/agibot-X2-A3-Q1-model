# humanplus 035匹配基线入口

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
