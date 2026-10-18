# humanplus PhysX原生退出协议HP09

记录2026-10-18北京时间12:01:58–12:02:13，阶段A3_HUMANPLUS_PHYSX_NATIVE_20261018_HP09，终态COMPLETE_PHYSX_CONTROL，独立审计verified_control_and_exit=true。修复完成，十步同步、回调清理、原生关闭与真实退出码0均有记录。

## 目标与修复

用户本轮要求“快修”。HP07取得STOP回调死循环栈，HP08公开clear_instance已修复死循环，但默认原生关闭直接退出进程，Python关闭后收据未写出，原监督器拒绝，旧终态保留。先保存HP07/HP08阶段README与索引，不覆盖或复跑旧阶段。

HP09只修收据进程边界：受原监督器管理的标准库broker启动原生Kit子进程，同一自有进程组、不新建后台会话；原生子进程按HP08同样十步、同步、clear_instance、app.close，物理与关闭参数不变。broker保存实际returncode，只有退出0、十步逐一返回、同步control收据、清理先于原生关闭、Kit包含Stage closed及释放framework记录同时成立，才生成COMPLETE最终收据。不依赖原生关闭返回Python，不将启动/零退出单独冒充通过，不用skip_cleanup或强制成功。

## 版本、预算与证据

- 新代码script/a3/humanplus_physx_native_validation.py及标准库test_humanplus_physx_native_validation.py。5控制分别检查完整收据、异常退出、缺日志、缺步和错误清理顺序，Mac/新机各5项通过。连同HP07的4项和HP08的2项，本轮新增11项标准库控制两端均通过，已通过旧控制未重跑。
- 固定HP08源码/环境源SHA，单一变量为标准库父进程验收原生退出；0策略加载/rollout/训练optimizer。父Git及代码/输入SHA见plan。
- 原CUDA dt0.01/10步、3次5秒8GiB启动、4GiB保留、CPU90/GPU85连续两次、900秒期限及boot/journal/v2/OOM门不改。单次登记/运行，不自动复跑或扩预算。
- 新根/home/humanplus/YUANQI_A3，/usr/bin/python3 -B -m script.a3.humanplus_physx_native_validation register及run各一次。
- data/manifests/a3_humanplus_physx_native_20261018_HP09_{plan,result,audit}.json；data/experiments及logs同名目录。logs/physx含control、native_exit、result、progress、state、health、console及native_console，缺任意必需证据拒绝。

## 实际结果

单次登记、单次run，阶段15.160秒。三采样启动门通过，原生子进程12:02:08启动、12:02:11完成十步/CUDA同步和clear_instance；Kit记录stage正常关闭及释放framework，broker12:02:12明确记录native_returncode=0、保存最终收据，监督器正常退出0。不是靠Python app.close返回，也不是仅按零退出猜测成功。进度/control/native_exit/result/state和原生日志均登记大小SHA。

4份健康采样同boot afb6796b-3883-4df4-99ea-bc381c52ea13，CPU最高81/GPU44°C，可用最小22,461,792,256字节、最大采样间隔5.0015秒，0FATAL/新增OOM；独立终态审计确认代码/环境/输入SHA未变、停止后健康无异常/自有组无存活进程。没有触发30秒栈转储。Kit原始日志396,912字节/SHA5ef6e46c2ae334f5ecaf190ea93210f3f3b573eb09c2b5c61a7577541de0a849，已归档并复核；IsaacLab日志392字节及全部原始日志同样LOCAL_ONLY。

HP07/HP08的STOPPED仍保留：前者实际定位退出回调循环，后者解决循环但不适配原生退出收据。HP06旧步数仍未知，不补填或复跑。本轮未修改共享包源码/配置/依赖、驱动/功率/BIOS或权限，Isaac默认环境内运行日志写入如实保留，未清缓存或干预其他任务。

## 保存与后续

阶段成功仍仅为空场景运行控制；连同HP05已通过CPU/CUDA四事务和MuJoCo10步，基本运行控制齐全，不等于训练资产、035新匹配基线或策略进步。0策略模型加载/rollout/训练optimizer；下一步在新根独立登记完整训练资产/035模型验证及六衣服四官方匹配基线，再按原固定8更新方案执行，不沿用旧ZHY效果数值。无需再做同类空场景验收。

原始数据日志LOCAL_ONLY，GitHub写权限未恢复，不重复已知失败push；README/索引/current state、小计划/结果/审计及代码经scoped审计提交，LOCAL_COMMITTED_UNPUSHED，提交号见Git。前台故障修复已完成，更新原zhy-a3为最新恢复入口并继续原30分钟主线节奏；旧yuanqi-a3/hp3090及每日研究不变。不得恢复HP05–HP08或旧科学STOPPED实验，只有新登记可接续；后续首新异常仍停分支留证。
