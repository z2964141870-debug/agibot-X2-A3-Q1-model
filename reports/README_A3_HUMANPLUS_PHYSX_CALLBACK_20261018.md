# humanplus PhysX STOP回调修复HP08

记录2026-10-18，阶段A3_HUMANPLUS_PHYSX_CALLBACK_20261018_HP08，终态STOPPED_MISSING_POST_CLOSE_RECEIPT。十步/同步及clear_instance完成，原生关闭退出0、未返回Python，原监督器因缺失关闭后result.json拒绝，原终态保留。

## 目标与依据

用户要求“快修”，本轮继续修复HP07已经取得栈证据的退出死循环。HP07十步及CUDA同步通过，但app.close关闭stage时进入IsaacLab STOP回调render循环；已主动终止并保存STOPPED，不覆盖或重跑。HP06实际步数仍未知，HP05已经通过控制不重跑。

当前安装SimulationContext.clear_instance第673–680行先unsubscribe _app_control_on_stop_handle再调用父类清理。HP08唯一功能变量是在app.close之前使用此公开类方法。复用HP07十步控制、分段进度与独立control/退出收据，不使用私有开关、skip_cleanup/强制成功，不改库包/配置/硬件。退出控制是本工程启动适配，不是模型算法或性能提升。

## 版本、预算与命令

- 新代码script/a3/humanplus_physx_callback_validation.py，引用固定HP07源码和已安装官方示例/库源码SHA，旧注册文件不改。
- 两项标准库回调顺序测试test_humanplus_physx_callback_validation：注销在原生关闭前完成、注销失败不能声称退出；Mac/新机各2项通过。HP07原4项生命周期控制已两端通过，不再次累加。
- 仍为空CUDA PhysX dt0.01/10步，0机器人环境/策略加载/rollout/训练optimizer；原三次5秒8GiB启动、4GiB保留、CPU90/GPU85连续两次、900秒期限和boot/kernel/v2/OOM门不变。
- 新根/home/humanplus/YUANQI_A3：/usr/bin/python3 -B -m script.a3.humanplus_physx_callback_validation register及run各一次；终态不恢复或复跑。
- data/manifests/a3_humanplus_physx_callback_20261018_HP08_{plan,result,audit}.json，登记data/experiments/a3_humanplus_physx_callback_20261018_HP08/，原始logs同名/physx/。
- 只读终态审计script/project_tools/humanplus_physx_callback_audit.py复用可传入runtime的HP07审计格式；不引入张量或重跑控制。

## 当前状态与保存

11:56:00十步和CUDA同步完成，11:56:01 clear_instance返回，原生Kit日志记录Stage closed、Simulation App Shutting Down及shutting down app and releasing framework，进程随后退出。注册监督器仅在poll()==0后才读取result.json，本次实际走到该读取并报FileNotFoundError，因此可从固定代码路径推断退出0，但没有独立原始returncode字段。原state仍STOPPED，不据此改写通过。

4健康同boot、CPU最高78/GPU43°C、0FATAL/新增OOM，15.164秒结束；审计确认停止后无健康问题/自有组残留。STOP回调死循环已消除，剩余为原生关闭直接退出进程、Python后置收据无法写出的协议适配问题。当前SimulationApp默认fast_shutdown=True，公开配置会传给/app/fastShutdown；不把改该配置当必要条件，不换运行环境。

同一“快修”授权内继续HP09，保持原仿真条件，以标准库broker作受监督worker、原生Kit子进程留在同一自有进程组。broker明确保存子进程真实退出码，再同时检查十步同步收据/回调清理/原生关闭日志，全部满足后才写最终COMPLETE，避免原生退出绕过Python后置代码。先保存本阶段README及恢复索引，不覆盖HP08，不自动续训或评测。

原始数据日志LOCAL_ONLY，GitHub原写权限未恢复；阶段README/索引/current state和小清单scoped审计提交。zhy-a3前台修复期间继续PAUSED；无新增认证/手动迁移需求。成功后再独立登记035匹配基线，不能以空场景通过代替训练资产或实际效果。
