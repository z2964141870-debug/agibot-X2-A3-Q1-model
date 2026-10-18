# humanplus PhysX退出适配HP07

记录2026-10-18，阶段A3_HUMANPLUS_PHYSX_EXIT_20261018_HP07。终态STOPPED，完成十步及CUDA同步，但关闭时陷入STOP回调循环，取得Python栈后主动终止自有监督器，原因SUPERVISOR_SIGNAL:15，未等待满900秒，未复跑。

## 目标与授权

用户在了解HP06超时及可能的退出回调后要求“快修”，授权纠正本次验收。只完成剩余PhysX空场景十步和正常退出；HP05/HP06终态保留，已通过CPU/CUDA四事务及MuJoCo十步不重跑。不改共享环境包或硬件设置，不注册035基线或训练。

## 实现与依据

当前安装IsaacLab-v2.3.2的scripts/tutorials/00_sim/create_empty.py在主循环结束后直接simulation_app.close()。HP07照此退出，移除HP06显式sim.stop()/clear()；不设置私有禁用标志、不修改库源码、不用skip_cleanup强制退出。旧STOP回调循环仍是HP06的可能卡点，不宣称已经确认旧运行栈。

新增script/a3/humanplus_physx_exit_validation.py。fsync进度分别记录AppLauncher、context、reset、每一步返回、CUDA同步、control完成、app.close进入和返回。十步同步结果写control.json，只有关闭返回才写COMPLETE result.json；监督器还要求worker正常退出。30秒后仅单次Python栈转储用于定位，不增加仿真预算。

仍固定CUDA空场景、dt0.01/十步、0机器人环境/策略加载/rollout/训练optimizer。复用不变监督器：三次5秒启动门、>=8GiB启动/4GiB保留、CPU90/GPU85连续两次、900秒期限、原boot/journal/v2内存/OOM门。单次登记/单次运行，STOPPED不自动复跑或扩预算。

## 版本与证据

- 父提交1183ad501e24610b9b0109e1abed06831b2927ae；新代码及官方示例/SimulationContext/SimulationApp固定SHA。
- 标准库生命周期控制：python3 -B -m unittest script.a3.test_humanplus_physx_exit_validation -v，覆盖十步与正常退出、步进失败、CUDA同步失败及关闭失败保留单独控制收据；Mac/新机各4项通过。
- 新根/home/humanplus/YUANQI_A3的登记/运行：/usr/bin/python3 -B -m script.a3.humanplus_physx_exit_validation register及run，各仅一次，非恢复命令。
- data/manifests/a3_humanplus_physx_exit_20261018_HP07_{plan,result,audit}.json；登记/基准data/experiments/a3_humanplus_physx_exit_20261018_HP07/。
- 原始logs/a3_humanplus_physx_exit_20261018_HP07/physx/{progress.jsonl,control.json,result.json,state.json,health.jsonl,console.log}，缺失的完成收据不可视为通过。

## 状态与后续

11:50:43 worker启动，11:50:46 reset返回、十步逐一返回、CUDA同步成功，control.json独立保存；随后进入app.close且未返回。30秒栈显示SimulationApp.close第814行关闭stage，触发IsaacLab _app_control_on_stop_handle_fn第1029行render循环。这次已确认退出回调卡点；HP06仍因未取栈而实际步数未知，不能反向补填。

取得证据后核对/proc cmdline、只SIGTERM本阶段监督器，由原保护清理自有组。29健康同boot、CPU最高79/GPU45°C、0FATAL/新增OOM，停止后健康无异常/自有组无残留。只读审计script/project_tools/humanplus_physx_exit_audit.py单次保存原生日志、进度、control收据和SHA。十步控制通过但完整退出验收未通过，0策略加载/rollout/训练。

已查到当前安装SimulationContext.clear_instance公开类方法第673–680行，会先unsubscribe这项STOP回调再调用父类清理。下一独立HP08在app.close之前调用该方法，保留十步、原健康门和900秒预算；这是已定位代码修复，在用户本轮“快修”授权内继续，不重置HP07或旧停止实验。必须保存本阶段README/索引/current state后再登记。

输入迁移63/63及关键10/10已完成，不再等待用户迁入或索取已有认证。原始数据日志LOCAL_ONLY；scoped审计提交，GitHub写权限仍未恢复，不重复同一失败push。zhy-a3在前台修复期间保持PAUSED。
