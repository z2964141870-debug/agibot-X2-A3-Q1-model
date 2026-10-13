# A3已验证启动接口的冻结闭环E26

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_FROZEN_20261013_E26。已编译、两项启动契约测试通过、登记READY，尚未回放。

## 目标与决定

E24/E25都是新增包装器在加载策略前失败，实际0回放/0optimizer。失败记录保留。E26以独立编号改为透明进程接口代理：仅路由到只读捕获入口并保持受保护进程组，其他属性完整转发；所有评测参数显式提供。复用E24六段完整ABI证据，不重算接口或已做的实验。

计算协议不变：六段合格开发参考、官方035 SHA `9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d`、固定vendor、30Hz stride1/50Hz499步各段/合计2994、seed0、零延迟、无缓冲、0optimizer。原下蹲拒绝。判据、保护、版本和输入SHA继承[E24](README_A3_SMPL_FROZEN_20261013.md)，不因入口错误放宽。

## 验证与版本

新增`frozen_evaluation_launch.py`、`test_frozen_evaluation_launch.py`、`smpl_frozen_validated.py`。两项测试在ZHY执行完整evaluate_explicit调用、使用假checkpoint/CSV/指标和假进程，验证参数、STDOUT/TimeoutExpired、入口路由、进程组、stride与完整长度，并完成manifest写入；没有启动真实仿真、模型加载或CUDA。测试不是动力学成功证据。实际小型测试receipt在新实验目录。

预算六段一次回放，run与独立verify各900秒，5秒遥测/CPU90与GPU85连续两次/保留4GiB，首故障停止，不并行、不自动恢复或重复。只在ZHY计算，原hp3090仅代码/Git/标准库传输。

## 命令与产物

```bash
cd /opt/data/private/hechengzhu/ZHY
data/environments/a3-sonic/bin/python -m script.a3.smpl_frozen_validated register
data/environments/a3-sonic/bin/python -m script.a3.smpl_frozen_validated run
data/environments/a3-sonic/bin/python -m script.a3.smpl_frozen_validated verify
```

登记已执行，恢复不能再次register。实验`data/experiments/a3_zhy_smpl_frozen_20261013_E26/`，日志`logs/a3_zhy_smpl_frozen_20261013_E26/`，清单`data/manifests/a3_zhy_smpl_frozen_20261013_E26.json`。下一为受保护run，尚无策略结果。

## 边界与保存

七段已反复用于开发，不能将旧test分区当作未见验收。当前只建立同参考冻结官方动力学基线，不能声称提升或真机可用。退出/参数更新/checkpoint不证明效果。阶段结果须独立复算覆盖、跌倒、误差、动作、接触与实际mesh最低点，再保存README/索引/current state及小型SHA清单，审计提交推送main核验。原始工件LOCAL_ONLY，未验证网盘备份。长期工程继续，达到验收或机器首故障才暂停heartbeat。
