# Q1 V01动力学可视化

记录日期：2026-10-23，北京时间。阶段：SM-Q1-V01。状态：完整动力学仿真、独立轨迹重算、全时长视频与浏览器播放核验完成；策略跟随仍不达标。

## 目标与范围

用户要求新跑一轮动力仿真域中的可视化。本轮使用R01最终策略，在huanyu的`/opt/data/private/YuanQi_Q1`中重新进行闭环MuJoCo动力学仿真，完整执行走路、浅蹲和开发出拳三段，保存实际状态/动作，再渲染全部执行时长。左侧为运动学参考，右侧为新动力学仿真的实际Q1；不把参考动画或世界模型预测称为动力学结果。

## 关键决定与版本

- 策略：`data/supermimic_for_q1/R01_20261023/final.pt`，205218451字节，SHA-256 `a3f2e09591b573d49b4029e1c5aecd2680ba49eb98f0699480a2bf4d28c2aa71`。
- 基础代码：`4c66c36998fed03c214c1b4b38a8d3192eeb878b`；保持R01冻结的资产、控制、学习模块和评测代码。仅新增展示页面与本轮记录。
- 物理：自由浮动基座，无骨盆焊接；50Hz控制，每控制步20个1ms MuJoCo子步，PD kp50/kd2，力矩上限为模型额定值25%。这些是原仿真参数，未进行真机验证。
- 接触摩擦保持R01实际条件0.6。原评测收据的`friction1`是脚参数请求值，实际脚地接触以P05实测为准；不在本展示中接入P06修正。
- 本次新增评测预算为1652+1157+678=3487控制步、69740物理子步，实际执行数与预算一致。0 optimizer，不新增训练，不恢复A3或改写旧结果。
- 入场实核：3090 24576MiB，GPU39°C/0%利用率，容器内存27907772416/34359738368字节，历史OOM14；无相关Python作业。权重SHA与三个执行入口SHA匹配本地代码。首次探测cgroup v2路径不存在，已按实际cgroup v1路径读取资源。

## 命令与产物

服务器根：`/opt/data/private/YuanQi_Q1`。Python：`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`，环境名沿用历史依赖，不代表A3任务。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 $PY -u script/supermimic_for_q1/evaluate.py --root "$PWD" --motions data/supermimic_for_q1/motions_P03_20261023 --checkpoint data/supermimic_for_q1/R01_20261023/final.pt --output data/supermimic_for_q1/V01_20261023/evaluation --device cuda:0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' $PY -u script/supermimic_for_q1/audit_evaluation.py --root "$PWD" --evaluation data/supermimic_for_q1/V01_20261023/evaluation --output data/supermimic_for_q1/V01_20261023/audit.json
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 $PY -u script/supermimic_for_q1/render_trace.py --root "$PWD" --evaluation data/supermimic_for_q1/V01_20261023/evaluation/receipt.json --clip <片段名> --max-seconds 100 --output data/supermimic_for_q1/V01_20261023/render_<walk或squat或punch>
```

片段名：`001_walk_front_slow`、`042_squat_shallow_repeated`、`058_punch_front_alternating`。仅在新空输出位置执行，不覆盖已有轮次。

运行日志：服务器`logs/supermimic_q1_V01_{evaluation,audit,render_walk,render_squat,render_punch}_20261023.log`。小型收据保存到Git的`data/manifests/supermimic_q1_V01_*_20261023.json`。

## 实际结果与边界

`evaluate.py`原生退出0，真实动力学循环58.954237秒，完整执行3487个控制步/69740物理子步，覆盖69.74秒仿真时间。三段均无提前终止；归一化回报、关节RMSE及执行步数与原R01最终评测逐项相同，证明这次重跑可复现原结果，不是新的训练改善。代码中的heldout仍是出拳开发集，不是独立最终测试。

| 片段 | 控制步 | 关节RMSE rad | 视频帧数 / 25fps | 播放时长 | MP4字节 |
| --- | --- | --- | --- | --- | --- |
| 走路 | 1652 | 0.3284507 | 826 | 33.04s | 1187946 |
| 浅蹲 | 1157 | 0.3384278 | 579 | 23.16s | 538032 |
| 开发出拳 | 678 | 0.2903787 | 339 | 13.56s | 414471 |

`audit_evaluation.py`与三个完整渲染命令均原生退出0。独立轨迹重算`EVALUATION_TRACE_AUDIT_PASSED`，最大标量差2.363e-8，保存字段最大差2.384e-7；原可用性门仍true/true/false。视频没有只截前12秒，`--max-seconds 100`包含全部实际轨迹；浅蹲奇数控制帧转25fps后视频比物理时长多0.02秒。

三个MP4经独立ffprobe时长/帧数和SHA核对，与各自render收据一致，浏览器实际解码尺寸均1280×548。视频SHA依次为`1e19f41644d98e955bd3fc910d69187ac3268c82cb9c9bb22491eb54b9fa994f`、`6cca60bba8459dbb8123b349500ecbeca8db655277f569db6e4138dc896b4ca3`、`64bad35e6fba14da12c556de78ebcd53de231708bb1d104c49f7314334435a4b`。像素std最小值73.494/73.637/73.629，三段首/中/末共9张PNG均实际目视，无空白、模型完整且未相互遮挡；参考在移动/动作而策略多保持近似站姿，跟随幅度明显不足。

浏览器三个动作切换/实际解码均通过：readyState4，videoWidth1280/videoHeight548，播放时间推进；桌面1280×720已截图检查，390px手机检查documentWidth=390、视频宽374，无横向溢出，之后恢复默认视口。页面已保留为交付标签。原先`ssh -fN`未保持本地监听，现由用户会话`launchctl`托管同一只读SSH转发，已重新确认网页和MP4返回HTTP200并实际播放；这不是后台训练或自动科学任务。

执行期间一次资源采样GPU40°C、容器内存29767880704字节；收尾GPU40°C、内存27924561920字节，OOM仍14，无相关仿真/评测Python作业。采样不等于连续峰值监控；未宣称未采样区间的最高温度或内存。HTTP视频服务与本地SSH转发有意保留供用户观看。

## 保存与下一步

原始状态、动作和视频留服务器，备份状态`LOCAL_ONLY`；只有小型收据、执行日志和9张PNG取回Mac。小型证据：[动力学运行](../data/manifests/supermimic_q1_V01_evaluation_20261023.json)、[独立重算](../data/manifests/supermimic_q1_V01_audit_20261023.json)、`data/manifests/supermimic_q1_V01_render_{walk,squat,punch}_20261023.json`。关键执行日志在`logs/session_records/supermimic_q1_V01_*_20261023.log`。

播放入口：`http://127.0.0.1:18863/`，仅本机访问，通过SSH本地转发连接服务器回环HTTP18863。页面源为`script/supermimic_for_q1/visualization.html`，部署副本为服务器本轮根`index.html`。视频在`render_{walk,squat,punch}/comparison.mp4`。Mac小预览在代码副本`data/supermimic_for_q1/previews_V01_20261023/`。

若播放入口失联，只恢复页面服务/SSH转发，不重复物理仿真。服务器服务命令为`$PY -m http.server 18863 --bind 127.0.0.1 --directory /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/V01_20261023`；Mac转发命令为`ssh -n -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18863:127.0.0.1:18863 -J yu@100.85.145.78 huanyu`。现存Mac托管标签`com.yuanqi.q1-v01-viewer`，可用`launchctl list com.yuanqi.q1-v01-viewer`查看；用户结束观看后可用`launchctl remove com.yuanqi.q1-v01-viewer`关闭此转发。服务器现存HTTP PID741503仅为本次快照，停止前须重新核对命令身份。

本阶段代码与小型记录随Q1分支提交，提交号见Git历史；推送后另核对远端提交。训练改进方向仍为独立接入P06及重新验证后端，R02尚未启动。恢复入口为本README及[Q1主阶段](README_SUPERMIMIC_FOR_Q1_20261023.md)。
