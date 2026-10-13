# A3已验证启动接口的冻结闭环E26

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_FROZEN_20261013_E26。11:29:12独立验收VERIFIED_FROZEN_BASELINE，六段完整/2994策略步；5/6未跌倒、1段5.66秒触发跌倒，整体动力学效果门不通过。

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

登记已执行，恢复不能再次register/run。实验`data/experiments/a3_zhy_smpl_frozen_20261013_E26/`，日志`logs/a3_zhy_smpl_frozen_20261013_E26/`，清单`data/manifests/a3_zhy_smpl_frozen_20261013_E26.json`。每段保存metrics/timeseries/实际inputs/actions/真实qpos/qvel和foot-floor contact采样，清单仅保存大小与SHA，不上传原始工件。

## 实测与独立验收

登记11:23:20，六段回放11:25:53–11:27:31，共98.50秒；独立核验11:28:57–11:29:12，共15.68秒。6×499=2994完整策略步，0optimizer，官方权重与输入SHA不变。独立复算覆盖/跌倒/关节/root/body误差、实际动作与捕获一致、真实qpos的脚mesh最低点，全部核验一致。这是计算证据通过，整体0跌倒效果门失败。

| 开发片段 | 跌倒时间s | 关节RMSE rad | 根位置均误差m | 手腕骨盆相对均误差cm | 腿骨盆相对均误差cm |
| --- | ---: | ---: | ---: | ---: | ---: |
| zj_183926_walk | 无 | 0.09152 | 0.40692 | 1.467 | 2.593 |
| zj_192333_warmUp | 无 | 0.12553 | 0.29794 | 3.541 | 3.035 |
| fjw_140008_walk | 无 | 0.08756 | 0.42488 | 2.085 | 2.335 |
| zls_155013_walk | 无 | 0.09162 | 0.38623 | 1.495 | 2.508 |
| yy_125106_walk | 无 | 0.12544 | 0.92873 | 1.848 | 3.562 |
| zzx_111505_walk | 5.66 | 0.32661 | 1.83793 | 14.610 | 9.568 |

zzx跌倒前283步单列：关节RMSE0.24282rad，根均误差0.61833m，手腕/腿相对误差4.102/4.277cm。不能把全程含跌倒后的动作放大当作先行原因。已保存每段raw action RMS/绝对最大/相邻差RMS及全部真实输入供后续区间排查。

只读保存轨迹分析：tick281/282/283对应5.62/5.64/5.66秒，roll52.82/58.76/62.45度，根高度0.610/0.580/0.545m；首次因roll越60度触发，而非高度先低于0.45m。关节和全局运动的差异可见fjw：参考终点相对起点XY=(+0.376,+0.023)m，实际=(-0.241,+0.027)m。该后验描述仅读已有timeseries，方法为last-first，XY路程为相邻欧氏距离之和；未新增模型、回放或训练。

50Hz接触数量左/右：zj_walk304/313、warmUp347/418、fjw448/481、zls339/328、yy323/335、zzx178/293（每段499帧）。它们是仿真contact存在次数，不能等同稳定承重；清单另保存法向力>1N帧数。未跌倒段动态脚mesh最低点约-7.26至-16.53mm，zzx全程-26.12mm；这些是动力学接触结果，不能套用参考IK的5mm门，亦尚不能据此定位接触参数根因。

25健康样本同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`，CPU41/GPU34摄氏度、GPU功率最高113.66W，容器最小可用7,540,961,280字节、OOM0，无新关键内核/遥测错误。最大152.34秒间隔为登记到启动，回放结束到验收也有空档；两计算阶段内部5秒保护，不宣称整个窗口连续监测。作业已退出、GPU计算进程查询空；未确认机器故障。

## 边界与保存

七段已反复用于开发，不能将旧test分区当作未见验收。本阶段建立同参考冻结官方动力学基线，没有候选策略对照，未证明超过官方/完整遥操/真机可用。五段不跌倒不代表世界轨迹跟随成功；整体动力学门失败，不能据此启动扩大训练。

下一优先：用保存轨迹拆zzx约5.2–5.66秒支撑/关节/控制异常的先后顺序；核对根位置指标与官方接口实际承诺的跟随目标，再做固定输入下根平移可辨识性检查。已知A3-fast无独立人体根平移通道，不能单凭字段缺失或全局误差认定编码器为根因。通过新片段验收仍未完成，旧下蹲质量拒绝保留。

README/索引/current state和小型SHA清单已scoped审计提交492dcb2dcab9ff15cc21c95f6754008ea8a50edb，源机/GitHub main/ZHY HEAD均核验一致且工作区干净；[同步回执](../data/manifests/a3_zhy_smpl_frozen_20261013_sync.json)记录结果提交及清单SHA。bundle发送SSH回执超时，但目标文件与fast-forward独立核验成功，不把超时当作重启。原始工件LOCAL_ONLY，未验证网盘备份。登记HEAD91b1624；启动修正代码阶段8c2fe75。实际compute时target未跟随该提交，仅新增入口文件；注册按精确代码/输入SHA固定，未把旧HEAD当作全部运行源码。旧incoming复制文件保留scoped stash，未覆盖并行工作。长期工程继续，无机器故障/未达最终验收，heartbeat保持ACTIVE。
