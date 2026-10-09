# A3 降低 CPU 额度后的受控续训

记录日期：2026-10-09，北京时间。实验：R01 第三次启动。状态：15:12启动，15:14已核验1800→1825，服务运行中；不宣称长期稳定。

## 目标与授权

用户明确要求“继续训练”，并询问 GPU 训练为什么 CPU 高温。本轮从有效 step1800 恢复到既定目标2000；保留 R01 原有预算、有效期和温度保护。仅修改训练服务的 CPUQuota，300% 降到100%（累计约一个逻辑核的 CPU 时间额度，不是固定绑定一个核）。Q1 工作副本不修改。

## 已核实的证据与解释

13:47:27、13:47:32，`x86_pkg_temp` 连续90°C，GPU43/42°C；控制器因 temperature_stop 发出SIGTERM，returncode=-15。15:08 空闲时 sysfs CPU封装40°C，原服务CPUQuota=3s/秒；无高负载后台任务证据。温度读取路径为 `/sys/class/thermal/thermal_zone*/{type,temp}`，没有混淆GPU读数。

GPU执行神经网络计算，并不使CPU闲置。本工程Python调度、环境接口、模型序列化/CPU重载、SHA校验与磁盘保存均涉及CPU。具体仿真CPU/GPU分工需按实际配置核实，不能从GPU训练推断全部仿真都在CPU，也不能把本次CPU温度保护当成过去重启的已证实原因。短时睿频或散热情况可能影响温度，但目前未核实散热器/风扇/供电故障。

## 改动与边界

- `script/a3/autoresume.py` 增加 `--rearm-temperature AUTHORIZATION`：仅接受温度保护blocked、剩余预算和未过期任务；原子记录授权历史，保留attempts和job指纹，操作本身不启动训练。拒绝操作不改变ledger。
- `script/a3/cpu_reduced.conf` 为用户服务drop-in，CPUQuota=100%；原CPU90°C/GPU85°C连续两次采样停训保护保持，环境64、线程2、训练算法及学习率保持。
- 启动最多消耗第三次机会，不重置2/3历史预算，不延长10/10 12:47:08有效期，不反复解除blocked。
- 这是资源/温度控制观察，可能变慢；不属于算法效果对照，不足以证明长期稳定。

## 版本与命令

服务器 `/media/yu/FAFF-E9771/YUANQI`，main起点 `7595dfac58e90f5adbcb05b77774e3205dd33b92`；官方vendor `fe6868ba37034f89b912f0fb851bce19f120266d` 保持。

安装drop-in并停止旧服务后执行，授权记录只能用于本次用户已明确要求的续训：

```bash
cd /media/yu/FAFF-E9771/YUANQI
systemctl --user stop yuanqi-a3-autoresume.service
install -D -m 644 script/a3/cpu_reduced.conf ~/.config/systemd/user/yuanqi-a3-autoresume.service.d/cpu_reduced.conf
systemctl --user daemon-reload
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
data/environments/a3-sonic/bin/python -m unittest script.a3.test_autoresume script.a3.test_checkpoint_store -v
data/environments/a3-sonic/bin/python -m script.a3.autoresume \
data/training/a3_20261009/autoresume_control_R01 --rearm-temperature \
'2026-10-09 user explicitly requested continuation; CPUQuota 300% to 100%; retain thermal guard and budget'
systemctl --user restart yuanqi-a3-autoresume.service
```

起点模型 `data/training/a3_20261009/E006_auto_02_s950/model_step_001800.pt`，402081495 bytes，SHA `c02949f78a221c96e20e2576a74d61cdbd51a2ee695bf2539bfe83eeee71c8e5`；新run预计为 `E006_auto_03_s1800`。续训恢复网络/优化器/计数，物理仿真episode重置，不声称逐帧轨迹连续。

## 验证与产物

32项恢复/持久化测试通过，包含重新授权保留预算、拒绝过期/超预算和拒绝操作不写ledger。已安装drop-in，systemd回读CPUQuotaPerSecUSec=1s。15:12:04启动服务，15:12:09登记第三次attempt，原job指纹、有效期和前两次attempt保留。

`resume_loaded.json` 核验loaded_global_step=1800、Actor/Critic与源一致、优化器steps=[3000,36000]、两个学习率2e-5，episode重新初始化。实际launcher声明Using device cuda:0，GPU有计算利用率；并非误把全部训练切到CPU。

15:14独立selector通过step1825的SHA、CPU重载和有限性核验；402081749 bytes，SHA `9c4f0b4d7da5ee5bce308984b40d849b0261862854ce10f5d5cd10596fd06f2b`。15:14:04 CPU59°C、GPU39°C/68%/127.76W，boot不变，服务active/running。此前数次运行采样CPU56–60°C，属于短时观察；尚未到2000，不代表策略效果或硬件根因已验证。

日志 `logs/a3_cpu_resume_20261009/{tests.log,rearm.json,checkpoint_validation.log}`；新run日志 `logs/a3_training_20261009/E006_auto_03_s1800/`，模型在 `data/training/a3_20261009/E006_auto_03_s1800/`。大模型不转移到Mac，仍LOCAL_ONLY。

## 保存、同步与下一步

本记录先落盘，再执行，现已补真实结果并更新索引/current state。发布前检查diff、scoped暂存和payload审计，推送main并比对远端；实际发布结果以Git历史与本地publication.log为准。

现有10分钟巡检任务更新为读取本README及第三次启动状态；不得因空闲温度回落再次擅自解锁。完成2000或到期后保存最终记录并暂停本巡检，保留每日研究简报。
