# X2 偶发慢回复容错候选

用户要求先跑通衣服和policy。独立候选保留50Hz请求、500Hz writer、目标80ms和反馈120ms有效期。单次策略回复晚于30ms或人体参考暂不可用时丢弃本次，保留上次新鲜目标；目标过期由独立writer进入阻尼，反馈过期及有效标志但格式错误仍故障，异常与其他保护保持。更改源经原R6 SHA校验生成，不覆盖既有R6/R7。状态：源码准备，板端及真实验证待完成；不可据此运行实控。

## 板端只读结果

独立R1已于SoC1构建，原writer gate测试1/1通过。20秒真人衣服、真实只读反馈observe正常退出，962请求、876接受、86拒绝：人体不可用81（其中48过期、22插值间隔过大、9历史不足、1无帧、1其他）、RPC慢3、反馈过期2；MC捕获完整、transport正常、0电机发布。observe处于STANDBY，不能验证CONTROL状态下目标保持、过期和故障锁存的完整行为；反馈过期2仍触发保护，故尚不能称真机控制链路通过。实际输出 `data/analysis/X2_TOLERANT_OBSERVE_20261115_R1`。源码未覆盖默认，仍须独立的CONTROL状态模拟和输入连续性检查，实控须现场人工执行。

## 隔离控制与现场命令

原隔离测试在机器人默认 DDS 配置下没有收到模拟反馈。独立域 191 及本机 DDS、取消继承厂商 DDS XML 后，可正常收发私有话题。新增两种模拟输入。首次断流夹具漏传 `publish_allowed=false`，被协议检查拒绝，保留 R3 日志。修正夹具后，R4 五种场景全部通过。单次 41.01ms 回复被丢弃，保持 CONTROL 并正常归还，持续人体断流触发 `policy_target_expired`，反馈断流触发原保护。最终五组均 RELEASED，真实电机发布和 MC 服务写入均为零。回执 `data/analysis/X2_TOLERANT_ISOLATED_20261115_R4/summary.json`。终后实读 MC Running(40982)/STAND_DEFAULT，无旧 writer，策略进程在。真人 20 秒只读 995 请求，启动拒绝 10 次，之后连续 985 次接受。首次真人容错版接管仍需现场短测。

等网页显示 SMPL 实时更新，MC 为 STAND_DEFAULT，现场确认支撑和急停后，用终端三在任意目录执行下面命令。终端一和二保持运行，先做小幅动作，让定时会话自动归还，等 `MC_STAND_CONFIRMED`。如进入故障阻尼，保持终端三运行，按本次日志中的身份走恢复流程。

```bash
base=/home/agi/x2_morph_human_20261113
sh "$base/script/x2_morph_backend/run_orin.sh" control \
  --binary "$base/data/analysis/X2_TOLERANT_R1_20261115/build/x2_morph_native" \
  --profile "$base/data/analysis/X2_TOLERANT_R1_20261115/profile_30ms.json" \
  --endpoint tcp://127.0.0.1:5765 \
  --output "$base/data/analysis/X2_TOLERANT_CONTROL_$(date +%Y%m%d_%H%M%S)" \
  --seconds 10 --control --confirm-live X2_ULTRA_MORPH_CONTROL
```
