# A3冻结官方闭环接续E25

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_FROZEN_20261013_E25。11:18:55停止，ENTRY_ERROR_BEFORE_POLICY，0策略回放；旧实验保留，不重跑。

## 目标与固定决定

E24已完整核验六段CSV/root/关节与loop模型身体FK，但评测入口漏传`summarize_only`，在加载策略前停止，0回放。失败记录已提交`91b1624e88efa4473c46530dcd22787990bb2bf6`并与远端main核对一致。E25独立补齐该参数，复用已核验ABI及相同六段输入，不重复E24或接口计算，不改历史源码/登记/账本。

唯一工程修正是显式`summarize_only=False`；模型、参考、帧率/完整长度、零延迟、seed0、采集与判据全部沿用[E24协议](README_A3_SMPL_FROZEN_20261013.md)。各段499策略步，六段合计2994；实际结果须完整覆盖且独立复算，0optimizer。独立核验跌倒、关节/root/body误差、真实动作一致性、50Hz脚地接触和实际脚mesh最低点。

六段均为反复使用的开发参考，不能当作最终未见测试。本次仅建立冻结官方在这些合格参考上的动力学基线；没有候选模型对照，不声称提升。原下蹲质量失败仍排除。

## 版本、命令与预算

新增`script/a3/smpl_frozen_followup.py`复用E24只读核验函数和捕获入口，父失败SHA与ABI也加入注册。先固定新实验版本/输入SHA再启动。

```bash
cd /opt/data/private/hechengzhu/ZHY
data/environments/a3-sonic/bin/python -m script.a3.smpl_frozen_followup register
data/environments/a3-sonic/bin/python -m script.a3.smpl_frozen_followup run
data/environments/a3-sonic/bin/python -m script.a3.smpl_frozen_followup verify
```

ZHY既有环境，0训练/0真机。六段仅一次回放，run/verify各900秒；原5秒采样、CPU90/GPU85连续两次、4GiB保留及首故障停止保持，无并行计算、无自动恢复。原hp3090只代码/Git/标准库传输。

## 产物与验证

实验`data/experiments/a3_zhy_smpl_frozen_20261013_E25/`；日志`logs/a3_zhy_smpl_frozen_20261013_E25/`；小型SHA清单`data/manifests/a3_zhy_smpl_frozen_20261013_E25.json`。ZHY入口编译和register通过，基线HEAD91b1624；旧incoming文件保留Git stash，不覆盖并行工作。

实际11:18:50–11:18:55运行5.31秒。补齐参数后发现第二处入口错误：用仅含Popen的SimpleNamespace代替subprocess，遗漏STDOUT常量。在真正调用Popen前AttributeError停止，因此manifest的running/attempt1是预启动持久记录，不证明已执行策略。指标/捕获工件0，实际0策略回放/0optimizer；六段ABI复用未重算。此错误来自新增包装器，不归因官方策略或数据。

3健康样本同boot，CPU41/GPU26摄氏度，最小容器可用9,629,343,744字节、OOM0、无新关键内核/遥测错误。最大80.28秒间隔为登记到运行空档，不当作连续覆盖；未确认机器故障。

## 保存与恢复

失败及SHA已scoped提交8c2fe75297d86fe0fd32450b854fe8206e99c9af并核对远端main，最终同步见[回执](../data/manifests/a3_zhy_smpl_frozen_20261013_sync.json)；原始工件LOCAL_ONLY。恢复不再run E25。后续独立[E26](README_A3_SMPL_FROZEN_VALIDATED_20261013.md)已完成六段实际回放/独立核验，详情与边界见该报告；原参考和门槛保持，长期工程继续接续。
