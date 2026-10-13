# A3保存动作裁剪协议与闭环对照E35

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_ASSEMBLY_AUDIT_20261013_E35；准备完成，待登记/执行。

## 目标与边界

结清E34六份完整回放的验收与官方基线对照，不重复回放/装配，不改权重。E34已记录zzx6.06秒跌倒，其他五段无跌倒；旧精确比较把捕获动作和记录动作当同一字段而失败，旧STOPPED不改写。固定vendor第4064–4066行实际将policy.act转float64后裁剪到[-20,20]再生成target，第3142行保存的是该后处理动作；captured actions来自调用内的原始输出。新语义检查不等于放宽旧跌倒/效果门。

## 固定协议与预算

唯一主变量为已保存数据的裁剪前/后语义及同条件误差分解。五个正负边界对照；clipped(captured)==timeseries动作精确相等，default+clipped*scale与q_des差<=1e-10，初始1570输入与E26精确相等；499帧全状态/足geometry独立重建差<=1e-10、根位置与保存qpos相等、原跌倒门独立核验，最后按E33/E34原能力保留和整体门统计。保留裁剪例子的tick/关节和幅值，不把动作裁剪直接当跌倒原因。

一次900秒，ZHY既有环境单CPU线程，无CUDA/权重/策略回放/optimizer/mj_step/装配复算；CPU90/GPU85摄氏度连续两次5秒、4GiB保留、首异常停止，无自动恢复/重复。代码script/a3/smpl_assembly_saved_audit.py；登记前固定E34所有metrics/trace/inputs/contact/入口回执及旧基线/模型/源码/桥SHA。

```bash
cd /opt/data/private/hechengzhu/ZHY
data/environments/a3-sonic/bin/python -m script.a3.smpl_assembly_saved_audit register
data/environments/a3-sonic/bin/python -m script.a3.smpl_assembly_saved_audit run
```

实验data/experiments/a3_zhy_smpl_assembly_audit_20261013_E35/，日志logs/a3_zhy_smpl_assembly_audit_20261013_E35/，清单data/manifests/a3_zhy_smpl_assembly_audit_20261013_E35.json。旧E34回放、原装配NPZ和全部原始数据继续LOCAL_ONLY。

## 当前验证与记录

新代码语法检查通过，数值检查尚未运行。E34机器健康、23样本同boot/无OOM或新关键内核错误，六段499步执行证据已保存，但完整验收曾失败。准备协议和E34结果先scoped audit/提交/push核验，再启动一次；不把新正确字段核验改写成旧E34精确原始字段门通过。

## 下一步

此项结清后结束被动机构初始化分支：仅延迟跌倒不足以升级。转入有官方行为保持约束的小微调候选，优先控制更新范围/冻结encoder，保留官方原动作和独立未见会话；实际参数选择须按固定网络核查，输入质量/真实训练精度和初始概率合同沿用已验证保护。论文接触/速度保持思想为参考修正候选，不自动扩展编码器或换框架。路线仍是改善官方而非增加训练次数。
