# A3候选与官方保存轨迹分解E43

记录日期：2026-10-13，北京时间。状态：代码与固定方案准备，尚未登记或运行。

## 目标与单变量

E42/R17训练合同通过，六衣服/四官方完整开发效果拒绝，衣服全程均误差+9.63%且zzx仍倒，三官方动作超过保留预算。下一先分析保存结果，避免将跌倒后的大误差当跌倒前退化，或仅凭两更新决定增加训练。

E43唯一主变量是已保存同条件轨迹的共同跌倒前窗口与关节分组分解。固定E42与E26/E02各十轨迹、输入捕获、指标及E28已验29关节映射SHA；不加载模型、不forward、不生成新仿真步、不做optimizer或CUDA。此实验是描述性比较，不是跌倒原因干预或下个方案收益证明。

## 方法、预算与判据

逐动作核对499或官方各自固定步数、连续tick、有限值，重建跌倒/全程RMSE、捕获preclip到记录postclip语义；双方首1570输入逐值相同、五类参考数组逐值相同，否则拒绝分析。共同跌倒前区间为[0,min(两方首次跌倒tick,全程长度))，双方共享同一窗口，跌倒tick不计入。腿/腰/臂和各关节分别计算RMSE及对总MSE变化的贡献，贡献须求和一致；固定zzx4.4–4.9秒局部窗口和膝0.35rad连续5帧起点仅供描述，不改跌倒判据或当充分条件。

两个最小控制覆盖连续事件起点及共同跌倒前截断，真实输入/初态/参考合同与贡献求和检查同跑。仅ZHY既有环境/单线程numpy，标准库controller，三次5秒被动预检/至少8GiB余量；CPU90/GPU85两次5秒保护、4GiB保留、900秒预算、首异常停止、无重启恢复/重复。原hp3090只Git/传输，Mac小型代码/记录/清单。

## 版本、入口与工件

代码script/a3/cloth_effect_audit.py，复用既有轻量只读资源监督；登记冻结精确代码/输入SHA，不重置E42和旧失败账本。

```bash
cd /opt/data/private/hechengzhu/ZHY
/usr/bin/python3 -m script.a3.cloth_effect_audit register
/usr/bin/python3 -m script.a3.cloth_effect_audit run --stage verify-training
```

data/experiments/a3_zhy_cloth_effect_audit_20261013_E43/及logs/a3_zhy_cloth_effect_audit_20261013_E43/独立；输出training_verification.json只是复用状态机文件名，语义为保存轨迹分解、0训练。小清单data/manifests/a3_zhy_cloth_effect_audit_20261013_E43.json，raw LOCAL_ONLY，未核验网盘。

## 结果与下一步

尚无实际结果。实现/固定方案经diff、audit及scoped提交推送核验后同步ZHY，顺序登记/一次运行；根据共同跌倒前与官方局部退化决定更新强度/保留约束或参考层单变量，不先追加训练、不放宽原效果门。官方仍基线，十动作都是开发数据，最终未见验证未开始。
