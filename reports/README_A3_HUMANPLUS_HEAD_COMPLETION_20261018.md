# A3 humanplus 原输出头候选剩余官方效果与总验收

记录日期：2026-10-18。HP17独立剩余效果合同。原HP16 STOPPED与六段已完成回放保留。本阶段新增仅四官方5775步及十项汇总，0训练/0模型验证/0衣服重放。用户已授权持续推进优化。

## 目标与依据

HP14最终8已训练/保存状态独立验证。HP16六衣服6×499完整，但验收器错误比较preclip capture和postclip trace后STOPPED；已保存数组单次审计确认按固定vendor float32→float64→±20裁剪后逐值完全相同，完整tick/观察/动作/接触/状态/关节RMSE/首倒均核验通过。只有yy6及zzx3标量裁剪，无新模型/forward/仿真。旧终态不修改、不恢复旧命令。

六衣服实测均RMSE0.15762531349rad，对HP11恶化13.1329674%，仍5/6不倒，zzx6.22s倒；候选明确不采用。仍补完原预算尚未开始四官方，留下完整遗忘/退化证据。新合同只接收原6保存数据及已审SHA，不重放cloth、不重训/验证、不评早期点。

## 实施与验证

新增humanplus_head_completion及独立阶段审计只有official/effect入口。官方动作执行函数保持，动作捕获核验仅改为与固定源码真实边界相同的float64+clip后严格逐值一致；不改变控制器/物理或使用浮点容差。最终effect从已审cloth与新official各读一次来源，原科学门公式不改。

新3标准库控制Mac/新机各0.002秒通过，覆盖完整十项通过/失败fixture与原effect输出完全相同、仅剩余阶段/最终8入口、明确裁剪表达式。原6保存数据审计不复跑，新0模型/科学执行。

## 版本、命令、预算

起点15eedee，正式执行/登记以Git/manifest为准。仅humanplus@100.101.30.68:/home/humanplus/YUANQI_A3；环境/vendor保持只读固定。模型来源HP14最终8、SHA c508fe2abf6613d4c9842333b3843d7303523349b9ce65ae5f0f36ff5287626a；HP16已验bundle/config不变。注册冻结新代码、输入大小SHA、原环境源码、HP11合同及单次保存数据审计。

```bash
/usr/bin/python3 -B -m script.a3.humanplus_head_completion register
/usr/bin/python3 -B -m script.a3.humanplus_head_completion run --stage official
/usr/bin/python3 -B -m script.project_tools.humanplus_head_completion_audit --stage official
# 再effect和独立审计；各项单次
```

四官方637/882/1652/2604=5775完整步，与原已审衣服2994组成原十项8769步；新衣服/训练/optimizer/模型验证0。衣服全部不倒/均RMSE改善>=1%、原不倒及四官方关节<=+2%、世界根/腕<=+5%、官方无新增跌倒门保持。目标失败不冒称部分进步，开发通过仍需未见测试。

独立输出data/experiments/a3_humanplus_head_completion_20261018_HP17、logs同名及data/manifests同名前缀。每阶段实时三次5秒、>=8GiB/无其他计算/GPU空闲/可靠遥测；运行4GiB/CPU90/GPU85两次5秒/900秒，cgroup v2与全boot journal门不改，首机器异常留证暂停，仅终止自有组。

## 保存与下一步

README/索引/current state、代码与小清单scoped审diff/audit_git_payload后本地提交，GitHub已知publickey失败不重试，LOCAL_COMMITTED_UNPUSHED。原始日志/模型LOCAL_ONLY。下一剩余官方→审计保存→十项effect→独立审计保存，再用完整失败证据选择实质不同短训练变量，不重复当前head配方/旧方向链或无限零更新诊断。前台原zhy-a3暂停避免重叠，完成后恢复原30分钟持续主线，其他自动任务不变。

