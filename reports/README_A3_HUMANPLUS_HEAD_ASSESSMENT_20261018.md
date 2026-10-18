# A3 humanplus 已验收输出头训练工件的独立评估

记录日期：2026-10-18。HP15，独立评估合同，非新训练。16:29实施完成，尚未登记/读取保存PT/回放。来源HP14的训练与关闭阶段已COMPLETE且独立审计通过；后续验证在读取PT前因导入路径失败，原HP14 STOPPED保留。

## 目标与决定

用户“继续推进呀。持续优化”恢复主线，本阶段完成已训练最终模型的验证与十项完整匹配效果，不重做训练或放宽门。仅引用HP14已验收训练的固定SHA，不恢复HP14 run、不改变其源码/终态；HP13/HP12及科学STOPPED模型不纳入本合同。

导入故障来自initial_probability在训练器模块，间接导入gear_sonic。新humanplus_saved_probability只保留原compare与initial_probability函数，torch在数值函数内导入；两个函数体的公式、阈值AST一致，不导入训练器/模型，不新增forward。其余成功模型验证、Adam/两key/frozen/precision/token/recipe检查及完整效果协议保持。验证入口、stage审计全部独立绑定HP15。

来源门精确要求HP14唯一完成train阶段、32提交/8更新/3072交互、正常关闭日志与native0、train_audit verified=true；后续唯一故障为固定gear_sonic导入错误、发生保存PT读取前，原结果SHA固定、无已完成验证输出、无健康异常/残留。所有原验收工件重新核bytes/SHA，科学拒绝、部分训练、其他试验、异常机器均拒绝。原训练模型只读使用，新评估不调用旧verify命令或旧审计。

## 实施与版本

起点e01f5aa。新6标准库控制Mac0.133秒/新机0.024秒各通过：来源正/反例、机器异常/部分训练拒绝、纯函数AST相等、原效果/Adam验证相同、无训练/native入口与精确READY换行。0本阶段张量/模型/仿真/optimizer，控制不冒充数值通过；原控制不复跑。单次只读来源SHA核验493项通过。

计算根humanplus@100.101.30.68:/home/humanplus/YUANQI_A3。固定环境与vendor均只读；Mac小代码/记录。登记须已提交干净工作树，冻结全部项目Python、输入大小/SHA、原环境源码、HP11对照合同及HP14 source result/train audit/导入错误收据与全部已验训练工件。

step8来源data/training/a3_humanplus_head_logged_20261018_HP14/R23_001_s0/milestone_8/model_step_000008.pt，226205975字节/SHA c508fe2abf6613d4c9842333b3843d7303523349b9ce65ae5f0f36ff5287626a。035/HP11合同仍与来源固定SHA相同。输出data/experiments/a3_humanplus_head_assessment_20261018_HP15、logs同名、data/manifests同名前缀。

## 固定命令与预算

以下为新登记后顺序入口；未执行项不是通过证据，每阶段单次，上一阶段与独立审计成功才下一阶段：

```bash
/usr/bin/python3 -B -m script.a3.humanplus_head_assessment register
/usr/bin/python3 -B -m script.a3.humanplus_head_assessment run --stage verify-training
/usr/bin/python3 -B -m script.project_tools.humanplus_head_assessment_audit --stage verify-training
# 再cloth、official、effect及各自独立审计
```

新增训练更新/optimizer/训练交互均0；检查0/2/4/8保存状态，只有最终8回放，六衣服2994步+四官方5775步=8769完整步/10动作。HP11六衣服基线均RMSE0.13932748084287783rad，5/6不倒，四官方4/4不倒。效果门六衣服全不倒/均RMSE改善>=1%、原不倒及四官方关节<=+2%、世界根/腕<=+5%、官方无新增跌倒。不因工程故障改科学门；开发通过仍非未见测试。当前035无新改善证据。

每阶段实时三次5秒启动门、>=8GiB/无其他计算/GPU空闲/可靠遥测；运行4GiB保留、CPU90/GPU85连续两次5秒、900秒/阶段。原cgroup v2/全boot journal门保持。首机器异常留证暂停，仅终止自有组，SSH失效UNKNOWN。任何新科学STOPPED不续跑/扩预算/早期效果。

## 保存与下一步

README/索引/current state、代码与小清单scoped审diff和audit_git_payload后本地提交。GitHub已知publickey写失败不重复push，LOCAL_COMMITTED_UNPUSHED；原始日志/模型LOCAL_ONLY，未验网盘。下一单次独立登记→verify-training→audit→完整效果及独立审计，每重要阶段即时保存。前台原zhy-a3暂停避免重叠，完成后恢复原30分钟持续优化，其他自动任务不变。

