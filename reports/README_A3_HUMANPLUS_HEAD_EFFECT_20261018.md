# A3 humanplus 最终输出头模型完整效果

记录日期：2026-10-18。独立HP16，最终效果合同，0新训练/0重复模型验证。16:37实现完成，尚未登记或回放。用户已授权持续推进优化。

## 目标、来源与决定

在HP11匹配条件下评测已训练且已独立验证的最终8。HP14训练8更新/32提交/3072交互与Kit正常关闭独立通过，HP15保存0/2/4/8完整验证通过；两次后续入口故障分别为纯概率导入和评测包缺config，均保留原STOPPED。本合同不恢复旧run、不追加训练、不重复已完成保存模型验证；HP13/HP12及其他科学停止模型不评测。

HP15 cloth在checkpoint_metadata拒绝Missing matching checkpoint config.yaml，尚无explicit_manifest、仿真子进程/策略回放。新来源门精确校验原模型验证成功、原失败仅打包元数据、0旧回放、无健康异常/残留，冻结原结果/成功审计/失败收据/全部已验工件大小SHA。部分训练、未验证模型、已有回放或科学失败不纳入本合同。

## 实施与静态检查

独立humanplus_head_effect及审计仅cloth/official/effect三个阶段，没有训练、native或保存模型验证入口。复用原完整replay和effect函数体，AST逐值相同。新增materialize_bundle将已验step8用硬链接实化到新checkpoint8目录，复制固定原官方config.yaml与原sidecar，fsync后再次核SHA；不使用symlink以免resolve返回原缺配置目录，不改变原模型/目录。

固定vendor sim2sim_a3_mujoco.A3Policy实际按state构建原架构，运行不从该config改变物理；config是既有checkpoint_metadata要求的来源伴随文件。注册直接调用真实checkpoint_metadata验包，只有原最终8允许进入效果。新6标准库控制Mac0.067秒/新机0.028秒各通过：真实metadata/硬链接/无旧目录写入、错误或早期SHA拒绝、来源已验证且0回放、已有回放拒绝、未验模型拒绝、原效果函数相同。0张量/模型/仿真/optimizer。只读来源508项SHA核验通过，原控制不复跑。

## 版本、输入与输出

起点5afbfde；正式执行提交与登记以Git/小清单为准。计算根humanplus@100.101.30.68:/home/humanplus/YUANQI_A3，固定只读环境与vendor fe6868ba37034f89b912f0fb851bce19f120266d。Mac只小代码/记录。

最终8来源HP14/R23，226205975字节/SHA c508fe2abf6613d4c9842333b3843d7303523349b9ce65ae5f0f36ff5287626a。新bundle位于data/experiments/a3_humanplus_head_effect_20261018_HP16/checkpoint8。登记冻结全部代码、输入大小/SHA、原环境源码、来源验证与HP11合同及新bundle配置SHA；不把来源收据当未来启动门。

独立输出data/experiments/a3_humanplus_head_effect_20261018_HP16、logs同名；小清单data/manifests/a3_humanplus_head_effect_20261018_HP16_{plan,result,cloth_audit,official_audit,effect_audit}.json。

## 固定命令与判据

新登记后各项单次、顺序与独立审计门不变；未执行项不冒称已有证据：

```bash
/usr/bin/python3 -B -m script.a3.humanplus_head_effect register
/usr/bin/python3 -B -m script.a3.humanplus_head_effect run --stage cloth
/usr/bin/python3 -B -m script.project_tools.humanplus_head_effect_audit --stage cloth
# 接official/effect及各自独立审计
```

仅最终8六衣服6×499=2994步、四官方637/882/1652/2604=5775步，共8769完整步。新增训练/optimizer/训练交互/模型验证均0。HP11基线均RMSE0.13932748084287783rad、5/6不倒，四官方4/4不倒。门保持六衣服全部不倒/均RMSE改善>=1%、原不倒及四官方关节<=+2%、世界根/腕<=+5%、官方无新增跌倒。开发通过仍不是独立seed/真正未见会话。当前035无新闭环改善证据。

实时三次5秒启动门、>=8GiB/无其他计算/GPU空闲/可靠遥测；运行>=4GiB、CPU90/GPU85连续两次5秒、900秒/阶段。cgroup v2和全boot journal严格门保持；首机器异常留证暂停，仅终止自有组，SSH异常UNKNOWN。科学STOPPED/拒绝不续跑、扩预算、扫LR/比例/权重/batch或早期点。

## 保存、恢复与下一步

scoped README/索引/current state/代码/小清单经diff及audit_git_payload后本地提交。GitHub已知publickey写失败不重复push，LOCAL_COMMITTED_UNPUSHED；模型/原日志LOCAL_ONLY，网盘未核验。下一新登记验包→实时资源门→完整cloth→审计并保存→official→审计并保存→effect→独立审计。前台原zhy-a3暂停避免重叠，阶段完成后按原30分钟持续主线，其他自动任务不变。


## 16:38 单次登记与真实评测包验收

执行代码830f7fa，HP16已REGISTERED，294源码/512输入及原环境来源SHA固定。新bundle通过真实checkpoint_metadata，最终模型226205975字节/SHA c508fe2abf6613d4c9842333b3843d7303523349b9ce65ae5f0f36ff5287626a，config SHA1043717530839e8ef9408e0c8dc470e137708089b64ec13d05b377382868b897。模型仅硬链接，旧目录不改；登记0张量/仿真，0新训练/模型验证。下一原三次5秒启动门后单次cloth，实际效果未知。
