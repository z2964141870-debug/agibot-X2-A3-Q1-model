# A3保存动作裁剪协议与闭环对照E35

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_ASSEMBLY_AUDIT_20261013_E35；14:38:26完成，VERIFIED_CAPTURE_SEMANTICS_AND_DYNAMIC_REJECTION。

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

一次执行15.75秒，五个裁剪边界控制通过，2994保存步已核验。六段初始1570维输入与E26精确相等；clip(captured)与记录动作、目标重建及独立足geometry重建差全部为0。无新回放、装配复算、策略权重加载、optimizer或mj_step；旧E34失败断言与STOPPED保留。

| 开发动作 | E26关节RMSE(rad) | E34关节RMSE(rad) | 比值 | E34跌倒 |
| --- | ---: | ---: | ---: | --- |
| zj_walk | 0.09151696 | 0.09017946 | 0.985385 | 无 |
| zj_warmUp | 0.12552501 | 0.12651902 | 1.007919 | 无 |
| fjw_walk | 0.08756301 | 0.08756551 | 1.000029 | 无 |
| zls_walk | 0.09162262 | 0.09121035 | 0.995500 | 无 |
| yy_walk | 0.12543865 | 0.12377955 | 0.986774 | 无 |
| zzx_walk | 0.32661258 | 0.37929278 | 1.161293 | 6.06秒 |

六段等权平均0.1413798054到0.1497577774rad，退化5.9258619%。五段原本不倒动作均在原2%误差保持预算内，但总体仍5/6不倒、均误差增加，开发效果门失败。zzx从5.66到6.06秒跌倒，仅延迟0.4秒，不升级。七段均是开发参考，未见最终测试尚未完成。

裁剪仅见zzx左腕pitch四帧(tick296-299，标签5.92-5.98秒)：原始输出-21.91362/-21.26835/-21.84959/-20.30467，记录均为-20。裁剪解释记录字段差异，不能直接认定为跌倒根因。

5健康样本同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，CPU最高40/GPU最高27摄氏度、可用内存最低9,716,871,168字节、OOM0，无新内核/遥测异常。作业已退出。runner SHA256为3c21e62b66ff160f62d3d2ad640e262690ae2f079ae4c52f8c1b5c3a384a2632；小清单SHA256为042dc2fdd818b2a49a1e09b656ab0871e4008e40261e9d4e79e3dae3818cc0f8。结果data/experiments/a3_zhy_smpl_assembly_audit_20261013_E35/result.json，SHA256为86af50722ee023d714cafb04fe771ddd4c1919ba238dd6439ffa92889ef1eaf7。

README/索引/current state及小清单纳入本阶段scoped审计与发布；提交号与同步证据由Git历史和远端核验提供，不预填未完成同步。原始保存数据/模型/日志LOCAL_ONLY，未验证网盘备份。

## 下一步

此项结清后结束被动机构初始化分支：仅延迟跌倒不足以升级。转入有官方行为保持约束的小微调候选，优先控制更新范围/冻结encoder，保留官方原动作和独立未见会话；实际参数选择须按固定网络核查，输入质量/真实训练精度和初始概率合同沿用已验证保护。论文接触/速度保持思想为参考修正候选，不自动扩展编码器或换框架。路线仍是改善官方而非增加训练次数。
