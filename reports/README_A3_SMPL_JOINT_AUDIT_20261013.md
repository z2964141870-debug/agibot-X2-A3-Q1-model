# A3保存关节与执行映射审计E28

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_JOINT_AUDIT_20261013_E28，12:47:52完成，VERIFIED_SAVED_JOINT_AND_MAPPING_AUDIT。

## 目标与范围

接续E27，逐关节定位zzx在4.4–4.9秒的参考、目标和实际反馈偏离，复用E26保存的2994完整步。只审计保存状态，不重跑策略，不修改权重、参考、限位或控制器。根输入可辨识性结论保留，不能用它直接解释跌倒。

## 固定协议与决定

单一主变量为对既有状态进行命名关节分解和独立反馈重建，无因果干预。固定模型/URDF/vendor/已编译solver及输入SHA、六段各499帧，重点窗口标签4.4–4.9秒；动态状态时间=(tick+1)/50，actor输入历史应对应上一个步进后的状态。与未倒五段同区间比较，但不当未见测试。

核验：从保存qpos/qvel经既有并联机构solver重建29关节反馈，误差门1e-8rad；与实际捕获actor最后一个历史槽比较，位置4e-6rad、速度2e-4rad/s；由实际action/default/scale重建q_des，门1e-10rad。五个分项历史布局用不同哨兵值正对照，不把拼接布局错误当控制异常。

描述性阈值预登记：关节误差>0.35rad连续5帧、目标超模型位置限位0.05rad连续5帧、并联逻辑/物理关节位置差0.02rad。记录首次区间与全部限位超量，阈值只用于定位，不是新增通过门。严格跌倒前不含首次跌倒帧，参考需仍在原模型限位内。位置目标越限不等于实际子步力矩饱和；本轮不重建或伪称实际控制力矩。

MuJoCo中六个并联机构actuator名称沿用逻辑关节，但实际驱动motor关节；例如waist_roll驱动left_waist_motor，waist_pitch驱动right_waist_motor。源码中的waist输出顺序需要按实际传动读，不能仅凭名称反转就判为错误。会保存29项完整命名映射，并比较模型物理被动关节与solver反馈；内部一致不等于证明Isaac/真机契约完全一致。

## 预算、代码与命令

只在ZHY既有环境单CPU线程；一次900秒、5秒遥测、CPU90/GPU85度连续两次、保留4GiB、首异常停止。不加载策略权重、无CUDA/optimizer/新回放/mj_step，不重编solver，不自动复跑/开机恢复。

```bash
cd /opt/data/private/hechengzhu/ZHY
data/environments/a3-sonic/bin/python -m script.a3.smpl_joint_audit register
data/environments/a3-sonic/bin/python -m script.a3.smpl_joint_audit run
```

代码script/a3/smpl_joint_audit.py；实验data/experiments/a3_zhy_smpl_joint_audit_20261013_E28/，日志logs/a3_zhy_smpl_joint_audit_20261013_E28/，小清单data/manifests/a3_zhy_smpl_joint_audit_20261013_E28.json。执行后禁止重复register/run。当前基线HEAD a0839562ed7d21fff483df7298530873ccf1b735，新增精确代码SHA将在登记中固定。

## 实际验证与结果

登记12:47:31，执行12:47:42–12:47:52，10.51秒。2994保存步全部核验，五个历史布局正对照通过。独立反馈重建最大误差0、实际actor历史位置误差最大1.1324e-7rad、速度9.0669e-7rad/s、历史action差0、q_des重建差0；六段参考均未超模型关节限位。支持当前保存证据内的关节顺序、历史时间对齐和动作缩放一致，不能扩大为已验证真机/Isaac全部接口。

重点窗口4.4–4.9秒分组误差如下，单位rad，均为相同26帧的RMSE。动作各异，因此非跌倒五段只作描述性参照，不是配对因果对照：

| 动作 | 腿 | 腰 | 臂 | E26跌倒 |
| --- | ---: | ---: | ---: | --- |
| zj_walk | 0.10484 | 0.03140 | 0.07387 | 无 |
| zj_warmUp | 0.18379 | 0.10282 | 0.22080 | 无 |
| fjw_walk | 0.09796 | 0.01640 | 0.09210 | 无 |
| zls_walk | 0.08878 | 0.02627 | 0.06296 | 无 |
| yy_walk | 0.12379 | 0.01367 | 0.05659 | 无 |
| zzx_walk | 0.25896 | 0.09518 | 0.16412 | 5.66秒 |

zzx窗口最差左膝RMSE0.56774rad、右膝0.41734rad，其次右肩yaw0.28668、左髋pitch0.28048。左膝首次连续5帧误差>0.35rad从标签4.66秒开始，4.74秒确认；左髋pitch从4.70秒开始。右膝较早1.66秒已有同阈值事件，不能称左膝是全轨迹的首个异常。时间标签对应已步进状态，实际状态均加0.02秒。4.70秒左膝参考0.4619、目标1.2547、反馈1.2536rad；此刻实际膝角接近策略目标，却远离参考。4.80秒参考0.5581、目标0.3735、反馈1.3014rad，目标与反馈又不一致。必须同时看策略动作、执行响应和支撑事件，不能一律归因PD不跟踪或策略目标错误；策略本来也会作平衡补偿/预瞄。

全部六段均有持续目标超位置限位事件，未倒动作也有；zzx双膝跌倒前目标均在限位内，故“膝目标越限导致这次跌倒”不符合这些证据。目标越限不是实际力矩饱和，本轮未记录子步力矩，不据此修改限位或夹紧目标。29项policy→motor完整映射已保存：脚pitch/roll实际驱动up/down电机，waist_roll/pitch驱动left/right电机，与官方传动命名一致，不能仅凭waist名称交换判错。

zzx的solver逻辑角与MuJoCo被动关节角连续5帧差>0.02rad：左踝roll从5.02秒、腰pitch5.24秒、右踝roll5.26秒开始；其余五段没有同样持续事件，但有瞬时差。zzx各并联关节最大角差0.0347–0.1491rad、速度差最高20.78rad/s；未倒zls的右踝pitch也曾有18.58rad/s瞬时速度差。solver与物理关节本就来自不同状态路径，差异既可能涉及约束偏离，也可能涉及速度转换，尚未证明错误或因果。下一先对保存状态做独立有限差分，验证solver FK与dfk的一致性；不把晚于膝误差的持续角差当作已锁定根因。

4健康样本同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，CPU最高41/GPU26摄氏度，容器可用最低8,427,171,840字节、OOM0、无新关键内核/遥测异常。10.97秒最大间隔为登记到运行，计算阶段5秒保护，不宣称全窗口连续覆盖。0策略权重加载/0新回放/0optimizer/0mj_step，无CUDA和solver重编，作业正常退出。代码SHA5461f9e2f2b2c40ee422f6690f663cd1e68967284b9c35cf7370252526ae9504，完整输入/bridge/SHA见清单。

## 保存、同步与恢复

原始证据LOCAL_ONLY，Mac只留小代码/README/清单。本README、索引/current state和结果清单已写，5文件scoped检查/audit PASS，364050字节；结果提交3d459a7bc7cc37ee652fa129ca3e70b54cc0aaf7已推送GitHub main，源机/ZHY快进并核验同HEAD、工作区干净，12:56:22仍同boot/GPU空闲。incoming三文件scoped stash保留，8代码SHA与Mac镜像一致，见[发布回执](../data/manifests/a3_zhy_smpl_joint_audit_20261013_sync.json)。本发布回执与后续记录随后另作scoped提交，最新HEAD以Git历史为准。

下一E29只对保存状态的并联FK/dfk做独立数值核验；若需要精确实际力矩/动态支撑则另登记子步采样或动力学检查。官方仍为基线，没有模型提升或未见验证，工程heartbeat保持ACTIVE。
