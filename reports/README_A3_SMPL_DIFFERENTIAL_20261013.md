# A3并联机构FK速度合同E29

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_DIFFERENTIAL_20261013_E29，13:03:15完成，VERIFIED_SAVED_DIFFERENTIAL_AUDIT，局部转换合同REJECTED。

## 目标与依据

接续E28保存轨迹：zzx并联逻辑/物理关节最大速度差20.78rad/s，但未倒zls也曾18.58rad/s。独立检查实际缓存solver的FK局部导数与DFK，区分“转换本身不一致”和“物理被动关节/电机路径在动态受力下偏离”；二者均不能仅凭一次差异定位跌倒原因。单一主变量为现有motor q/v的独立局部数值导数核验，不改策略/控制/参考，不复跑E26。

## 固定协议

六段各499保存帧，双踝与腰三套机构，每帧两列单位电机速度从DFK构建解析Jacobian。电机q分别±1e-3/1e-4/1e-5rad做中心有限差分，预定最小尺度为数值导数；三个尺度相互差必须在5e-3+1%绝对数值导数内，且回到相同base的FK变化≤1e-8rad，才列为数值可判定。每次DFK前恢复base FK，避免使用扰动位置留下的内部Jacobian状态。所有尺度/稳定性/错误均保存，不能事后挑尺度或放宽门。

数值可判定帧上，解析与数值Jacobian门同为5e-3+1%；DFK(motor_v)与J_num×motor_v门为0.03rad/s+1%数值速度。核对DFK与保存actor上一状态速度≤2e-4rad/s，保证检查对应实际执行路径。三机构×六段共18个反号Jacobian错误正对照应被门拒绝。合同门只描述局部转换精度，不是机器人行为验收。严格跌倒前不含首次跌倒帧，4.4–4.9秒另列，跌倒后结果只保留原始数组，不混入主结论。存在可判定失配则REJECTED；无失配但有不稳定状态则INCOMPLETE_NUMERICAL_STABILITY；全部可判定且无失配才PASS。

## 版本、预算与命令

固定E28验收文件及继承的E26输入/模型/URDF/static solver/bridge和全部include文件SHA，不重编。仅ZHY既有环境，单CPU线程，无策略权重/前向/optimizer/CUDA/mj_step；一次900秒、5秒遥测、CPU90/GPU85度连续2次、4GiB保留、首异常停止，不复跑/开机恢复。

```bash
cd /opt/data/private/hechengzhu/ZHY
data/environments/a3-sonic/bin/python -m script.a3.smpl_differential_audit register
data/environments/a3-sonic/bin/python -m script.a3.smpl_differential_audit run
```

代码script/a3/smpl_differential_audit.py，复用未改的E28辅助函数。实验data/experiments/a3_zhy_smpl_differential_20261013_E29/，日志logs/a3_zhy_smpl_differential_20261013_E29/，清单data/manifests/a3_zhy_smpl_differential_20261013_E29.json。原始Jacobian数组只存ZHY/LOCAL_ONLY，Mac只保存小型代码/报告/SHA清单。登记后禁止重复上述入口。

## 实际验证与结果

登记13:03:00，运行13:03:04–13:03:15，共10.77秒。六段2994保存帧×3机构均已计算；严格跌倒前2778帧×3=8334机构状态，其中8310可判定、24未达预登记数值稳定门，不能称全部通过。18反号错误正对照均触发检测，DFK与实际捕获actor速度最大误差6.786e-7rad/s，回访相同base FK最大变化0。

8310可判定状态中，2个Jacobian和速度失配都来自zzx右踝，其余8308在原门内；矩阵差最大0.23177、速度差最大1.57359rad/s。两例标签时间5.44/5.62秒，实际状态5.46/5.64秒，严格早于跌倒5.66秒，但晚于关节RMSE首次增大4.68秒和倾斜30度5.40秒。4.4–4.9秒zzx可判定76/78机构状态没有超门，另2个左踝状态数值不稳定；不把缺失判定当通过。其余五段可判定状态未发现转换失配，但有共21个不可判定状态，不能宣称全域正确。

| zzx右踝状态 | 5.44秒 | 5.62秒 |
| --- | ---: | ---: |
| up电机角rad | 1.623975669 | 1.575002513 |
| down电机角rad | -0.888562725 | -0.791872455 |
| DFK pitch速度rad/s | 3.34852 | -2.10652 |
| 数值pitch速度rad/s | 2.85096 | -2.51946 |
| DFK roll速度rad/s | -2.76566 | 5.23316 |
| 数值roll速度rad/s | -4.33926 | 3.79375 |

两例数值Jacobian的up电机整列为0，而DFK仍给非零列。固定MJCF中up/down电机range均[-1.55,1.55]rad；两例up实际角已超过上限0.07398/0.02500rad。提示在越界状态中FK可能存在夹紧/平台而DFK未使用相同导数，尚未独立确认具体实现或阈值。模型位置限位在动力学里出现越界不等于关节交换；本轮没有源码级solver实现，也没有实际子步力矩。需要新的固定电机输入扫描核实有效域/边界行为，禁止把两例晚期转换失配直接认定为这次跌倒的初始原因。

4健康样本同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，CPU最高41/GPU26摄氏度，最低容器可用8,118,632,448字节、OOM0，无新关键内核/遥测异常，最大采样间隔5.196秒。0权重加载/新策略回放/optimizer/mj_step/无CUDA，作业正常退出；合同拒绝是已完成的诊断结果，不是机器故障。原始6份differential.npz合计1,229,973字节，仅ZHY/LOCAL_ONLY。代码SHA5b9e3297188cf8e8f2ac1123e39ab68aa64aac1c386a8f03b1cea26c9bb53605，清单SHA cc1168a6bef2a0622b4d2e4c707bae453a61d39e1192e5add6df713d2d0aa1a7，9注册代码SHA与Mac镜像一致。固定基线提交e5ee9055100ef0075af6129da4e6c5e2ccaaa3ca。

## 保存与恢复

本README/索引/current state/小型清单已保存，结果scoped提交推送与目标同步尚待完成。下一E30只对右踝up电机边界做固定输入扫描，确认FK平台与DFK在域内/域外的合同；不重跑E29，不改控制或放宽门。随后仍须独立定位4.4–4.9秒更早的膝/支撑偏离。官方仍基线，开发七段不能充当未见验收，工程heartbeat ACTIVE。
