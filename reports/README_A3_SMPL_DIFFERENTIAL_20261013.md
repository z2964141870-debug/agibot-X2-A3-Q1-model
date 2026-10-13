# A3并联机构FK速度合同E29

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_DIFFERENTIAL_20261013_E29，准备，未登记/执行。

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

## 结果与恢复

尚未执行，不能把实现当验证。下一先实际核验，完成/失败立即更新README/索引/current state、小型清单，audit后scoped提交推送main核对远端；保留并行修改。若转换合同通过，继续聚焦参考腿部时序/动态支撑可行性和实际子步控制；若拒绝，先定位转换合同而不自动将其当跌倒根因。官方仍基线，开发七段不能充当未见验收，工程heartbeat ACTIVE。
