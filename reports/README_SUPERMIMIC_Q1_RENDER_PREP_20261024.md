# Q1新接触协议的已审计轨迹渲染准备

记录日期：2026-10-24，北京时间。阶段：SM-Q1-V02-PREP。状态：独立入口实现、本地与服务器各4项控制、R02B三片真实只读预检完成；实际EGL执行及完整视频渲染尚未验证。

## 目标与范围

为R03固定预算结束后的完整动作目视验证准备入口。原`render_trace.py`仅回放保存的物理姿态，不推进动力学，但直接构造R01旧资产，其渲染收据会登记旧接触配置。新入口明确绑定已审计的动力学轨迹及实际新接触资产；不修改R03冻结的`script/supermimic_for_q1/`或`script/supermimic_q1_extension/`。

## 完成内容与决定

新增`script/supermimic_q1_analysis/render_audited_trace.py`，保持与现有分析工具相同的文件身份/独立审计模式。先核对评测、对应独立审计、checkpoint、admission、选定轨迹/参考、MJCF和冻结渲染源码；拒绝旧接触协议。真实渲染时复用原渲染器，仅把该进程内的资产构造器绑定到`ContactCorrectedAssets`，并核对实际资产manifest与admission一致，返回后恢复构造器和argv。

完整输出自动覆盖全部实际执行步数，50Hz保存轨迹按隔帧渲染到25fps；奇数步向上取整。提前失败时保留完整参考分母和失败标记，不将较短视频称为完成整段。输出拒绝覆盖。`--check-only`只用标准库读文件及计算SHA，不导入MuJoCo/PyTorch、不创建视频目录、不进行渲染或推理。

渲染后再核对原生收据中的资产、评测/轨迹/参考身份、视频帧数和所有输出SHA，保存`audited_render.json`。真实视频的ffprobe、首/中/末画面和浏览器播放仍需后续实际核验，不能以预检通过替代。

## 版本、命令与证据

本地副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`。服务器`huanyu:/opt/data/private/YuanQi_Q1`，Python沿用`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。代码版本由本次Git提交和文件SHA提供，未改变科学发布`d5f9476`的61项冻结文件。

本地命令`python3 -m unittest discover -s script/supermimic_q1_analysis -p test_render_audited_trace.py -v`实际原生退出0、4项通过：完整/失败前缀与奇数帧计数，错误审计/被改轨迹拒绝，旧协议/渲染源码变化拒绝，check-only不调用渲染且拒绝覆盖。测试使用合成小文件，只验证元数据入口，未加载真实模型或执行EGL。

发布/部署后，以已完成R02B保存工件做真实只读预检，每片使用独立输出文件，不重跑旧评测：

```bash
$PY script/supermimic_q1_analysis/render_audited_trace.py \
  --root "$PWD" \
  --evaluation data/supermimic_for_q1/R02_cachefixed_pair_20261023/R02B_batch128_eval_final/receipt.json \
  --audit data/supermimic_for_q1/R02_cachefixed_pair_20261023/R02B_batch128_eval_final_audit.json \
  --clip 001_walk_front_slow \
  --output data/supermimic_for_q1/V02_render_prep_20261024/render_walk \
  --check-only \
  --preflight-output data/supermimic_for_q1/V02_render_prep_20261024/preflight_walk.json
```

R03完整流程结束后，将evaluation/audit换为其`eval_final/receipt.json`与`eval_final_audit.json`，三个片段各指定新输出目录，移除check-only与preflight-output，再核验真实渲染及画面。R03运行期间本阶段只做元数据预检，不启动额外GPU渲染/模型推理。

## 结果与边界

00:07:26原R03实际1550/12000轮、24800新交互、569组WM/519次actor，累计4550轮；原controller762428/start3606081278与worker762431/start3606081380在同boot下存活，冻结条件通过，GPU40摄氏度/OOM14。此快照不证明当前实时状态或最终跟随通过，恢复仍核对原进程及日志。

代码发布`9af07bbe47b5a50996a5de0fbe83e3055f06ab9f`已独立核对远端Q1。服务器部署逐文件大小/SHA通过：入口7538字节/SHA`c6fb425367704a3ea2636f2d69d8a374e908d02d5c1ccdce1a0c36c3303cd3b3`，测试5158字节/SHA`3f664e5a9a2d94598285b77dac865207999a828e0ac29bd0e9040cb4151a4e8c`。服务器同一unittest命令原生退出0，4项全部通过、0跳过。

R02B真实评测与独立审计身份通过，走路/浅蹲/出拳三次check-only均原生退出0。实际完整轨迹1652/1157/678控制步，对应预期826/579/339帧；均未创建视频目录。新增物理步、策略推理、optimizer更新和渲染帧全部为0。这是已保存真实输入的元数据预检，不是EGL或视频通过证据。

服务器小证据在`data/supermimic_for_q1/V02_render_prep_20261024/`，汇总收据4816字节/SHA`17431f5be9ba5523f2cd1d349c9eaee2984beaf0208c135d134871b1cb1aded2`。Git小副本为`data/manifests/supermimic_q1_V02_render_preparation_20261024.json`及三份`supermimic_q1_V02_preflight_{walk,squat,punch}_20261024.json`，日志为`logs/session_records/supermimic_q1_V02_{controls,preflight_walk,preflight_squat,preflight_punch}_20261024.log`。五个SCP会话均收集原生退出0，汇总收据、本地源码和七份证据/日志逐项大小/SHA一致。

00:21:23汇总收据绑定R03原进程及冻结条件，1975/12000轮、31600新交互、739组WM/689次actor，累计4975轮，GPU39摄氏度/OOM14。00:28:01再次只读核验原controller762428/start3606081278、worker762431/start3606081380同boot且非僵尸，源码/输入/运行时通过；2150/12000轮、34400新交互、809组WM/759次actor，累计5150轮，GPU39摄氏度、空闲16671MiB、容器31050973184字节、OOM14，尚无终态收据。没有加载checkpoint张量或更改冻结代码/预算。

模型、轨迹和后续视频仅留服务器LOCAL_ONLY；本阶段只同步代码、报告和小型证据。实际EGL、ffprobe、画面及播放仍未验证。先检查diff/暂存区及payload audit，显式推送Q1并独立核对远端，不把计划记成已执行。

## 下一步

等待R03原固定预算及保存审计、完整配对评测/独立重算结束，再用新入口实际渲染，核验ffprobe、首/中/末画面及播放；结合误差、腿部、根位移与实际画面决定是否得到可用策略。R03训练源码/预算不变，不恢复A3。
