# Q1推理精度独立诊断

04:38:32唯一quick33已实际启动，controller156963/start_ticks13355775、worker156971/start_ticks13355927、原boot/GPU；冻结e83324f8084cd8051d2db357c2b59793237cc7c8独立远端及8文件227821字节三端SHA一致。04:39:28 bf16_static050 repeat1/frame250、三状态/命令/mtime一致，0optimizer、GPU787MiB/36度/共享17067728896字节/OOM0；尚无精度效果结论。只监督原四档三重复，不重复启动/热改来源。[启动](../data/manifests/supermimic_q1_precision_probe_launch_20261029.json)。

2026-10-29 04:19北京时间。3项CPU反例、3源AST解析及服务器真实Python经activate.py的CLI help通过，尚未物理启动。当前只在渲染已存ACTIVITY full330三段视频，先等交付子进程退出再核GPU空闲，不能并行GPU诊断。用户夜间自由探索至09:00、07:00停止新增物理探索；不恢复旧训练或失败重试，采纳null。

## 假设与代码核验

冻结runtime的`x2/supermimic/evaluate.py::load_policy`根据checkpoint args.amp设置`networks.AMP_DTYPE`，`networks.Policy`目标编码器和动作解码器都调用`run_mlp`；BF16 autocast内部结果虽最终转FP32，内部精度已不同。本步只核到代码行为，尚未证明它影响Q1真实步态或导致此前训练退化，也没有据此认定实现错误。未取得可读的新外部资料，不凭搜索结果补造出处。

## 配对实验

同一RF14两小时教师FP32权重/归一化，固定原AP02 30验证+两普通走/抬手33环境，3配置seed32000/32001/32002，新环境、0optimizer。四候选按顺序bf16_static050/fp32_static050/bf16_activity050075/fp32_activity050075；每个FP32与对应BF16同alpha规则比较，另保留bf16_static050共同前缀基线。两种模式均关闭TF32，原PD/effort/动作尺度/腿腰/输入/4096前向块固定。checkpoint只复制内存args.amp，源文件和张量不改；模型摘要/归一化和实际全局精度前后检查。

首个真实4096 padded teacher call用一次forward hook在完全相同history/goal和权重上计算另一精度；先卸hook，计算后恢复当前精度，再原样返回已选输出，不额外reset或仿真。保存动作差和按实际关节尺度换算的目标rad差，不能拿此差当轨迹误差或物理改善。记录每条片终止前状态/接触/足高/摆脚/滑移/真实前进、上肢/腕/峰高/时序和漂移。原逐片继续门只诊断；缺证不提升到训练或用户采纳。

原依赖1188加3新源码共1191项，正式运行前须独立冻结/发布/三端SHA；独立根`data/supermimic_for_q1/Q1_INFERENCE_PRECISION_PROBE_20261029`。2400秒壁钟/900秒实际无进展、100GiB余盘/12GiB实验/30GiB共享cgroup/OOM0/86度、07:00截止，失败不自动重试。权重/大NPZ服务器留存，已有小JSON/CSV/PNG/PDF/MP4镜像Mac大小SHA；09:00交付原始数据和固定视频，不承诺此诊断必有改善。[准备收据](../data/manifests/supermimic_q1_precision_probe_preparation_20261029.json)。
