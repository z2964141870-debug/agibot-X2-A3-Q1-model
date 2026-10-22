# 新版CLOTHO离线配对诊断

记录日期：2026-10-22（北京时间）。实验：T11。状态：三段CPU配对诊断及独立指标复算完成，属于发布训练数据上的开发诊断。

## 目标与范围

用户在确认已有目标和指定GitHub仓库后明确“请继续工作”。按已保存计划推进新版FGP离线入口和同场人体诊断，随后定位标签参考自身的A3几何残差。旧STOPPED实验、旧自动任务和035基线保持原状态；本次独立实验不复跑旧训练。

## 已完成准备

- 重新连接huanyu的25893容器。快照GPU为RTX3090、24576MiB，6348MiB已占用、利用率0%、38摄氏度；容器内存上限34359738368字节。CPU诊断显式屏蔽CUDA并限制2线程，不把GPU瞬时利用率当独占许可。
- 当前主机boot为`d994e71f-bae0-46ba-a5f7-6c2163771c6c`。Python环境`/opt/data/private/yuanqi_ops/envs/fgp/bin/python`：Python3.10.12、torch2.8.0+cu128、numpy2.2.6、scipy1.15.3。没有安装或修改共享环境。
- 新包已传到`/opt/data/private/yuanqi_ops/data/fgp/CLOTHO-mocap-main_974bd8d9.zip`，12043763字节，SHA为`974bd8d94c5384cc47b7021d0e1f13092f34948d480a933603c8aeb6db57977b`。安全展开186个文件至`data/references/clotho_974bd8d9/`，逐文件身份保存在`source_manifest.json`。第三方源码仅作为用户提供的参考留服务器，不随Git分发。
- 新实现`script/teleop_research/clotho_offline_diagnostic.py`，只执行原在线文件中的六个指定推理定义，不执行其顶层硬件、输出目录创建或Unity代码。网络/外部进程审计钩子拒绝相关调用。主干模型严格加载、输入/输出有限性、旋转、标签骨架和状态重置检查均实际通过。三段总运行86.304秒，已退出0。

## 固定实验合同

输入为先前T09相同的三段发布训练会话：`183926_zj_walk`、`193310_yy_oneLegStand`、`192333_zj_warmUp`。每段最多300源帧，按原`cano_eval.py`的发布PT方式去掉槽5；剩余10槽中9为根IMU，加速度按30缩放。PT与当前现场原始传感器校准等价性仍未验证，device2bone只登记身份，不再次盲乘标定。

Canonicalizer采用原在线30帧窗口、末帧输出、单Euler步、causal_ema=0.08。源帧0至28只预热；LIP第一次输出对应源帧29，再跳过30次LIP输出，评分为源帧59至299，共241帧/会话。按包内30Hz合同解释时间单位，实际采集时间戳未独立校验。

唯一配对变量为LIP激活`tanh`对`relu`；同一权重、同一Canonicalizer输出、同一状态初始化。姿态位置用标签前30帧校出的共同骨架比较，并独立检验标签pose与joint一致性。该骨架校准是离线标签辅助条件，不能作为实时输入或独立真值。标签根朝向替换仅用于误差归因，绝不进入估计器输入。

两份StageTran权重仍缺，分支关闭；不输出世界根位移精度、不将零平移视为移动跟随。无策略加载、动力学步进、机器人连接或训练。显示层头部清零等后处理未加入模型输出诊断。

## 命令与产物

在huanyu内运行，命令不含认证材料：

```bash
cd /opt/data/private/yuanqi_ops
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  envs/fgp/bin/python -u script/teleop_research/clotho_offline_diagnostic.py run \
  --source data/references/clotho_974bd8d9/CLOTHO-mocap-main \
  --data data/fgp/train_released.zip \
  --canonicalizer script/tools/FGP_Full/FGP-main/checkpoints/canonicalization/U100hL7.07h_residual.pth \
  --lip script/tools/FGP_Full/FGP-main/checkpoints/LIP/LIP_10_real_denoised_7.07_10imu_10.pth \
  --output data/experiments/teleop_20261022_T11_clotho_activation --experiment T11
```

输出目录仅允许首次创建；已有目录不覆盖。输入身份和正在运行时的恢复依据为`admission.json`，只有最终`receipt.json`才表示三段诊断完成。终端日志计划放`logs/teleop_20261022_T11_clotho_activation.log`。NPZ留服务器`data/`，只将审核后的小清单与报告提交Git。所有新工件备份状态为`LOCAL_ONLY`。

独立复核命令：

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 envs/fgp/bin/python \
  script/teleop_research/verify_clotho_diagnostic.py \
  --receipt data/experiments/teleop_20261022_T11_clotho_activation/receipt.json \
  --prior-onnx data/experiments/teleop_20261022_T09_paired_fk.json \
  --output data/experiments/teleop_20261022_T11_clotho_activation/verification.json
```

## 实测结果与边界

每段评分241帧，源帧区间均为`[59,300)`。MPJPE为含根点的24点根相对误差，使用同一标签辅助骨架；下表单位cm。

| 开发动作 | tanh MPJPE | 同权重ReLU MPJPE | 旧ONNX相同源帧MPJPE | tanh手腕 | tanh脚腕 |
| --- | ---: | ---: | ---: | ---: | ---: |
| walk | 2.974 | 12.650 | 6.272 | 4.240 | 5.835 |
| oneLegStand | 2.768 | 8.508 | 10.089 | 3.910 | 4.617 |
| warmUp | 3.096 | 11.817 | 8.273 | 6.342 | 3.761 |

同一Canonicalizer输出上，tanh相对ReLU的MPJPE分别下降76.49%、67.46%、73.80%；局部旋转均误差分别为0.04531、0.05528、0.05523rad。该证据支持旧ReLU入口不适合作为当前LIP基线；不能把已有tanh设置或版本纠正作为新算法贡献。

旧ONNX比较复用了T09已保存输出，只对齐新评分帧重新计算，没有重跑ONNX。两分支架构、输入处理及预热不同，不能将该列差异归因于单个激活设置；只有tanh/ReLU两列是单变量比较。

位置误差改善不等于所有动态指标改善：根相对加速度误差（假定30Hz，m/s²）tanh/ReLU分别为walk 2.970/3.132、oneLegStand 1.826/1.587、warmUp 2.736/2.611。后两段没有改善，不宣称已减少抖动。

两线程CPU的Canonicalizer末帧推理中位数分别116.92、83.25、83.76ms，P95为149.58、96.37、91.85ms；LIP约1.5ms。该运行条件超过30Hz的33.33ms预算，不是CPU实时通过证据。29源帧窗口预热约0.967秒，随后LIP评分另跳过30次输出；实测计算耗时与信号延迟不能混为一谈。

独立验证器重新核对全部输入/NPZ大小与SHA，以齐次变换重新做FK、旋转迹重新做角误差，核对T09相同标签、骨架和数据身份，`verified=true`。三段各自Canonicalizer首输出和两LIP状态重置最大差均不超过1e-6。完整证据见`data/manifests/teleop_clotho_T11_receipt_20261022.json`和`data/manifests/teleop_clotho_T11_verification_20261022.json`。原最终receipt SHA为`bca6796a8a86c7da600129880b64cacaaa621cfb7be084101bb1f532e4ba0e5d`。

这仅覆盖三段开发会话前300帧，发布集有训练曝光，标签/世界轨迹来源及现场校准仍未独立验证；不是新测试、完整动作、机器人改善或论文效果。新增StageTran两权重仍缺。

## 同步与下一步

接下来读取T10已保存轨迹，核查肩/膝目标与A3骨架几何的关系，保留既有5cm及其他参考质量门。未完成几何参考资格验收前，不进入035动力学比较。

本阶段在基于`d0de05cacf2c87d46a92853d1c0734b1275007b9`的本地发布副本实现，未改动服务器既有未跟踪脚本。按指定仓库`main`进行scoped审计、提交、推送和远端核验；实际同步以发布后的远端HEAD比对为准，不将本文先写成推送成功。
