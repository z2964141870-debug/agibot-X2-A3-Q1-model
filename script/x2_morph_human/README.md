# X2 MorphMimic human 本地开发包

主线：SMPL24局部旋转/根平移 → 官方统一人体骨架 → 十帧human策略 → X2 Ultra原生31关节建议目标。

本包支持Mac开发、离线回放、开发场景MuJoCo、独立策略服务及C++客户端。`shadow_reader.py`只订阅机器人反馈并写日志，ROS/AimDK部分需在设备环境验证。**尚未接入现有X2的电机writer，所有输出均为advisory；没有真机控制入口。** 不把本包替换SONIC模型文件，也不运行原训练。

## 本机直接使用

从项目主根执行（环境和模型已经准备好）：

```bash
bash script/x2_morph_human/run.sh doctor --meshes
bash script/x2_morph_human/run.sh replay --frames 150 --output data/analysis/my_x2_policy_replay
bash script/x2_morph_human/run.sh replay --frames 150 --simulate --video --output data/analysis/my_x2_dev_sim
```

输出目录必须是新的，避免覆盖历史。第二条使用合成SMPL、固定默认机器人状态，仅验接口。第三条使用真实ONNX与独立开发场景，保存3秒50fps MP4、逐帧目标/时序、`summary.json`和浏览器可打开的六图`report.html`；不是正式跟踪质量评估。发生跌倒时记录并结束，不偷偷重置。

网络完备的新电脑可执行`bash script/x2_morph_human/setup.sh`。它只建立本项目环境并按SHA下载公开资产，不触碰ROS、厂商服务或SSH配置。`X2_MORPH_HOME`可指定资产目录，`X2_MORPH_PYTHON`可指定运行解释器。Mac虚拟环境不可拷给Linux直接执行。

## 输入约定

`convert`读取安全NPZ（`allow_pickle=False`），字段固定为：

| 字段 | 形状/单位 |
|---|---|
| `local_axis_angle` | `[T,24,3]`，SMPL24局部轴角，弧度，标准SMPL y-up rest轴 |
| `root_translation_m` | `[T,3]`，同坐标系中的真实骨盆根位置，米 |
| `capture_ns` | `[T]`严格递增int64，原始采集时钟 |
| `capture_clock` | 标量字符串，时钟标识 |

不是SMPL-X55直接截前24槽；不是已变换过的世界四元数。衣服已有`human_data`时，必须先核其坐标和rest frame，再转到本消息，不能盲加PICO变换。转换只按固定骨长FK，不需要SMPL body model、逐帧GMR或X2 IK。

复制`config/smpl_yup.example.json`为实际设备标定，填写测量得到的id、坐标旋转、根平移比例/地面偏移。示例id故意不能执行，避免将示例当实测标定。

```bash
bash script/x2_morph_human/run.sh convert --input recording.npz --calibration my_calibration.json --output human.jsonl
bash script/x2_morph_human/run.sh replay --human-jsonl human.jsonl --output data/analysis/recorded_human_replay
```

无`--state-jsonl`时机器人反馈是明确标记的固定合成状态；不能当闭环。保存机器人测量的回放必须同时将接收时间映射至该回放的`replay_monotonic`时钟，不可直接混用另一台设备的单调时钟。

HumanFrame v1包含`kind=morph_human_frame/version=1/sequence`、`capture_ns/capture_clock`、`arrival_ns/arrival_clock`、`calibration_id/skeleton_id`、`convention=world_zup_m_smpl_rest_wxyz`、`valid[24]`、`positions[24,3]`、`quaternions[24,4]`。进入ONNX的168维按**每关节xyz+wxyz交错**，不是先72位置再96旋转。网络服务用自身单调时钟重记arrival，不信发送端声称的接收时间。旧72位置+单四元数消息被拒绝。

接收器保留十个真实50Hz槽，跨度180ms；插值记录右端原始支持，未到齐不输出目标。序号、采集时间、标定/时钟跨epoch或未知骨架拒绝；重连须明确reset。3秒人流过期沿用所锁定的上游teleop值。额外的缺槽/无支持仅使advisory不可用，不触发机器人模式/阻尼。预览跨度、接收年龄、策略计算和硬件端到端延迟分别记录；未知硬件延迟为null。本版没有启用预测尾部/短预览。

## 策略服务与后端接口

```bash
bash script/x2_morph_human/run.sh serve
```

服务只绑定`127.0.0.1`，默认5765，ZeroMQ REQ/REP JSON，每次一个消息。先发HumanFrame，再发测量；`reset`清空人体流、对齐朝向、历史与last_action。缺人流返回`ready=false`且**不含可执行目标**。

`x2_measurement` v1字段：`sequence`、模型覆盖的`joint_names[31]`、`q_rad[31]/dq_rad_s[31]`、`imu_quat_wxyz[4]/imu_gyro_rad_s[3]`、`imu_body`及`body_from_sensor_wxyz[4]`。显式声明IMU安装体和外参；服务可用导出FK将torso姿态/角速度一起重建为pelvis，扣除腰部关节速度。反馈必须按名完整匹配，模型目标顺序也由发布JSON驱动。接收时间不冒充传感器采样时间。

返回`kind=x2_morph_advisory_target`，包含模型SHA/机器人profile、对应state_sequence、原生名字和31个弧度目标、canonical_action29及分段时序。`publish_allowed=false`始终为false。动作只按发布JSON进行一次default+scale/sign/perm映射，头保持默认，last_action仍为canonical policy action。

Python调用示例：

```python
from script.x2_morph_human.client import Client
c = Client()
c.request(human_packet)             # 衣服适配器在原GMR调用前分流
reply = c.request(x2_measurement)    # 现有后端反馈/只读影子回调
# 先记录reply，不发布HAL命令。
c.close()
```

`client.py --input frames.jsonl --output replies.jsonl --realtime`可按采集节奏发送保存人体流。ROS只读工具运行于机器人ROS解释器、先配置`robot_shadow.example.json`实际IMU外参和话题，再调用其`--help`所列参数；它只订阅四组`JointStateArray`及`Imu`，以最多50Hz请求策略，保存每组实际接收年龄，不停止MC也不创建publisher。本Mac未安装ROS/AimDK，不能冒称该节点已通过设备ABI验证。

`cpp/morph_advisory_client.hpp`提供C++ REQ/REP、模型身份、序号、有限性与31槽校验；探针不链接SDK。编译：

```bash
data/environments/x2-morph-human-20261112/bin/python -m script.x2_morph_human.build_probe --output data/runtime/x2_morph_human/bin/advisory_probe
```

Orin需原环境的libzmq开发依赖和C++17编译器。本Mac已实际编译并用真实ONNX进程测试C++客户端；这不等于AimDK整包已编译。

## 搬到Orin及现场剩余工作

```bash
bash script/x2_morph_human/run.sh bundle --output data/deliveries/X2_MORPH_HUMAN_PORTABLE.tgz
```

压缩包含代码、固定来源/许可证、真实X2权重/参数和原始MJCF/mesh；没有Mac虚拟环境、机器人凭据、Linux轮子或HAL写入器。在Orin SoC1解压，准备Python3.11与对应Linux依赖后运行doctor、测试与离线回放。若现场无法上网，需要事先按**实际JetPack/Ubuntu/Python ABI**补齐wheelhouse；当前不知道该ABI，不能把Mac wheel冒充Orin安装包。

CPU ONNX先作为可复现基线。本Mac推理足够轻；CUDA/TensorRT需根据Orin的JetPack、算子支持、动作等价与完整周期p95/p99再启用，不移植训练WM在线优化。SoC0无需部署本包。

现场依次补齐：衣服坐标/骨长/根高度标定和真实流；X2固件反馈话题/关节顺序/IMU外参；Orin依赖/性能；只读影子日志；将C++策略适配接进**现有唯一writer和状态机**，核Morph独立PD/action profile。当前提供的client/reader不负责MC交接，也不另建第二writer。真机控制仍是后续明确任务，需遵循x2分支现场卡。不能许诺带过去不作核对就直接遥操。

## 固定来源与证据边界

- MorphMimic源码：`lzyang2000/morphmimic@770394b4a0cad9effbf807a6c03435d1699733d0`，Apache2源码和X2资产各自许可证随包。
- [官方v0.1](https://github.com/lzyang2000/morphmimic/releases/tag/v0.1)，统一发布的X2 human权重，作者说明X2仅仿真；本包不使用Q1/G1自训权重冒充X2。
- `assets_manifest.json`锁定每个文件大小/SHA；bootstrap不会静默覆盖不同原件。现有政策、特征、normalizer与X2转换使用原官方实现。
- 现有X2部署参考固定`79c4a1e7c019fe57130a2f530fcec78875eb7e4f`，当前包不修改其现场工作树。
- 本地开发场景由公开原始MJCF+发布PD/timing构建，有明确命名；其solver/contact等未过官方`check_same_model`导出门，不能据其三秒存活判断真机可用、自然步态或训练效果。
- 实际阶段结果、失败记录和工件见项目`reports/README_X2_MORPH_LOCAL_SETUP_20261112.md`。

## 最新版 CLOTHO FGP 无头推理与 50Hz 异步预测试入口（2026-11-13）

针对最新两阶段 FGP（`CLOTHO-mocap-main` + `checkpoints`：`U100hL7.07h_residual.pth` 11-IMU 去噪 + `globalpose/weights.pt` 或 `LIP` 全身姿态与真实 3D 根位移），提供零 Unity 依赖的无头推理导出与实时 30Hz→50Hz ZeroMQ 压测入口：

```bash
# 1) 从原始 11-IMU 穿戴数据跑最新两阶段 CLOTHO FGP（推荐 --backend globalpose），导出标准 NPZ 与 HumanFrame JSONL
bash script/x2_morph_human/run.sh clotho-export \
  --clotho-root /path/to/CLOTHO-mocap-main \
  --checkpoints-root /path/to/checkpoints \
  --recording /path/to/imu_recording.pt \
  --backend globalpose \
  --calibration script/x2_morph_human/config/clotho_fgp_yup.json \
  --output-npz data/analysis/my_clotho_gp.npz \
  --output-jsonl data/analysis/my_clotho_gp.jsonl

# 2) 使用导出的最新 FGP HumanFrame 流驱动 X2 Ultra 50Hz 开发级闭环 MuJoCo 仿真并保存 50fps MP4
bash script/x2_morph_human/run.sh replay \
  --human-jsonl data/analysis/my_clotho_gp.jsonl \
  --clock-50hz --max-interp-gap-ms 60 --max-freshness-ms 100 \
  --human-points full --simulate --video \
  --output data/analysis/my_x2_clotho_sim

# 3) 运行全部 18 项单元测试（含最新 CLOTHO 标定等价、Torso->Pelvis 转换、异步丢包 Hold 恢复）
bash script/x2_morph_human/run.sh test
```

关键参数说明：
- `--human-points {full,five}`：默认 `full` 使用全部 24 个 SMPL 关节；`five` 仅保留骨盆、双腕、双踝 5 个关键点（其余关节置零并标记 `valid=false`），用于稀疏降级对照。
- `--max-interp-gap-ms` 与 `--max-freshness-ms`：分别控制帧间线性/球面插值的最大允许原始包间隔（如 `60~100 ms`）与实时流断包后的最大新鲜度保持窗口；超限时返回 `ready=false`（`NotReady`），绝不向外盲目外推姿态。
