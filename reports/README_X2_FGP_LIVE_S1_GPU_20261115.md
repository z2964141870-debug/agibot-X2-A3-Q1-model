# 衣服网页恢复 S1 GPU 推理（2026-11-15）

Done（用户）：网页已从旧 B/U100h CPU 入口切换为 S1 CANO TensorRT mixed_gemm + CUDA Graph。32个录制输出与之前 R7 的姿态、根平移逐位相同，中位31.063ms / p95 36.218ms / 最大37.429ms。上下衣接收恢复，等用户重新标定。

To do（用户）：刷新 http://10.0.1.41:8768，点击 T-pose，3秒准备后保持3秒。通过后检查实际真人输出、耗时及网页连续性。

## Agent done / 恢复记录

用户明确要求当前网页使用之前 S/GPU 后端。检查发现网页进程72852仍加载旧 LatestFgp（B/U100h CPU），因此出现约1秒耗时；不是先前GPU优化失效。原生 Mac→run@10.0.1.41交互登录，核设备SoC1/boot 53eb7038-d4bc-4a86-b891-d93e300f76a4。资源准入可用11588591616B、最高66.187度。

新增 gpu_engine.py，复用原CLOTHO fgp_realtime/onnx_s1、准确mixed_gemm engine SHA 89cf34659df7d686abd4230fcd6b1c17a47b05b9d46dda4ecdbad5381d0f3f08；CANO GPU，LIP/GP/原生carticulate物理CPU，保留精确特征缓存。此前GP GPU无明确收益，本次不采纳。初始化暂时替换工厂后finally恢复；每调用清理探针保留数据，不积累长序列输入。直接读取原生根平移，避免Unity显示偏移加减带来舍入误差。无新权重、训练、反馈订阅或电机发布。作者GP训练来源的LIP哈希不匹配警告保留。

复用61帧夹具（29历史+32输出）在实际网页同类线程中执行，32输出/有限性/原R7姿态和根平移最大差均0。15个网页测试Mac及板端通过，shell语法及Python编译过。不是新30Hz长序列认证，不包含BLE/网格/HTTP/机械响应。

旧CPU网页精确核PID和命令后仅对72852发TERM，确认退出。新自有启动143736，日志 data/analysis/X2_FGP_LIVE_GPU_20261115/server.log，实时输出根 data/analysis/X2_FGP_LIVE_20261115_154806。API已回model_profile=TensorRT CUDA Graph / device GPU，上下衣receiving、无error，reconstruction waiting_calibration。保留重连、137B V2解析及3+3秒标定。

run_orin.sh smpl与run_live_orin.sh默认GPU，缺engine明确失败而不回CPU。CPU诊断仅显式--fgp-backend cpu。桥接骨架改从Morph资产home读取以兼容新的FGP runtime根。第二第三终端未启动或更改。

## Agent to do

用户重新标定后只读采样 /api/state 和 /api/smpl，确认实际递增SMPL、标定稳定性、真人动作和推理/网格耗时。标定失败不要擅自放宽2.5度门；初次旧实测3.254度>2.5度而加速度通过。需要真人新样本才能确认质量。当前仅等待标定，不宣称遥操全链或稳定30Hz通过。仅训练需要六宫图。

来源及代码在Q1分支；本报告发布/设备镜像以实际后续收据为准。原Q1共享训练继续暂停。

## 用户调整标定阈值：5度

用户明确要求姿态抖动门由2.5度放宽至5度，判定和API阈值元数据同步更改。加速度标准差0.8m/s²、至少60样本、3秒准备+3秒采集保持。合成4度通过/6度拒绝，15回归通过；该阈值是工程静态稳定性门，不是T-pose几何质量认证。仅重启本网页服务加载新阈值，用户需刷新并重新标定。S1 GPU链和电机控制未改。
