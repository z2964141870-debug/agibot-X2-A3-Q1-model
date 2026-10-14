# 头环旧录制完整审计与时间/坐标合同

日期：2026-10-14。阶段HB02：COMPLETE；数据用于正式同步根运动效果验收：未通过。没有启动模型、SLAM、IK、仿真、训练或设备操作。

## 目标与范围

接续HB01，回答现有3组配对材料能否支持“衣服基线 vs 加头环根运动”的对照。只在Mac以内存流方式解析原ZIP的JSONL/MP4容器表，不解压视频、不解码图像。原始资料和主线程A3任务不改。

## 完成内容与实测

原ZIP完整SHA256：`cbaf340f86f54c2c2c557ab70cc338aa58e0af522229b040403f883cd6c28c67`。全部3组的身体JSONL、相机索引和MP4读至EOF，ZIP CRC通过；另存成员SHA。MP4仅检查容器/采样表，不声称视频像素已可解码或每帧不同。

| 会话 | 身体记录 | 相机索引/MP4采样数 | 不同相机主机时间戳 | 相机数据年龄中位/P95 | 身体记录跨度 / MP4时长 |
|---|---:|---:|---:|---:|---:|
| ble_transport_smoke_20260824 | 638 | 638 | 3 | 16.635 / 28.409秒 | 77.570 / 21.267秒 |
| manual_full_test_001 | 9475 | 9475 | 7985 | 0.076 / 0.139秒 | 577.584 / 315.833秒 |
| worn_acceptance_retry_20260824 | 4611 | 18 | 9 | 0.705 / 1.381秒 | 213.348 / 0.600秒 |

“年龄”为身体记录`monotonic_s - camera_timestamp_s`。相机时间是主机读帧后的赋值，不是曝光时刻，因此这里不是端到端物理延迟。全量索引中无时间倒退/跨文件索引不一致；带视频索引的身体帧与相机索引全部匹配。未见可据此恢复真实曝光时钟的独立同步事件。

- `smoke`虽然manifest为complete，638条相机索引连续复用时间635次，最大年龄30.172秒。不能把“文件完整”理解为“采样新鲜”，本次拒绝作为同步动作对照。
- `manual`保留为离线研究候选。约9.6分钟记录中1490次连续复用相机时间，最大年龄1.261秒；manifest遗留running，没有结束记录。未发现JSONL/MP4容器尾部截断，但仍不能把它修写为正常结束。
- `retry`保留为掉流失败例，4611条身体记录只有18条有相机索引。不能用少量成功相机帧掩盖整段失配。
- 三段所有身体记录的`fusion.state`都是`fgp_fallback`。此字段说明记录中没有Stage-A融合生效状态；它不能独自证明SLAM是否成功，也不能说明具体fallback原因。
- 视频是固定30fps封装，实际身体记录频率分别约8.21/16.40/21.61Hz，摄像头有效更新率不能由这几个数替代。直接按视频PTS与衣服时间对齐会改变动作时间。回放必须使用原相机索引与独立身体时间轴，保留重复观测和缺失，不按30fps重新解释身体时间。
- 视频写入时间与身体wall_time差异有小负值，写线程/记录调用先后可造成该现象；不把它解释为负曝光延迟。

小结果分别见`data/manifests/headband_hb02_<session>.json`及[媒体表](../data/manifests/headband_hb02_media_20261014.json)。

## 来源与标定问题

1. **`cam_pos`不是独立视觉定位结果。** 包内`fgp_mocap_adapter.py:forward_frame`用SMPL FK的头部位置加网络tran构造它，主程序记录的是backend帧的这个值。不能用它为“头环世界定位”自我监督，再声称得到独立位移证据。`translation_m`是EgoLocate融合输出，同样不是外部真值。供查阅源码节选与ZIP内容SHA一致。
2. **标定通路有不一致线索。** 三段manifest均选SLAM配置4；包内4.yaml注释为ZED单眼640×360近似参数（fx320/fy312），实际保存视频为1280×720，而RTSP内参JSON为1280×720、fx约961.986/fy961.395，含畸变。未有当时执行二进制/有效SLAM配置快照证明二者如何衔接，因此只能记录“未核验一致”，不能直接认定这是历史失效根因。
3. **外参是root-camera。** `camera_smpl_calibration_new.json`明确保留新旋转、复制旧平移，`position_override`有记录；尚无独立固定head-camera标定合同。转头/弯腰时固定root-camera不应作为刚性外参使用。
4. **来源完整性有限。** 包内源码可能后于录制更新；本轮验证的是包内静态证据，不假设它就是当时运行版本。未在包内找到与三会话绑定的独立SLAM世界轨迹/跟踪状态/地图epoch日志。

证据：[标定与字段来源清单](../data/manifests/headband_hb02_calibration_20261014.json)。未复制设备地址或原始大段资料到报告。

## 时间/坐标原型与实际验证

新增`script/headband/reference_contract.py`：平面刚体解析参考工具，只使用Python标量与math，不是SMPL FK、真实3D校正器或机器人控制器。

12项控制全部通过：转头时骨盆固定、错误固定外参负控制、颈部相对平移补偿、保留真实身体平移、世界坐标变换一致、已知地图重定位不产生局部速度、未知坐标epoch拒绝、曝光到达使用时间区分、显式时钟偏移漂移、仅到达时间不能冒充曝光标定、设备复位、无效dt拒绝。

解析假设中的动态头身变换/标定是已知量；这不证明可从实际衣服准确估计它们，也不覆盖roll/pitch/全身软组织运动。样例说明主张可以写成可检验合同，不能作为A3效果证据。

## 版本、命令、预算

预先登记源码/原ZIP SHA、单会话60秒、256MiB输入/行262144字节/30000行上限；全部JSONL审计耗时每会话约0.10/1.42/0.68秒。媒体检查单文件60秒/128MiB，内存moov上限2MiB；3项独立固定媒体表检查通过。新数据检查未占用ZHY训练额度。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest script.headband.test_reference_contract -v
PYTHONDONTWRITEBYTECODE=1 python3 -m script.headband.recording_audit --archive '<既有EgoLocate ZIP>' --session manual_full_test_001
PYTHONDONTWRITEBYTECODE=1 python3 -m script.headband.mp4_inventory --archive '<既有EgoLocate ZIP>' --member EgoLocate/recordings/manual_full_test_001/camera.mp4
```

[HB02登记](../data/manifests/headband_hb02_registration_20261014.json)、[媒体登记](../data/manifests/headband_hb02_media_registration_20261014.json)、[完成清单](../data/manifests/headband_hb02_result_20261014.json)。历史结果只存证，不为更新代码或状态复跑。

## 结果与下一步

有可复用的历史手动录制，**没有一段已经通过正式头环同步根运动效果对照的前置验收**。这不表示所有视频报废：manual可研究回放时轴/跟踪恢复，smoke/retry可测缺失观测处理；不可直接提升为可信训练标签或未见最终测试。

下一HB03固定最低数据条件、同条件对照与停止条件，提供简短补录单和交付字段。正式比较前需要可信相机定位及时间、动态头身关系和独立参考；已有模型/视频文件的数量不能替代这些信息。前端数据就绪后才登记ZHY计算，当前主线运行与资源条件另行核实，不由本侧解除。

## 保存与同步

本阶段README/代码/小清单LOCAL已保存；Git发布以`headband_hb02_publish_20261014.json`回执及Git远端为准。原ZIP与大录制LOCAL_ONLY，0大文件复制/网盘核验。只更新侧支入口；公共索引和主线状态若正有并行修改，不覆盖，其已有侧支链接保持可恢复。
