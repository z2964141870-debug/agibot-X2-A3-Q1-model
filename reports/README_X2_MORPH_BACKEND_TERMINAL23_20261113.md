# X2 Morph 第二、第三终端适配与只读验证

后续：本页保留最初advisory阶段记录；同日新增原生writer、故障与MC交接及真实反馈推理见[后续阶段](README_X2_MORPH_NATIVE_WRITER_20261113.md)。

记录日期：2026-11-13。阶段：TERMINAL23_R1。状态：建议目标接口与独立只读后端已实现并验证；现有电机 writer 尚未接入，真实反馈策略推理仍待实测 IMU 标定。

## 范围与实现

用户在本侧会话授权打通 Morph 第二终端和 X2 后端第三终端；随后明确改为 Mac 有线直连。新增 `script/x2_morph_backend/`，不改主线衣服/FGP代码或原 Sonic 启动/二进制，不停止MC，不恢复Q1训练。

`contract.py`从已固定SHA的Morph `x2.json`加载原生关节与PD元数据；按名称整理31关节测量，核四组及IMU的本地接收年龄、显式安装体/外参、四元数和单位数值。目标必须匹配模型/profile/反馈序号/原生顺序，输出按12/3/14/2分组并精确拷贝，无二次动作变换。`Backend`拒绝超时、缺帧和陈旧结果；通信故障锁定且不后台重置服务。

`__main__.py`提供pipe与ROS只读观察；无电机publisher或MC API。`cpp/native_adapter.hpp`提供原生数组适配；`native_probe.cpp`以真实ONNX回复验证C++解码和模型增益元数据。尚非完整ROS C++部署包编译或电机writer集成。

## 现有底座核对

已读取远端小写`x2`分支`79c4a1e7c019fe57130a2f530fcec78875eb7e4f`的根入口、FIELD_CARD v3、完整runbook、2026-10-23故障结案与部署源码。文档记载Sonic真机键盘行走成功；不据此认证Morph真机。

现场 `/home/agi/gear-sonic/ritual_start_kb.sh`调用`start_x2_deploy_ritual.sh`，后者仍有废弃冻结MC流程的注释，实际参数又不同；归档wrapper默认关闭软退出。新只读入口不运行这些脚本。原安装二进制SHA `466a9864e01d6b522e87d199ff87c6e1b98e8e07c1ba845703403c7d1167e97d`未改。

默认登录Python缺`aimdk_msgs`，系统Python另缺`zmq`；并非SDK不存在。进程内组合项目Python3.10与`/home/agi/aimdk_msgs_pc2`、ROS Humble两处Python路径后，消息及类型支持加载通过；实际结构为`header:MessageHeader/state:DomainErrorState/joints:JointState[]`。无系统安装或配置修改。

## 实际结果

| 项目 | Mac | SoC1 Orin NX |
|---|---:|---:|
| 边界单测 | 11通过，0.004s | 11通过，0.020s |
| 独立真实ONNX IPC请求 | 83 | 83 |
| 原生候选数 / C++精确匹配 | 64 / 64 | 61 / 61 |
| 原生变换最大差 | 0 rad | 0 rad |
| 全部完成RPC中位 / p95 | 2.389 / 3.424ms | 9.256 / 13.155ms |
| 成功候选RPC中位 / p95 | 2.459 / 4.206ms | 9.742 / 13.197ms |
| 完成RPC最大值 / >20ms | 8.825ms / 0 | 16.032ms / 0 |

合成人体和固定合成反馈，约4秒小样本。完成RPC统计包含未就绪回复；成功候选另列，故不能将等待回复的低耗时当模型耗时。180ms人体停流后无目标、恢复后重新就绪、独立策略进程退出后通信故障锁定均过。probe仅终止自有服务，不操作主线服务。无新物理仿真、训练、MC操作或电机命令。

Mac直连`agi@10.0.1.41`已成功；Mac有线`en7=10.0.1.10/24`由用户侧配置，Agent未修改网络。此前WiFi地址`192.168.43.12`的旧尝试结束；已关闭自有Windows会话。之后的采样和取证均走有线直连。

真实ROS只读采样3秒：leg1506、waist1501、arm1499、head995、IMU470消息，149个完整31关节快照。分组名称无错误，149条全部通过完整性、有限性、单位四元数及120ms本地接收年龄检查，观测最大接收年龄41.229495ms。IMU实际`frame_id=base_link`，但实测外参未完成，真实反馈策略请求0。

所有候选`publish_allowed=false`。上述接收年龄不是传感器采样延迟；RPC不含FGP/BLE全部计算、ROS回调与writer执行，硬件端到端延迟为null，不认证50Hz整链或自然遥操质量。各端候选数差异来自实际调度与插值就绪条件，不作跨设备逐帧等价声明。

## 证据与命令

- 代码与操作说明：[独立后端README](../script/x2_morph_backend/README.md)。模型SHA：`cf95194793e4732447f66432fc2381d82dd7c3d82c94da15ad074f2cfe295512`。
- Mac原件：`/Users/yu/Documents/ChatGPT/元启/data/analysis/X2_MORPH_TERMINAL23_20261113/MAC_R1`及`ORIN_R1`。
- SoC1原件：`/home/agi/x2_morph_human_20261113/data/analysis/X2_MORPH_TERMINAL23_20261113/ORIN_R1`。
- 单测：项目Python `-B -m unittest script.x2_morph_backend.test_contract -v`。
- 独立IPC：项目Python `-B -m script.x2_morph_backend.probe --output <新的输出目录>`。
- ROS读取：`sh script/x2_morph_backend/run_orin.sh ros --inspect --profile script/x2_morph_human/config/robot_shadow.example.json --seconds 3 --output <新JSONL>`。
- 迁移包：`data/deliveries/X2_MORPH_BACKEND_CODE_R1_20261113.tgz`，SHA`845a5fb1cd4dba0a721dd4f478c6b664a9129d24c3348068d98325a4b2eb3840`。8代码文件部署前目标不存在，部署后全部逐文件SHA匹配；Linux tar忽略Mac provenance扩展头，无代码差异。
- 小记录：[TERMINAL23结果](../logs/session_records/x2_morph_backend_terminal23_20261113.json)。本阶段自身测试进程结束，未在后台保留控制器。

## 后续

先实测IMU安装体与外参并生成独立profile，再做真实反馈→Morph建议目标的只读联调。之后才在独立副本接入既有writer，核Morph零位/限位/PD、故障和MC交接路径，现场准备齐备且获得真机测试指令后才允许电机输出。此阶段不启动原ritual、不复用Sonic动作与增益、不将示例标定改名过门。

Git/主根/设备小记录同步状态以后续publication收据为准。原始工件为Mac与SoC1本地保存，未核验网盘上传。
