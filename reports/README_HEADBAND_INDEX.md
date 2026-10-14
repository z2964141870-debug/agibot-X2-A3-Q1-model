# 头环侧支阶段索引

最新状态：[恢复入口](README_HEADBAND_CURRENT_STATE.md)。本索引独立于主线程，避免在公共索引有并行修改时覆盖它。

| 阶段 | 结果与证据 |
|---|---|
| HB01 | [协议样例、文献与候选方向](README_HEADBAND_REFERENCE_20261014.md)：11IMU、13控制通过，实物尚未验证 |
| HB02 | [完整旧录制审计](README_HEADBAND_DATA_AUDIT_20261014.md)：配对文件完整性可查，但同步/标定/独立相机轨迹条件不满足；12解析控制通过 |
| HB03 | [首次对照与最低采集合同](README_HEADBAND_EXPERIMENT_PLAN_20261014.md)：计划完成、计算未登记；补信息进入控制器前置门 |
| HB05 | [公开数据与接口决策](README_HEADBAND_PUBLIC_DATA_20261014.md)：RoSHI评测代码/数据格式已核验；训练数据条件未完成，下一独立回放工具 |
| HB06 | [离线计分核心与来源阻塞](README_HEADBAND_REPLAY_CORE_20261014.md)：9项自制标量控制通过；公开包两端连接超时/0字节，真实数据WAIT_DATA、本侧循环暂停 |

本侧不接手主线程A3计算，不将工程检查数量当作超过官方035的效果。
