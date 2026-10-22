# FGP当前ZIP与历史推理身份复核

2026-10-22；目标轮3。只流式读取本地ZIP中的指定成员，不解压大模型、不pickle加载、不推理。服务器SSH经Windows仍banner exchange timeout(exit255)，未新跑远端作业。

## 实测结果
当前FGP-main.zip内两权重SHA与历史E003/E004记录完全相同：
- Canonicalizer U100hL7.07h_residual.pth：384845450字节，SHA b613eb8226b26b9dc82904c22f4aec78773992503337c3cbde674c998f61ce12。
- LIP LIP_10_real_denoised_7.07_10imu_10.pth：26748498字节，SHA efc4bb8583bf053890d9ae754e97415c8658dba237893982aa51b9541edbb9a2。
- 走路183926与单腿193310两场，各acc/rot/device2bone/pose/joint五成员，全部大小和SHA匹配历史E003/E004。
共14/14检查通过（10数据成员，2权重在两记录各比一次，不是14个独立实验）。

因此包名/日期不能证明比10/8历史checkpoint更新。这不否定用户另有最新版，只证明目前定位到的这份包的两个核心权重相同。需继续寻找后续交付文件/权重指纹，不凭记忆称已找到最新版本。

## 缺失依赖
历史离线入口还加载独立velocity扩展和clotho_v2_displacement/last.pt。当前FGP ZIP中未唯一定位此权重；仅凭GlobalPose/ASIP文件存在不能当作同一速度模型替代。缺速度头时可以设计pose-only诊断，但不能静默填零根轨迹并声称全身移动已验收。

## 时间合同发现
历史body_only导出 frame_index=59..T-1，source_time_s却从0开始（导出段相对时间），两者不是同一时间原点。后续必须按源frame_index连接包内标签，另存导出相对时间与原始参考时间。已有source_time_s不直接判为错，它可表示相对时间；不可拿它直接join原始标签。原始text与PT的帧数不同，不能硬套设备时间。

## 复现与产物
- 脚本：script/teleop_research/verify_historical_inputs.py
- 清单：data/manifests/teleop_historical_identity_20261022.json
- 输入历史：yuanqi/data/manifests/q1_cloth_fgp_20261008_E003_summary.json 与 E004_summary.json。
- 命令见脚本CLI；输出使用open(x)拒绝覆盖。全部21项标准库测试通过，其中身份流式读取1项；工具测试不代表人体重建质量或机器人效果。

## 三维评估与后续
创新性无新增；实用性确定复现可用源并发现速度依赖缺口；严谨性将“最新版”与实际SHA区分、源帧与相对时间区分。先查新版FGP交付记录和速度资产，再核远端文件与资源；不重复训练E21等失败路线。所有记录LOCAL_ONLY，未Git提交或推送。
