# BONES到Q1专用原速重定向准备

记录日期：2026-10-24，北京时间。阶段：SM-Q1-BONES-P03-C01。状态：实现及服务器5项数值/端到端控制完成；真实BONES输入尚未到位，独立准入未完成。

## 目标、实现与依据

在[BONES主目标](README_SUPERMIMIC_Q1_BONES_GOAL_20261024.md)下建立独立来源/协议，不将G1关节直接写入Q1，不伪装PHUMA来源。新增bones_retarget.py、bones_retarget_control.json及bones_retarget_mapping.json。入口要求[P02-A01](README_SUPERMIMIC_Q1_BONES_ARCHIVE_20261024.md)完整原包接入收据、冻结P01名单和[M01](README_SUPERMIMIC_Q1_BONES_G1_MODEL_20261024.md)已核官方运动学的源模型身份；每批显式offset/count，最多128片，新输出不复用。

按[F02正确命名CSV](README_SUPERMIMIC_Q1_BONES_FORMAT_V2_20261024.md)读120Hz/厘米/度，以50Hz原时轴插值29关节/根姿态，经实际G1 FK得到14身体参考。只减固定水平原点，不按首帧倾斜旋转世界位移；源/目标站姿分别按真实足碰撞底部落地，腿长确定显式位移尺度。复用既有纯函数初始化、Mink Q1Retargeter和Q1导出，独立配置保持2rad/s、原IK/支撑相关门，没有为增量放宽或减慢动作。源模型为声明的官方兼容候选，仍不冒称BONES原生成模型字节身份已证实。

核提取CSV大小/SHA和metadata帧数，保存完整IK、源FK和Q1导出；前后核代码/收据/名单身份、每片核OOM与存储预算、写进度和失败明细。即使转换器门通过，转换输出训练标志仍强制false，状态为INDEPENDENT_CHECK_PENDING；必须经BONES独立准入，不能沿用PHUMA收据进入训练。

## 实际验证与故障记录

第一次5项中4通过，端到端收据检查因inventory_amass.identity只接受Path而传入JSON字符串失败，原生退出1；未产生真实参考/训练。随后复用既有支持str/Path的supermimic.q1_assets.identity修复。21:51:31第二次5项全部通过、原生退出0、5.074秒：原始Frame100至112的120到50时间/原点；三帧独立手构状态与MuJoCo FK；错误30Hz/关节顺序拒绝；实际Q1静态初值IK及缩放根移动；合成tar接入→冻结名单→源FK→Q1 IK/导出全链路且不可训练。

端到端控制是13原始帧/0.1秒的人工夹具，导出6个Q1帧，不是实际BONES，不计入训练/开发/测试时长。没有物理积分、策略推理或优化器更新。Mac仅语法检查通过；四部署文件SHA与本地逐一一致。失败A及通过B日志均保留，不覆盖失败结果。

## 版本、命令与工件

前序发布96675a0c1658b82c95fce980cf9354b76f0df50d。精确代码SHA、原生控制输出及初次故障见[控制收据](../data/manifests/supermimic_q1_bones_retarget_controls_20261024.json)。huanyu日志logs/BONES_P03_retarget_controls_A_20261024.log和logs/BONES_P03_retarget_controls_B_20261024.log。

```bash
/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -m unittest discover \
  -s script/supermimic_q1_sources -p test_bones_retarget.py -v
```

真实转换必须先有实际接入收据，显式传--root、--intake、--model-comparison、--output、--offset、--count、--expected-oom-kill，并通过P00外层保护执行。本阶段未运行真实转换，不能把该命令准备记成已处理候选。

## 结果与下一步

新增真实Q1参考0，训练更新0。下一实现BONES独立来源/原速/几何/导出检查；完整原包到位后先从固定训练候选取小批原速验证，再按合格秒数扩库并核跨划分数值重复。旧实验/输入不变。小代码/README/收据按Q1审计提交推送、核远端并同步两端；大型原料和工件留服务器LOCAL_ONLY。
