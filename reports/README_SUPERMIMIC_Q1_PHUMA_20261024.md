# PHUMA驱动的SuperMimic for Q1

记录日期：2026-10-24，北京时间。阶段：SM-Q1-PHUMA-P01/P02/P03。状态：旧轮已停止；P02全库分组/独立审计通过、12片Q1几何0合格；P03首帧修正后1/12合格。下一固定112候选批量筛选；新训练尚未启动。

## 目标与范围

用户要求停止当前整轮、重新开始、停止使用目前A3来源动作，改用PHUMA数据集。继续SuperMimic世界模型训练路径。新实验采用随机初始化的world model/actor/critic，不加载旧A3来源参考训练得到的Q1权重；保留旧实验，不重启R04V2或A3。

工作根为huanyu `/opt/data/private/YuanQi_Q1`，代码副本 `/tmp/yuanqi-supermimic-q1-20261023`、Git分支Q1。新源码置于 `script/supermimic_q1_phuma/`，不改旧冻结目录或历史终态。大型源包、官方模型、转换结果和权重保存在服务器data，Mac仅保留代码/报告/小清单。

## 已完成与关键发现

11:26:30旧R04V2总控/worker均STOPPED_SIGNAL，A932轮/14912控制转移，B未启动；11:27:32核验无存活组成员、三份权重SHA及冻结条件通过，详见[停止记录](README_SUPERMIMIC_Q1_HORIZON_PAIR_20261024.md)。

PHUMA原包 `/opt/data/private/yuanqi_ops/data/phuma/data.zip` 实测3,416,370,206字节，SHA256 `7bd38634f1ec075af907a2bce10616b083a5414885674ce6630664124e22e495`。ZIP有146152个NPY、展开4,290,810,000字节；G1与H1-2各73076。两个机器人版本不是两套独立采集。新主线优先G1源模型到Q1的重定向，不直接复制G1关节角，也不使用历史PHUMA→X2/A3派生数据。

抽查G1/H1-2各一份，字段为root_trans、root_ori、dof_pos、fps；均为30Hz、有限数组，G1为48帧/29关节，H1-2为35帧/27关节。初查使用仅允许NumPy数组构造器的受限反序列化，不执行包内脚本。此抽查不证明全库格式或质量。

官方源码固定 `DAVIAN-Robotics/PHUMA` 提交 `2faf15b52a3498bfcc3352ecb443dc0b5cd642ac`。已读取官方G1配置、MJCF与预处理入口：root_ori使用xyzw，读取到MuJoCo须转wxyz；G1官方MJCF启用23铰链，导出NPY在前19关节后和末尾各插入3个零腕部列，形成29列。须核验零列并恢复23关节顺序，不能把29列直接塞进该模型。官方源包不含LAFAN1/LocoMuJoCo，本轮先使用已在ZIP中的动作，不宣称已有额外两库。

## 输入准备与验证顺序

新增 `acquire_source.py`：只下载固定官方元数据、划分文件、G1 XML与其引用网格；逐文件核对Git blob身份/大小并登记SHA256，不执行官方安装或下载脚本。源代码许可证已读取为Apache-2.0；原始动作来源与外部资产不据此自动宣称可重新分发，仍留忽略目录。

获取代码 `3f5fbb2050d8063f3537423decfcb32f7c793be8` 已发布并核验Q1远端。11:43:02–11:45:21获取原生退出0，46文件/22,780,139字节；独立逐文件复核大小/SHA及MuJoCo编译通过，模型nq30/nv29/nu23，实际铰链顺序与恢复23列约定一致。源收据服务器 `data/references/phuma_source_2faf15b5/source_receipt.json`、Git小收据 `data/manifests/supermimic_q1_phuma_source_20261024.json`，11,838字节/SHA `533242e77229686dce1d19ac8d4f659c39ca711d45bb7488900b2b47fbe68986`；两份执行记录取回后大小/SHA与服务器一致。0新物理积分/optimizer更新。

服务器请求Hugging Face元数据失败（网络地址不可分配），尚未独立比对官方LFS对象身份；当前原包SHA是已在本机独立读取的固定输入身份，不能称为官方LFS匹配通过。

新增 `motion_io.py` / `scan_corpus.py` / `test_motion_io.py`：受限NumPy解码，验证30Hz、长度、有限值、单位xyzw和六零列。官方test涉及的整个来源组保留test，其余官方train组固定hash留10%development；移除chunk和clip后缀，候选每源组最多一片。固定seed2026102405，首批候选配额train80/development16/test16，分别覆盖移动/下蹲/上肢/其他；时长至少1.98s、根倾斜不超0.6rad、源关节速度P95不超4且最大不超12rad/s。该配额是输入候选，不能称为已合格Q1参考或全库训练。本地六项针对性控制全部通过，无跳过；服务器控制与全库扫描待执行。

11:57:30–11:58:43全库扫描原生退出0，服务器六项控制亦全部通过。73,076个G1 NPY均通过字段/30Hz/单位四元数/六补零列/有限值检查，0格式拒绝；15类别、35,023来源组。官方文件级train/test在5,213个来源组中混合，本轮将这些组整体留test。最终train49,548/development5,438/test17,567/unlisted523片；112候选足额选出（80/16/16），每个来源组仅一片，尚不是Q1合格参考。OOM计数保持14，0物理积分/optimizer更新。

选择清单 `data/manifests/supermimic_q1_phuma_selection_20261024.json` 为114,601字节/SHA `cf22cc86981f088cf4c9ffbe7a5f5b2afbee12a6948295a3fe5acaeb13e0cc11`，执行日志 `logs/session_records/supermimic_q1_phuma_scan_20261024.log` 为5,328字节/SHA `600ceb3a8ba255f08d25946b1f47294858d5eccffa2ee7a2e1cb308305a2a6bf`，取回一致。完整58,852,874字节JSONL留服务器 `data/supermimic_for_q1/PHUMA_P01_scan_20261024/all_chunks.jsonl`，SHA `512df6cd955fe7c7a8b90330e8a429e951f8a6b6237606f2bbc0afed468281d8`。

选后复核发现fitness存在 `_clip_17` 后缀，P01仅处理 `_clip17`，因此P01分组不是最终训练名单。保留该扫描终态/索引；修正为两种clip后缀均移除，在新 `PHUMA_P02_scan_20261024` 重新选择，再独立复核全索引、分组计数和候选原字节。分组只按命名来源茎，尚非subject-disjoint或全库语义去重证明。

新增独立 `retarget.py`、两份配置、`verify_reference.py` 与三项重采样控制，代码已完成、尚待两端数值验证和12候选小批运行。复用只读Mink/quadprog通用求解及Q1导出，输入仅PHUMA G1 FK，不读取A3动作；按中立位落地/腿长比例约0.5082缩放，水平原点归零而保留位移，30→50Hz仅重采样、不慢放。独立验证器显式检查PHUMA 30Hz、Q1 FK/速度导出/源位移，并追加90%帧支撑脚15mm内、最大支撑间隙30mm、自碰撞穿透不超3mm的固定几何门。尚未实测参考是否通过，不把运动学门作为动力学跟随证明。

12:10:51–12:13:43 P02分组扫描/独立重算/12片重定向/独立检查四进程均退出0；九项服务器控制通过。P02共23,225来源组，4,184组官方混合整体留test；train33,440/development3,864/test35,249/unlisted523片，112候选仍80/16/16，独立重读原字节/统计/全索引/分组零交集通过。收据 `data/manifests/supermimic_q1_phuma_P02_scan_check_20261024.json`；P02选择113,320字节/SHA `b26fa28ea35d06321369bc8f50fa487d399182f0711ac3c2f6389340afd8935e`。

P02首批12片独立参考合格0，虽时间/FK/限位/速度全部通过，11片支撑比例或间隙失败、5片自碰撞超过3mm；不是动力学或训练通过。完整小结果 `data/manifests/supermimic_q1_phuma_P02_pilot_check_20261024.json`，服务器原件 `data/supermimic_for_q1/PHUMA_P02_pilot_20261024/` 保留。进一步逐帧核查：每片首帧关节离中立位均被截在0.04rad；下蹲第0帧脚底-142.85mm，后段-63.58至-8.04mm，导致整段固定抬高142.85mm。证据支持首帧从中立位限速起步造成部分失真，仍不能解释所有动态支撑/碰撞问题。

下一独立P03只增加静态首帧Mink初始化，不改真实帧时间/2rad/s速度门、不添加参考帧、保持原质量门；源协议V2。先数值控制与同12片检查，再决定批量转换。PHUMA训练入口及固定3000轮后匹配完整评测流水线已实现，尚未发布/验证/登记/启动。

P03实际12:21:15–12:23:05三进程原生退出0、10项数值控制通过。首帧初始化后12片中1训练上肢片通过全部独立门（P02为0）；下蹲原固定抬高142.85mm等异常减少，但其余仍未通过支撑/碰撞，不足以启动三划分训练。P03原件 `data/supermimic_for_q1/PHUMA_P03_pilot_20261024/`，独立收据30,192字节/SHA `370ecb9a6c928b605b738334e9e3a71cfa3c353702fedae0e4c4d686e42f9d98`；代码 `7eae5a90a35f50e832fdaff7182d34e48238920f` 已发布核验。

下一用相同首帧算法转换固定112候选并独立筛选。验证器只需位置/速度/接触几何，改用MuJoCo fwdPosition/fwdVelocity以避免无关约束加速度求解；新增八种Q1状态与完整mj_forward逐数组/接触精确一致的控制，待服务器实跑。新增四项训练输入合同控制，拒绝旧A3协议、pilot、跨划分同源和已变参考；首轮计划3000轮/16环境/batch128/H32，原生MuJoCo在线交互和可微世界模型BPTT，三网络随机初始化，完成后匹配全片评测；900秒无进度停止、温度/OOM保护、无自动重启/追加。尚未登记/启动训练。

## 版本、命令与产物

服务器Python `/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`；运行前重新核验资源和其他作业。源码与小清单先审核/推送Q1并独立核对，再部署/验证身份。获取入口：

```bash
$PY script/supermimic_q1_phuma/acquire_source.py --output data/references/phuma_source_2faf15b5
```

上述获取和以下全库扫描均已完成；完整数据/模型文件不进Git：

```bash
$PY script/supermimic_q1_phuma/scan_corpus.py \
  --archive /opt/data/private/yuanqi_ops/data/phuma/data.zip \
  --vendor data/references/phuma_source_2faf15b5 \
  --output data/supermimic_for_q1/PHUMA_P01_scan_20261024
```

官方来源为PHUMA GitHub、Hugging Face数据卡及固定提交的 `src/curation/preprocess_g1_folder.py` / `asset/humanoid_model/g1/custom.xml`。全库扫描/重定向及新训练的实际结果在执行后补记，当前不预写通过。

## 保存与下一步

停止阶段 `307826c8`、获取入口 `3f5fbb2` 和扫描源码 `400c230fa383e0c9a0c993cb61ac63dc917207e4` 已推送并独立核验Q1一致。下一发布扫描结果与Q1重定向入口后部署；尚无PHUMA训练作业、无新效果结论。所有大型工件LOCAL_ONLY；旧R03/R04V2与D05B不作为本轮输入或通过证明。
