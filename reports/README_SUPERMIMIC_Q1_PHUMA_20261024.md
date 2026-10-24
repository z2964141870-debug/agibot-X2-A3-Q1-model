# PHUMA驱动的SuperMimic for Q1

记录日期：2026-10-24，北京时间。阶段：SM-Q1-PHUMA-P01。状态：旧R04V2已按用户要求停止；PHUMA原包身份/两份格式抽查完成，官方模型映射已核对，独立输入准备进行中，新训练尚未启动。

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

下一核对全库源帧率、根轨迹、关节语义与源片段分组；依据官方train/test并对相邻chunk/同源clip做分组隔离，固定首批覆盖移动/下蹲/上肢的候选名单后才进行Q1重定向。30→50Hz仅重采样，不慢放。来源、未通过参考和质量门全部记录；Q1接触/限位/速度、FK/时间导出和原地替代移动风险分别核验。新参考合格后才登记随机初始化的独立训练与完整动力学评测。

## 版本、命令与产物

服务器Python `/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`；运行前重新核验资源和其他作业。源码与小清单先审核/推送Q1并独立核对，再部署/验证身份。获取入口：

```bash
$PY script/supermimic_q1_phuma/acquire_source.py --output data/references/phuma_source_2faf15b5
```

此命令尚待执行；完整数据/模型文件不进Git。官方来源为PHUMA GitHub、Hugging Face数据卡及固定提交的 `src/curation/preprocess_g1_folder.py` / `asset/humanoid_model/g1/custom.xml`。获取收据/全库扫描/重定向及新训练的实际结果在执行后补记，当前不预写通过。

## 保存与下一步

停止阶段 `307826c8ff7e26af6c7ecd47b83dcdd0a70189a3` 已推送并独立核验Q1一致。本阶段新源码/准备记录尚待审核发布；无PHUMA训练作业、无新效果结论。所有大型工件LOCAL_ONLY；旧R03/R04V2与D05B不作为本轮输入或通过证明。
