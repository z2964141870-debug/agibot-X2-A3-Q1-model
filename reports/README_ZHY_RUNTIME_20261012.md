# ZHY运行环境迁移与首轮A3训练

记录日期：2026-10-12。状态：A3文件与运行依赖迁移验收完成；独立短训练即将启动。

## 目标与授权

用户明确要求继续迁移，全部迁移完成后运行一轮训练。接续已通过SHA的A3数据/代码迁移，补齐其运行环境和外部源码依赖；通用YUANQI与Q1整个历史工程不在本轮范围。新机根 `/opt/data/private/hechengzhu/ZHY`，四目录保持。旧机器不启动GPU计算或训练，不自动恢复旧账本。

## 方案与预登记

- 原22GiB conda运行环境分批原文件迁移，每批及完整清单SHA校验后，在新机用conda-pack0.8.1按新绝对路径重定位；工具隔离于data/tools，不改变原训练环境。
- 补齐IsaacLab源码（原commit37ddf626871758333d6ed89cf64ad702aef127d0），复制134MiB A3 overlay，修正其pyvenv.cfg/解释器链接、.pth与IsaacLab editable路径。保持第三方代码实物内容身份，环境注册路径变化单独登记。
- 新机额度28 CPU等效/16GiB RAM，内存保护按cgroup limit-usage+inactive_file估算可用值，同时记录RSS/cache/OOM，不使用宿主251GiB当作额度。
- 独立实验A3_ZHY_20261012_E01/R11：官方035 PT权重初始化，优化器/计数清零；16环境、4minibatches、epochs1、LR尺度0.1、seed0，目标2更新，每次保存。复用已验证VerifiedFineTuneTrainer与持久化callback；这是迁移后运行检查，不是长训或效果验收。
- 5秒CPU/GPU/内存/boot/dmesg监测，CPU90/GPU85°C连续2采样保护、可用内存保留4GiB、OOM/异常/900秒超时停止；无自动恢复、不开机启动、没有并行计算。内核监测使用容器可读dmesg，不把空journal视为完整覆盖。

## 代码、产物与恢复

`script/project_tools/prepare_zhy_runtime.py` 仅调整新机复制的环境路径。`script/a3/zhy_trial.py` 复用corrected_trial，独立数据/日志/训练目录，拒绝重复登记/启动。

- 迁移档案/完整SHA：原机data/transfers/zhy_runtime_20261012与logs/zhy_runtime_20261012，服务器直传，Mac无大型文件。
- 新机环境：实际base位于data/env（公开入口data/environments/isaac-runtime指向../env）与data/environments/a3-sonic；IsaacLab在script/vendor/IsaacLab。
- 新机实验登记/状态：data/experiments/a3_zhy_20261012_E01。
- 新checkpoint：data/training/a3_zhy_20261012_E01/R11_001_s0。
- 训练console/健康/optimizer审计：logs/a3_zhy_20261012_E01。

迁移和依赖验收后才执行register/run；不调用旧E22/E25/R01–R10训练入口。正常结束且同boot后，独立verify核验step0/1/2 SHA、CPU重载、网络/优化器有限性、8真实optimizer步与辅助loss，未做效果评测。全部模型/档案LOCAL_ONLY，原件保留。

## 当前验证与边界

原机打包连接中断后，2026-10-12 UTC02:59:16只读核验boot为7eb95ab3-684b-4cfd-99af-44d83c2292a6、uptime309.64秒，wtmp记录北京时间10:53/10:54的开机事件；没有原boot附近连续遥测，不能判定原因。原A3服务disabled、GPU无计算进程。虽打包日志100%，18379440128字节tar经列表检查报Unexpected EOF，拒绝传输/使用；截断文件保留为失败证据。改用migrate_zhy_runtime.py逐批<=512MiB原文件复制和SHA验收，已通过数据部分不重传，不重复原机打包负载。

原文件分批迁移完成：173862普通文件、22347506787字节，48批及完整SHA通过，manifest SHA8eacb64fffd89274a91d6f243ee62b011130387fc58245dbfe83e3d5b6c60921。两份原用户缓存Isaac扩展实物纳入raw副本。IsaacLab与overlay补充4147文件/130247799字节全SHA通过，manifest SHAfaf50bca55409a576f3ab8f4ed94733427350698b0ecc65f0bd1e5eb20102132。

新机重定位初次出现不完整输出：conda-pack0.8.1 NoArchive.__exit__返回self，会吞掉底层异常；已在调用时令其传播异常并做全条目存在检查，不修改第三方安装文件。明确异常为二进制prefix替换negative padding，新长路径超过原安装前缀可用空间。选择ZHY/data/env实际短路径，公开入口仍保持data/environments/isaac-runtime；既有不完整输出改名保留为incomplete_01/02。conda原托管文件text/binary模式从原安装包缓存导出273条，不把原机安装目录和build placeholder混用。训练未启动，重定位需再次验收。

数据8440文件/21.07GiB与68链接已在上一阶段通过，见[迁移记录](README_ZHY_MIGRATION_20261012.md)。最终no-archive处理175255条目并检查全部存在，解释器Python3.11.15、torch2.8.0+cu128/CUDA12.8、OpenSSL3.5.7实际CPU导入通过；overlay base_prefix确为ZHY/data/env，6个IsaacLab editable入口和overlay路径已修正。所有环境/IsaacLab链接扫描无断链。可选smpl_sim在原base环境亦未安装，当前False，仅涉及SMPL效果评测，不作为本轮训练验收前置。

9项CPU契约（缓存可回收/实际内存压力/OOM记录/估算边界、原boot且READY才启动、历史checkpoint拒绝、遥测失败停子进程及既有温度保护）通过；官方launcher --preflight检查16个训练motion文件通过，未产生训练/仿真。已确认新机GPU空闲26°C、CPU包温41–43°C，dmesg可读；内存计费高主要是文件缓存，RSS约1GiB。验收清单见data/manifests/zhy_runtime_20261012.json。下一步只登记并启动R11一次，训练结果尚未产生。

验收命令：在新机根运行data/environments/a3-sonic/bin/python -m unittest script.a3.test_zhy_trial script.a3.test_reboot_trial；ISAAC_PYTHON显式指定overlay、MOTION_FILE指定新机data/experiments/a3_fullchain_20261010/motionlib/train，运行vendor/train_a3_035_fromscratch.sh --preflight。训练顺序为python -m script.a3.zhy_trial register、run，正常结束后verify、report；不重复register/run，不恢复原机旧实验。
