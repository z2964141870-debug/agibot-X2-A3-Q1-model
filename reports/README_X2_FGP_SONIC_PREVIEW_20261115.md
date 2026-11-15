# X2 共享 FGP 输入与 SONIC 离线验证

2026-11-15。FGP 仍只做一次衣服接收、T-pose 和人体推理。共享 `smpl_source.SmplFrame` 读取现有 `/api/smpl`，接收状态与帧时效统一在此检查。下游各用原有骨架转换。MorphMimic 转为其 HumanFrame，SONIC 转为根相对 24 关节点和 WXYZ 根朝向，再进入旧 tokenizer 的窗口及打包逻辑。策略分支独立，现阶段 SONIC 入口只在本进程运行推理，实际控制选择仍由既有部署流程负责。

SoC1 以现有录制 `data/analysis/X2_FGP_LIVE_MORPH_20261115_162032/preview/smpl.jsonl` 的前 150 帧，复用旧衣服骨架转换及实装 `x2_smpl_tokenizer_v11release.onnx`、`x2_sonic_s1ft16000_g1_token.onnx`。输出 218 个有限 64 维 token 和 218 组有限 31 维策略动作。最大人体源间隔 95.557ms，骨盆中心残差 0。token 使用合成站立反馈，策略的 990 维本体历史使用全零夹具，动作仅用于验证输入接通。未发布控制或 planner 消息。原结果 `data/analysis/X2_FGP_SONIC_PREVIEW_20261115/recorded_r2.json`。同一录制的首帧也经过共享入口生成 Morph HumanFrame，板端原 Morph bridge 启停测试通过。

三个文件已在 SoC1 工作区逐字节核同。原运行中服务保持，网页当前过期帧被共享入口过滤，未进行正向实时人体验证。衣服恢复后可用以下入口观察 SONIC 编码和策略数值：

```bash
cd /home/agi/projects/smartwear_v2
PYTHONPATH=/home/agi/x2_morph_human_20261113 CUDA_VISIBLE_DEVICES= \
  /home/agi/.old_agi_home.bak/miniconda3/envs/teleop/bin/python \
  -m script.x2_fgp_preview.sonic_preview --preview-url http://127.0.0.1:8768 --seconds 12
```

下一步先用真人静立、小幅单臂与转身数据对照旧 FGP 路径，检查左右、根朝向、动作幅度和变化率，再用实测反馈做只读回放。真机实控继续暂停。

## 真人输入回执与实控缺口

现场回执：180 帧、286 token、286 组策略动作，最大源间隔 90.748ms。策略的真实硬件观测仍未接入。随后在 SoC1 利用第 5729 帧和原 SONIC pack/unpack 函数完成 1623B `pico_intent` v4 内存往返，关节点和根四元数逐值一致，engaged=0，零电机发布。新版 FGP 到正式 token service 仍缺持续转发和断流测试。板端已装 SONIC 二进制具有 `--no-publish` 实现，旧仪式不应直接执行来代替影子验证。下一步由两个终端分工：衣服/FGP 和实时 intent 转发；正式 tokenizer 和 SONIC 后端读取实机反馈、以 no-publish 保存策略目标。验证前不提供本组合的实际接管入口。

## SONIC 原接收端实时影子测试

新增 `sonic_shadow_intent.py`，从共同 FGP 源读帧，经原衣服骨架和静止门生成私有 loopback 15773 intent。原 token service 使用独立 15774/15557 且 planner 端口为 0。原 C++ SONIC 始终带 `--no-publish`，独立测试有界退出，MC 继续管理机器人。

- 20:59:52 真人帧 661，发送 611，token 服务接收约 25Hz。原后端因缺 `--input-type zmq` 在初始化拒绝启动。
- 21:01:24 加入必需输入参数。原 SONIC 真实机器人反馈接近 50Hz，保存 1071 条 joint 和 target 记录。640 真人帧的静止门未就绪，intent 0；原主参考 pose 也未接入，控制状态进入 SAFE_IDLE，外层时限回收。实测尚无完整实时 token 覆盖。
- 三个自有测试进程退出。EM 回读 MC 持续 Running(2230)；终后模式服务超时，不能宣称终后站立模式确认。未执行 MC 写入、控制交接或原仪式。

原件见 SoC1 两个 `X2_SONIC_REAL_FEEDBACK_SHADOW_20261115_205952` / `X2_SONIC_REAL_FEEDBACK_SHADOW_20261115_210124` 输出目录。下一步在不发电机命令的同一环境先接原站立 pose 参考，并由穿戴者静止通过原门，验证同一窗口内存在真人 intent、真实反馈和 token，随后检查断流回退及目标幅度。

## 部署模型配对预检

核实板端 `robot_env.env` 在加载时将 `X2_WHOLE_BODY_TELEOP` 设为 0，旧 Stage 3 调用传入的开关变量和当前启动脚本读取的变量也不同。旧 tokenizer 的 `pairs_with` 指向 armA 14900，且没有共享指纹，不能从有限输出推断 S1FT16000 配对。板端 v16ft8 45000 tokenizer/decoder 的配对文件名、指纹和张量契约一致，tokenizer 规定 full 朝向。新增 `sonic_deploy_preflight.py` 在板上确认旧组拒绝（exit 1）、v16 通过（exit 0）。

同一录制 150 帧用 v16 两模型离线输出 218 token / 218 个动作，实测代码退出 0，原件 `data/analysis/X2_SONIC_V16_PAIR_PREVIEW_20261115/result.json`。仍使用合成反馈及零本体历史，无电机发布。没有改板端默认模型、环境启动项或执行交接。尚缺同一影子会话的正式参考流、实时人体 token、实测反馈和动作包络证据。

## 21:12 v16 真人与实测反馈同会话

用板端 `idle_stand.x2m2` 和原 pose watchdog 隔离发布站立参考，原 tokenizer/SONIC 后端均选配对的 v16。后端全程 `--no-publish`，MC 未停止。真人 intent 14–15Hz、机器人反馈 50Hz、token 达 50Hz，原后端明确记录 WHOLE-BODY ENGAGED。600 个正常 tick 和 101 个归还 tick，测试干净退出，MC 回读 Running(2230)/STAND_DEFAULT。

**动作不通过。** 对齐 701 条采样的时间戳后，原 SONIC `target_pos.csv` 腰部横滚的相邻目标最大跳 2.274976 rad（约 130.35°），最大目标与实测位置差 2.38716 rad（约 136.77°）。这是 shadow 目标，不是机器人实际运动。运行未提供真机调参预设，日志显示目标限幅未启用。需要查清朝向、站立参考和目标跳变来源，并以实际拟部署参数、保留保护再做影子复验。当前不可把该影子命令去掉 `--no-publish` 后使用。结果在板端 `data/analysis/X2_SONIC_V16_SHADOW_20261115_211209`。
