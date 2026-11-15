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
