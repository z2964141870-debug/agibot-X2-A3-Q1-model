# X2 SONIC 两终端操作书

当前用于真人衣服和真实关节反馈的影子测试。终端二固定使用 `--no-publish`，电机由官方 MC 管理。

1. 两个终端都从 Mac 登录 `ssh run@10.0.1.41`。以下命令可在机器人任意目录执行。
2. 终端一启动衣服和网页。若网页已运行，沿用现有终端。

```bash
bash /home/agi/x2_morph_human_20261113/script/x2_fgp_preview/run_live_orin.sh
```

3. 打开 [FGP 网页](http://10.0.1.41:8768/)，等两件衣服都在收数，点 T-pose。准备 3 秒、保持 3 秒，等 SMPL 实时更新后恢复自然站立。
4. 终端二启动 SONIC v16。穿戴者先站稳，静止门通过后再做小幅动作。

```bash
bash /home/agi/x2_morph_human_20261113/script/x2_fgp_preview/run_sonic_shadow.sh
```

看到 `WHOLE-BODY ENGAGED` 表示正式 SONIC 后端收到真人 token。约 12 秒后自动退出，在打印的日志目录查看 `target_pos.csv`。若提示未切到真人 token，先查 `intent.log` 和 `token.log`。衣服网页可留着，关闭时在原终端一按 Ctrl-C。

真机接管暂停。上一轮在只读日志中发现腰部横滚目标相邻跳变约 130°，需完成目标约束和复验。[试验记录](README_X2_FGP_SONIC_PREVIEW_20261115.md)
