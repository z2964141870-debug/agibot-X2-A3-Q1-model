# data：数据与大型工件

保存原始录制、人体参考、训练输出、checkpoint、导出模型、评测数据、图像和视频。本站点以 `hp3090:/media/yu/FAFF-E9771/YUANQI/data/` 为后续训练主位置。

建议按用途和实验编号组织：

```text
data/
├── manifests/     版本化工件索引：文件大小、SHA-256、来源与备份状态
├── references/    本次用户资料的只读节选；非完整原始压缩包
├── raw/           原始衣服、相机和机器人数据
├── processed/     清洗、重定向和时间对齐后的数据
├── checkpoints/   训练权重与优化器状态
├── exports/       ONNX 等部署模型
├── evaluations/   评测输出和视频
└── verification/  从百度网盘下载的校验副本
```

这些用途目录随实际需要创建。Git 只收本 README 和 `manifests/` 中的小型清单，其他内容默认忽略。

原始 `EgoLocate_副本.zip` 与双目资料 ZIP 仍在用户 Mac 原位置，未因项目初始化搬动或删除；远端保存的是规划所用节选。索引见 [来源记录](../reports/README_SOURCES.md)。

关键大文件用 `script/project_tools/register_artifact.py` 登记；该脚本只计算哈希并写清单，不上传。备份状态必须与真实证据一致。规则见 [存储规范](../reports/README_STORAGE.md)。
