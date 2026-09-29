# YUANQI 初始化记录

日期：2026-09-29。

## 用户确定的约定

- 训练主机：`ssh hp3090`。
- 项目：`/media/yu/FAFF-E9771/YUANQI`。
- 目录：`data`（产生的数据）、`logs`（纯执行记录）、`script`（代码）、`reports`（重要 README）。
- GitHub：`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。
- 关键大型模型：百度网盘；先排查已有入口。

## 完成内容

1. 确认 SSH、3090 GPU 和磁盘读写挂载；目标路径原先不存在，未覆盖其他项目。
2. 从用户指定的空 GitHub 仓库克隆，项目分支使用 `main`。
3. 建立四个主目录和记录规范，将前一阶段五份学习/研究文档归入 `reports/`。
4. 同步规划使用的 17 个来源节选到 `data/references/`，逐文件 SHA-256 核验一致。没有复制 716 MB 的整个 EgoLocate ZIP，也未改动原始附件。
5. 加入只读环境检查、暂存区检查和工件登记工具；三份 Python 工具通过语法检查。
6. 实测 `.gitignore` 会排除示例 checkpoint、ONNX、原始资料 PDF 和未经审核的运行日志。
7. 用真实相机规格 PDF 验证工件登记流程；记录大小 1,092,597 字节、SHA-256 和 `LOCAL_ONLY` 状态。
8. 保存主机检查日志和关键说明，作为本次初始化提交内容。提交身份可用 `git log -1` 查看；GitHub 同步以本地 HEAD 与远端 `main` 的比对为准。

## 百度网盘结论

发现官方 Linux 客户端 `8.7.0`，入口 `/opt/baidunetdisk/baidunetdisk`，具有执行权限；桌面入口和用户配置目录也存在。检查时客户端未运行，未发现常用 CLI。

目前没有验证登录和文件往返，也没有上传任何模型，因此没有已完成的网盘备份。本次没有安装新客户端、读取认证内容或创建云端目录。第一次有重要大型工件需要备份时，再验证现有客户端的登录与传输；如入口不能满足需要，再讨论替代方案。

## 证据与边界

- [主机检查 JSON](../logs/session_records/host_inventory_20260929T143803Z.json)
- [来源节选清单](../data/manifests/reference_excerpts_20260929.json)
- [工件登记结果](../data/manifests/artifact_source_review_20260929_14c7538d23387f76_4beb9137.json)
- [存储规范](README_STORAGE.md)

此阶段只建立训练项目与记录基础设施，没有启动训练、安装训练依赖或控制机器人。服务器上原有 X2、A3、FGP、IsaacLab 等项目保持原状。

下一步按[训练工作包](训练工作包.md)进入 P0：核验可复现训练框架、准确机器人资产及 PPO 入口。
