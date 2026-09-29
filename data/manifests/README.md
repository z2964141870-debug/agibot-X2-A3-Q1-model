# 工件清单

- `reference_excerpts_20260929.json`：本次规划使用的资料节选与 SHA-256。原始压缩包不在 Git。
- `artifact_*.json`：由 `register_artifact.py` 登记的模型或数据。

登记后的初始状态是 `LOCAL_ONLY`。百度网盘目标路径只是计划位置；只有上传、大小核验和下载 SHA-256 校验均有证据后才标记为 `VERIFIED`。

清单不得包含账号密码、BDUSS/STOKEN、OAuth token、Cookie、私钥等凭证。
