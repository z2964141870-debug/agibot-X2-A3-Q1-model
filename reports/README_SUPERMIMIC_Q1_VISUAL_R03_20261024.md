# Q1 R03完整动力学评测回放

记录日期：2026-10-24，北京时间。阶段：SM-Q1-V02。状态：完整渲染、文件/帧数核验、九帧目视与浏览器播放完成；策略移动与跟随目标未达。

## 目标与范围

检查[R03最终结果](README_SUPERMIMIC_Q1_R03_CONTINUATION_20261023.md)对应的真实动作。直接渲染本轮已审计的MuJoCo闭环保存状态，左侧为运动学参考、右侧为策略实际动力学结果。使用[新接触协议入口](README_SUPERMIMIC_Q1_RENDER_PREP_20261024.md)，本阶段不新增物理积分、策略推理或训练，不将世界模型想象轨迹当成动力学结果。

## 完成内容与决定

R03于06:11:04完成八阶段流水线后，逐片调用`render_audited_trace.py`，三命令均原生退出0；每片原生渲染收据及`audited_render.json`通过。EGL已实际执行，资产与admission的`Q1_FOOT_PRIORITY_V2`一致。两个冻结训练源码目录保持原身份，09:52:14重新verify_plan通过。

新增独立`script/supermimic_q1_analysis/visualization_r03.html`，明确R03/累计15000轮及未达跟随目标；切换动作同步显示全身、腿部RMSE和最大水平离起点距离。保持既有展示风格；原18863/R01不改写。新18865只经服务器回环HTTP和SSH本地转发提供，保留供查看。

## 版本、命令与工件

服务器`huanyu:/opt/data/private/YuanQi_Q1`，Python为`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。科学冻结发布`d5f9476125f2c3c7ed2d7c1e6ac898e1f69d6429`；V02入口发布`9af07bbe47b5a50996a5de0fbe83e3055f06ab9f`。新增页面版本由本次Git历史及SHA标识，不修改原科学文件。

final checkpoint为205225747字节/SHA `10d0838663b5d24215f493ce0738228597452e899b1a378df6d6b453b7e10ea7`；页面6020字节/SHA `4d3b8bd80c2ce1fabb4f35e57a1eaf9ec1103e5ea0509254b6603d5922abd3de`，部署副本与源码一致。

原始输出为服务器`data/supermimic_for_q1/V02_R03_final_20261024/render_{walk,squat,punch}/`：MP4、三PNG、原生收据、审计绑定、ffmpeg日志。原生执行日志为`logs/supermimic_q1_V02_R03_render_{walk,squat,punch}_20261024.log`。

已实际调用的走路命令如下。现有输出拒绝覆盖，恢复观察不要重渲染；另两片是`042_squat_shallow_repeated`和`058_punch_front_alternating`，各用独立输出目录。

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 $PY -u script/supermimic_q1_analysis/render_audited_trace.py --root "$PWD" --evaluation data/supermimic_for_q1/R03_B128_extend12000_20261023/eval_final/receipt.json --audit data/supermimic_for_q1/R03_B128_extend12000_20261023/eval_final_audit.json --clip 001_walk_front_slow --output data/supermimic_for_q1/V02_R03_final_20261024/render_walk
```

精确三条命令、全部收据与视频/PNG身份见`data/manifests/supermimic_q1_V02_R03_verification_20261024.json`；ffprobe核验程序及退出记录见`logs/session_records/supermimic_q1_V02_R03_verification_20261024.log`。

## 实际核验

独立读取两种渲染收据并重核所有输出大小/SHA；ffprobe对每片实际count_frames。尺寸均1280×548、25fps，完整帧数/时长如下。浅蹲奇数控制步向上取整，比物理时长多0.02秒。

| 片段 | 保存控制步 | 视频帧数 | 时长 | MP4字节 | 最小像素std |
| --- | ---: | ---: | ---: | ---: | ---: |
| 走路 | 1652 | 826 | 33.04秒 | 1224767 | 73.497 |
| 浅蹲 | 1157 | 579 | 23.16秒 | 609098 | 73.639 |
| 出拳/复用开发 | 678 | 339 | 13.56秒 | 470404 | 73.629 |

视频SHA依次为`31ea6784499db2d9986473cc006d95d86d61f8895dc3c62ca3e7522eb7e63984`、`aa8ddd7d2c4bf51e51a457b6841fc2500605c4e041c67aadfdc24b03b704745c`、`3f89f7457166660ea673443af35bdaa2138d0f1450e82e6e1fb5ceeaca25dfa1`。

每片首/中/末PNG通过大小/SHA与1280×548尺寸检查；取回9张小预览和335209字节合成图到Mac`/tmp/yuanqi-supermimic-q1-20261023/data/supermimic_for_q1/previews_V02_20261024/`，没有下载模型、轨迹或完整视频。九帧合成图实际目视，未见空白、机器人裁切或重叠。参考与策略动作存在明显差异，策略走路仍近原地，与最大离起点距离2.912cm吻合；不能由站稳和上肢改善称为完整移动跟随。

浏览器三动作实际切换成功，readyState4、解码1280×548、error=null、paused=false。走路时间1.599→19.581秒、出拳0.242→10.742秒；浅蹲首次跨循环，另重查0.284→0.661秒。桌面1280px及手机390px均实际截图，documentWidth等于viewportWidth；手机videoWidth374px，无横向溢出或文字遮挡。临时视口已恢复，回放标签已保留。桌面/手机截图只留被忽略的预览目录。

09:52:14收尾：冻结条件通过，GPU38摄氏度/空闲17788MiB、容器25255591936字节、OOM14未增。HTTP进程790235/start3609830228同boot且命令身份核对通过；Mac18865转发监听PID56727。PID仅为本次快照，恢复需重核。新增物理积分、策略推理、optimizer更新均0。

## 保存与下一步

小证据、源码与报告同步Q1；模型、NPZ和视频只在服务器LOCAL_ONLY，未验证网盘备份。此前R03完成记录已发布，独立核对远端`e75ff14515ef3f34178fd46d7b5e9bfb15567ec8`。本次新增V02阶段另审查diff/payload、提交并显式推送Q1后独立核对远端；提交号由Git历史提供。用户工作区只镜像本阶段小文件，保留A3及研究内容。

入口`http://127.0.0.1:18865/`，原18863仍为R01。服务器HTTP绑定127.0.0.1:18865，目录为本V02输出；转发依赖18864训练主机SSH通路。失联只恢复HTTP/SSH，不重跑已完成实验。服务器命令为`$PY -m http.server 18865 --bind 127.0.0.1 --directory /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/V02_R03_final_20261024`，临时认证不得写入配置。

额外预算有学习收益，仍未解决移动。下一检查参考、奖励及世界模型动作响应，针对“上肢误差下降、走路腿部基本不变/根位移停滞”选择独立改进。D05B根高候选无训练资格，不接入；不自动追加R03预算。A3停止，无真机验证，原可用性门仍true/true/false。
