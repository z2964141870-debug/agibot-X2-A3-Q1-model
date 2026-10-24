# BONES独立参考验收准备

记录日期：2026-10-24，北京时间。阶段：SM-Q1-BONES-P03-V01。状态：实现和服务器8项控制通过；真实动作包未接收，无新训练。

## 目标与实现

承接[C01原速重定向](README_SUPERMIMIC_Q1_BONES_RETARGET_20261024.md)，新增bones_verify_reference.py和独立控制。验收不调用BONES解码器或转换器来产生期望值，使用标准CSV接口重新读原始Frame、root及完整29关节列，重建120Hz源状态、50Hz源FK、源/目标站姿落地、腿长尺度和映射目标。逐项比对源FK/目标trace、根位置/姿态、Q1状态/速度/关节及身体导出。

22项门保留原有限、原速、2rad/s、限位、IK P95 5cm、固定落地10cm、足穿地1mm、90%支撑15mm/最大30mm、自穿透3mm等要求，并加入独立源FK、映射和根姿态检查。另生成6位小数的原生数值身份，去除全局平移/初始航向/Frame偏移并规范四元数符号，供后续跨批重复隔离；仅生成指纹，尚未做真实全库去重。逐片通过不等于库就绪，corpus_ready_for_training始终false，仍需合格时长/独立移动集/去重及训练登记。

输入manifest、P01名单/来源分组、Q1/G1身份、转换器及验收依赖前后核SHA，输出新文件不复用；复用既有独立足底几何纯函数，不调用PHUMA来源解析或伪造其收据。源G1仍按M01声明候选，不声称BONES原生成模型身份或真机通过。

## 实际验证与证据

首轮7项通过后，补验收依赖版本冻结与表头空白一致性；22:05:59最终8项/9.957秒原生退出0。涵盖独立单位/原时轴，平移/航向/Frame变化的数值查重不变及关节变化可区分，错误别名/缺帧拒绝，真实模型上的合成完整验收，重算SHA后的源/目标错误、改慢动作、改位移尺度以及模拟验收源码变化拒绝。后者只mock身份函数，未修改共享源文件。Mac仅语法通过；两部署代码SHA与本地一致。

全部是人工13原生帧/0.1秒夹具和真实模型FK/IK控制，新增真实BONES Q1参考0，物理积分0，优化器更新0。收据[supermimic_q1_bones_independent_controls_20261024.json](../data/manifests/supermimic_q1_bones_independent_controls_20261024.json)保存精确SHA和原生结果。日志logs/BONES_P03_independent_controls_A_20261024.log及B_20261024.log。阶段前发布3cb933c74df3ec606b05e999266d396be3cc3586。

```bash
/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -m unittest discover \
  -s script/supermimic_q1_sources -p test_bones_verify_reference.py -v
```

真实独立验收须先有完成的BONES原包接入和重定向manifest，再传--root、--motions、--output、--expected-oom-kill；本阶段未执行真实验收，不把测试输出当训练集。

## 下载变化、资源与下一步

22:05:59 D:\tmp先前6.895GB的partial已消失，未见完整包，新增0字节partial。用户随后确认一直未下好，要求代理尝试Git直接取得BONES-SEED。另一下载目录探测SSH Broken pipe，尚无读取结果，不据此称文件不存在。下一核官方Hugging Face Git/LFS/Hub权限与服务器直连能力，不再依赖用户浏览器下载；不能把查看器代码或LFS指针当动作数据。

22:07:33 huanyu同boot、OOM14，内存26945957888字节、GPU38摄氏度/17788MiB空闲/利用率0%，盘余1964940419072字节，无BONES tar.gz。记录/代码按Q1审计提交推送并核远端，重要小文件同步两端；大工件LOCAL_ONLY，不新增Mac大文件。
