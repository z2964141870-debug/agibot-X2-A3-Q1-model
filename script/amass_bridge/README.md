# amass_bridge — AMASS(SMPL-X G) -> SMPL-24 / A3 数据桥

落地: /opt/data/private/yuanqi_ops/script/amass_bridge/  (huanyu, 纯 CPU 开发, GPU 让训)
输入: /opt/data/private/yuanqi_ops/data/motionx/AMASS_smplxg.z01..zNN + .zip (87GB 分卷, 下载中)
输出: data/amass_raw/  (解包后 npz 语料)

## 文件
- unpack.sh            分卷齐(轮询 dl_progress.log 出现 ALL_DOWNLOADS_DONE, 10min 间隔)->体积稳定->解包
                       模式: run / validate(小规模自证) / status ; 日志: unpack.log ; 完成标记: unpack_DONE
- split_zip_extract.py 自研分卷 zip 解包器(生产路径; 无合并临时文件, 逐条 CRC32 校验, 断点续跑)
- extract_zip.py       普通单卷 zip 的 zip64 安全解包器(备用工具)
- smplx2smpl24.py      AMASS npz -> p16_pool 同构 motionlib pkl 转换器(含 FK 校验与 selftest)
- triage.py            walk/run/exercise-stretch/dance/daily/other 分桶统计 -> triage_report.json
- build_cmu_index.py   从 mocap.cs.cmu.edu 抓官方 per-trial 动作名/主题标签 -> data/cmu_index.json
- data/cmu_index.json  已构建: 109 subjects / 2345 trials (2026-10-20)
- self_test_report.py.json 见下"精度"

## 为什么自研解包器 (huanyu 实测, 2026-10-20)
- zip 3.0 "zip -s 0 --out" 合并产物损坏跨卷条目: unzip 报 invalid compressed data / overlapped
- "zip -FF" 修复是交互式的, 管道喂 yes 会死循环 (2GB 日志实证)
- cat 分卷 + unzip: local header 偏移是"所在卷内相对"(逐条目不同), central dir 偏移是"末卷相对",
  标准工具全部误读 -> split_zip_extract.py 按此语义直读分卷集 (字节级验证过偏移规则)
- 校验: 小规模 sha256 全等; 生产逐条目 CRC32, 失败即停, 重跑跳过已正确解出的文件

## 转换器 (smplx2smpl24.py)
拓扑: SMPL-X poses[:, :66] 22 身体关节(含根) 与 SMPL-24 前 22 关节同序; 手指/下颚/眼睛丢弃;
      SMPL-24 关节 22/23 (L/R_Hand) 默认置零 (手向由腕旋转经运动链继承, 即"腕旋转近似";
      --hand wrist 可显式拷贝腕旋转, 默认 zero 避免腕旋转二次叠加)
重采样: 每关节 SLERP + transl 线性插值 -> 30fps (mocap_framerate 源帧率)
输出 pkl (joblib compress=3, p16_pool/m2 同构):
  {name: {pose_aa (T,32,3) [24身体+8零手; 展平前24关节= pose_aa(T,72)],
          smpl_joints (T,24,3) 根相对 FK (SMPL_male.pkl 模板, betas=0, 与 SMPLight 约定逐毫米一致),
          root_rot (T,4) wxyz, root_trans_offset (T,3), dof (T,29) 零, fps:30,
          pose_aa_72/betas10/src_fps/gender/source (扩展元数据)}}
用法: smplx2smpl24.py convert --input data/amass_raw --out-dir data/amass_smpl24 --jobs 16 --verify 8
      smplx2smpl24.py verify --input <npz>   # 只跑体检(身高/关节角/NaN)

### 精度 (合成自测, self_test_report.json, 全 PASS)
- FK 自洽: 恒等姿态 vs J_regressor 误差 0.000000 mm; 根旋转 90° 解析对照 0.000000 mm
- 60->30fps 精确 2:1 已知答案: FK 关节最大误差 0.000077 mm, transl 0.000028 mm
- 59->30fps 常角速度解析对照: 旋转 0.000004 deg, transl 0.0569 mm
- 30fps 直通: 逐位相等 0.000000; 手关节置零/腕拷贝变体: 0.000000
- 站姿身高体检: 1.827 m (模板 1.791 m) ∈ [1.5,1.9]
- FK 与 p16_pool 所用 SMPLight 对拍: 根相对关节最大差 0.000 mm (同一骨架约定)

## triage.py
优先级 exercise_stretch > dance > run > walk > daily > other (exercise 单列为 warm-up 特训弹药)
CMU 数字文件: data/cmu_index.json 官方 per-trial 名优先, 其次主题标签;
其余数据集: 路径关键词 (ACCAD/EKUT/SFU/DanceDB/HDM05/BMLrub 等描述性文件名)
用法: triage.py --root data/amass_raw [--scan-frames 50]  (scan-frames 读 npz 头, IO 重, 默认关)

## 红线遵守
- 未触碰任何训练目录 / gate_log; 全程 numpy/scipy CPU (GPU 让训: 开发期间 GPU0 60%->12% 波动)
- 分卷未齐不解压 (unpack.sh 在 ALL_DOWNLOADS_DONE + 体积双采样稳定前不做任何解包 IO)
