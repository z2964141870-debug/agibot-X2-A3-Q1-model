"""Generate the experiment report from closed-loop receipts and paired estimates."""

from design import D
from assess import read, rows


def main():
    res = read(D / "results.json")
    sel = read(D / "selection.json")
    fit = read(D / "fit.json")
    ver = read(D / "verification.json")
    parity = read(D / "parity.json")
    a, b = sel["pace"], sel["gain"]
    names = {
        "published": "发布版",
        a: "起步渐变",
        b: "命令年龄预测",
        a + "+" + b: "两者组合",
        "linear": "线性外推",
        "fixed": "固定预测",
        "motion": "延迟重练",
    }
    d = res["daily"]
    lines = [
        "# Q1 起身与五秒续走工程验收",
        "",
        f"本轮在 yu-3090 用 50 Hz MuJoCo 完成闭环补训与验收。按日常任务的预设门槛，采用结果为**{names[res['selected']]}**。所有成绩均来自仿真。",
        "",
        "## 日常使用",
        "",
        "两组留出种子各测 24 个倒姿、20 次单推与 5 条连续双推。单推覆盖直线和八字，改变受推方向、角度、速度增量和相位。双推复用同种子的前五个初态，单独统计。",
        "",
        "| 方案 | 静态起身 / 48 | 真实倒地 / 40 | 站住未倒 / 40 | 原 3 秒通过 / 40 | 交还后 5 秒通过 / 40 | 交还后再跌 | 双推通过 / 10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, r in d.items():
        c = r["counts"]
        lines.append(
            f"| {names[arm]} | {r['static']} | {c['down']} | {c['avoided']} | {c['success']} | {c['pass5']} | {c['refell']} | {r['repeat']['pass5']} |"
        )
    lines += [
        "",
        "真实倒地、交还后连续五秒站直、足宽均值 ≤22 cm、位移 >0.25 m、无再次接管和发出时力矩越限，计为五秒起身续走通过。站住未倒单列。原三秒指标沿用原停止时点与判分，五秒指标从首次完整交还起算。双推在首次交还后两秒反向施扰，最迟在第一推后十二秒仍会施加第二推。第二段允许站住化解。",
        "",
        "| 起步渐变 vs 发布版 | 配对数 | 发布版中位数 | 渐变中位数 | 配对相对变化中位数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, label in [
        ("support_roll", "首次单脚支撑横滚峰值（度）"),
        ("three_roll", "前 3 秒横滚峰值（度）"),
        ("late_roll", "第 3–5 秒横滚峰值（度）"),
        ("first_forward", "首秒前进速度（m/s）"),
        ("late_forward", "第 3–5 秒前进速度（m/s）"),
        ("first_lateral_rms", "首秒横向速度 RMS（m/s）"),
        ("release_s", "受推至交还时间（秒）"),
    ]:
        v = d[a]["paired"][k]
        if v["n"]:
            lines.append(
                f"| {label} | {v['n']} | {v['base']:.3f} | {v['arm']:.3f} | {100 * (v['ratio'] - 1):+.2f}% |"
            )
    lines += [
        "",
        "配对限于两边均实际倒地、完成交还且未再跌的试次，并要求对应窗口与测量均存在。根部速度投影到参考躯干朝向，横向指标取垂直方向。首次单脚支撑要求连续四拍一侧载荷 >20 N、另一侧 <5 N，横滚取起点起 15 拍峰值。两列中位数由各自样本计算，最后一列由逐对比值计算。",
        "",
        f"渐变相对发布版新增 {d[a]['paired']['gain']} 例五秒成功、损失 {d[a]['paired']['loss']} 例。发布版与渐变的五秒结局分别为 {d['published']['outcome5']} 和 {d[a]['outcome5']}。",
        "",
        "采纳要求日常静态与动态成功数、双推通过数和再跌次数守住基线，首秒与后段的配对速度比至少为 80% 和 95%，交还延迟中位数最多增加 0.5 秒，两个长窗口的横滚峰值均不得增大。还要求五秒成功率提高至少 5 个百分点，或首次单脚支撑横滚峰值降低至少 10%。各门逐项数值与判定保存在 results.json。",
        "",
        "## 补训与迁移",
        "",
        f"起步渐变只调整完整交还后的参考动作播放速度，身体位置、朝向和关节共用同一小数时间索引，朝向采用同半球归一化四元数插值。开发集在 80% 或 90% 起速、0.6 或 1 秒两种时长及原速间比较，活动候选选为 {a}。恢复原速后保留积累的参考相位差。控制和物理频率分别为 50 Hz 与 200 Hz。",
        "",
        f"命令年龄预测器在独立种子的 {fit['train']:,} 行上训练，另用 {fit['validation']:,} 行验证。两个 128 单元隐藏层，40 轮共 {fit['updates']:,} 次 AdamW 更新，第 {fit['epoch']} 轮验证最佳。静态和动态命令预测 RMSE 分别为 {fit['history'][fit['epoch'] - 1][2]:.4f} 和 {fit['history'][fit['epoch'] - 1][3]:.4f} rad。ONNX 与 PyTorch 最大差 {fit['onnx_max_abs']:.2g}。增益依四种传输条件的静态、五秒动态结果选择，取 {int(b[1:]) / 10:.1f}。",
        "",
        "监督目标按本次命令实际传输时长，在后两拍的原策略目标之间插值。策略输入只取决策时已知的本体量、起身状态、已发命令和实际执行确认。用上次传输时长、最近四次均值、确认的新鲜度与命令差分表征链路。首次接管之前、确认缺失、确认超过 60 ms 或最近确认传输时长为零时输出原目标。训练抖动每五拍重抽 0–30 ms，留出的变延迟条件用不同相位的九拍周期分段波形，两者均含短暂确认丢失。各命令以序号递增方式执行，同一微步只应用最新到达的有效目标。",
        "",
        "每个压力条件固定同一组 24 个倒姿与 10 次行走受推。格内为静态通过数 / 五秒续走通过数。起步渐变只运行于行走参考接口，静态列复用完全相同的控制路径。固定预测与延迟重练均使用上一阶段按 20 ms 选定的模型；在线性、固定预测和重练对照中按已知非零延迟启用。",
        "",
        "| 条件 | " + " | ".join(names[k] for k in res["stress"]["lag"]) + " |",
        "|---|" + "---:|" * len(res["stress"]["lag"]),
    ]
    for scene, title in [
        ("half", "10 ms"),
        ("lag", "20 ms"),
        ("long", "30 ms"),
        ("variable", "未见变延迟"),
        ("slick", "低摩擦 + 20 ms"),
        ("payload", "加 0.7 kg + 20 ms"),
        ("combined", "低摩擦、负载 + 20 ms"),
    ]:
        vals = res["stress"][scene]
        lines.append(
            "| "
            + title
            + " | "
            + " | ".join(f"{v['static']} / {v['counts']['pass5']}" for v in vals.values())
            + " |"
        )
    lines += [
        "",
        "## 复核与复现",
        "",
        f"归档的三秒基准覆盖 {parity['episodes']} 条历史轨迹、{parity['ticks']:,} 拍，位姿、参考、遥测、足载荷及通过判定相同。日常无推直线与八字各 701 拍，四种方案逐帧一致。日常零延迟旁路核对 {ver['no_delay_relay_ticks']:,} 拍，渐变前缀核对 {ver['pace_prefix_ticks']:,} 拍。按真实执行确认调用独立推理入口，变延迟接力 {ver['independent']['variable']:,} 拍、20 ms 静态 {ver['independent']['lag']:,} 拍与实验入口位姿和遥测一致。",
        "",
        f"留出集共 {res['physical_jobs']} 个完整仿真任务。发出时估算 PD 越限 {res['issued_oob']} 项，到件时限幅前需求越限 {res['delivered_demand_oob']:,} 项，物理分段末次采样实际力矩越限 {res['actual_force_oob']} 项。计数为关节采样，不能据此估计连续时间或真机风险。地面为仿真平地，载荷为骨盆集中质量。",
        "",
        "原始轨迹、选模记录、权重和 25 fps 原速对照视频位于 yu-3090 的 `/home/yu/projects/SuperMimicHP/runs/h003_gait_engineering_01/flow_fluency`。Git 保存源码、试次级记录、六宫图和带 SHA-256 的清单。独立入口为 age_branch.AgeAwareRecoveryBranch，在新控制周期前传入已发送序号和执行时间戳（5 ms 为单位，同一时钟域）。真实设备的确认语义与时钟条件尚待验证。",
        "",
        "在同一冻结实验父目录的新子目录放入本阶段脚本，保留父目录控制器、场景、轨迹及 delay_learning 中的原预测器和旧动作权重。然后运行",
        "",
        "```sh",
        'export Q1_V2B_BUNDLE="$(cd ../formal_03/pinned/q1_v2b_gantry_deploy_bundle && pwd)"',
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "py=../../../.venv/bin/python",
        'for stage in data pace; do "$py" jobs.py "$stage"; done',
        '"$py" assess.py parity',
        '"$py" fit.py',
        '"$py" jobs.py gains',
        '"$py" assess.py select',
        '"$py" jobs.py test',
        '"$py" assess.py final',
        '"$py" verify_flow.py',
        'MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json "$py" visuals.py',
        '"$py" evidence.py',
        '"$py" report.py',
        "```",
    ]
    i = lines.index("## 补训与迁移")
    flips = [
        y
        for split in ("hold0", "hold1")
        for x, y in zip(
            rows(split, "daily", "published", "relay"), rows(split, "daily", a, "relay")
        )
        if x["success"] and not y["success"]
    ]
    wide = [v for v in flips if v["foot_width_cm"] and v["foot_width_cm"] > 22]
    if wide:
        widths = [r["foot_width_cm"] for r in wide]
        lines[i:i] = [
            f"按原三秒指标，由通过转为未通过的 {len(flips)} 例中，{len(wide)} 例的前三秒平均步宽越过 22 cm，范围为 {min(widths):.2f}–{max(widths):.2f} cm。其中 {sum(v['pass5'] for v in wide)} 例通过五秒判定，五秒均值会掩盖部分起步时的宽步。",
            "",
        ]
    idx = lines.index("## 复核与复现")
    lines[idx:idx] = [
        "固定延迟与变延迟分开判断。开发集选出的单一模型和增益在全部压力条件共用；这种选择能保护零延迟旁路，正延迟条件的效果仍须按表逐项判断。",
        "",
    ]
    (D / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
