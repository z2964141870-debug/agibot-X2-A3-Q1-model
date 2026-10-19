import json, hashlib, numpy as np
from pathlib import Path

ROOT = Path("/opt/data/private/hechengzhu/ZHY")
OPS = Path("/opt/data/private/yuanqi_ops")
E26 = ROOT / "data/experiments/a3_zhy_smpl_frozen_20261013_E26"
EV = ROOT / "data/experiments/a3_zhy_reachability_audit_20261019_G0"
OUT = OPS / "data/experiments/a3_zhy_reachability_audit_20261019_G0" / "tables"
FPS = 50.0
WIN = int(2 * FPS)
SEEDS = [0, 1, 2]
KINDS = ("cloth", "official")

def load(seed, kind, name):
    d = EV / ("seed%d" % seed) / kind
    met = json.loads((d / (name + ".metrics.json")).read_text())["motions"][0]
    ts = json.loads((d / (name + ".timeseries.json")).read_text())["motions"][0]
    return met, ts

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def body_group(key_names):
    torso = [i for i, n in enumerate(key_names) if ("torso" in n or "pelvis" in n)]
    wrists = [i for i, n in enumerate(key_names) if "wrist" in n]
    legs = [i for i, n in enumerate(key_names) if ("ankle" in n or "knee" in n or "foot" in n or "hip" in n)]
    return torso, wrists, legs

def world_means(ts, lo, hi):
    ref = np.asarray(ts["ref_key_body_pos_w"], dtype=float)[lo:hi]
    sim = np.asarray(ts["sim_key_body_pos_w"], dtype=float)[lo:hi]
    err = np.linalg.norm(sim - ref, axis=-1)
    names = ts["key_body_names"]
    torso, wrists, legs = body_group(names)
    return float(err[:, torso].mean()), float(err[:, wrists].mean()), float(err[:, legs].mean())

def analyze_motion(seed, kind, name):
    met, ts = load(seed, kind, name)
    q = np.asarray(ts["q_state_29"], dtype=float)
    qr = np.asarray(ts["reference_q_29"], dtype=float)
    n = q.shape[0]
    jerr = q - qr
    height = np.asarray(ts["root_height"], dtype=float)
    roll = np.abs(np.asarray(ts["root_roll_deg"], dtype=float))
    pitch = np.abs(np.asarray(ts["root_pitch_deg"], dtype=float))
    fallmask = (height < 0.45) | (roll > 60.0) | (pitch > 60.0)
    ticks = np.flatnonzero(fallmask)
    fall = bool(len(ticks))
    first = int(ticks[0]) if fall else None
    full_rmse = float(np.sqrt((jerr ** 2).mean()))
    bff_hi = first if fall else n
    bff_rmse = float(np.sqrt((jerr[:bff_hi] ** 2).mean()))
    per_joint_full = np.sqrt((jerr ** 2).mean(axis=0))
    per_joint_bff = np.sqrt((jerr[:bff_hi] ** 2).mean(axis=0))
    torso_f, wrists_f, legs_f = world_means(ts, 0, n)
    torso_b, wrists_b, legs_b = world_means(ts, 0, bff_hi)
    wdr_full = max(wrists_f, legs_f) / torso_f
    wdr_bff = max(wrists_b, legs_b) / torso_b
    root_pe = float(np.asarray(ts["root_pos_error_m"], dtype=float).mean())
    row = dict(seed=seed, kind=kind, motion=name, steps=n, fall=fall, first_fall_step=first,
               first_fall_s=(first / FPS if fall else None),
               metrics_match=(met["fall"] == fall and abs(met["fall_time_s"] - first / FPS) < 1e-9 if fall else met["fall"] == fall),
               cloth6_full=full_rmse, bff=bff_rmse,
               guard_max_full=float(per_joint_full.max()), guard_max_bff=float(per_joint_bff.max()),
               wdr_full=wdr_full, wdr_bff=wdr_bff, root_pos_err_m_mean=root_pe,
               metrics_sha256=sha(EV / ("seed%d" % seed) / kind / (name + ".metrics.json")))
    win = dict(lo=max(0, (first or 0) - WIN), hi=min(n, (first or n - 1) + WIN + 1))
    if fall:
        wj = np.sqrt((jerr[win["lo"]:win["hi"]] ** 2).mean(axis=0))
        order = np.argsort(wj)[::-1][:6]
        jnames = ts["body_names"]
        refv = np.abs(np.diff(qr[win["lo"]:win["hi"]], axis=0) * FPS).mean() if win["hi"] - win["lo"] > 1 else 0.0
        row["window"] = dict(lo=win["lo"], hi=win["hi"], lo_s=win["lo"] / FPS, hi_s=win["hi"] / FPS,
                             top_joints=[dict(joint=jnames[i], rmse_rad=float(wj[i])) for i in order],
                             ref_joint_speed_rms=float(refv))
        try:
            c = np.load(E26 / "evaluation" / (name + ".contact.npz"), allow_pickle=True)
            cc = c["foot_floor_contact_count"][win["lo"]:win["hi"]]
            row["window"]["support_L_frac"] = float((cc[:, 0] > 0).mean())
            row["window"]["support_R_frac"] = float((cc[:, 1] > 0).mean())
        except Exception:
            pass
    row["per_joint_full"] = [float(x) for x in per_joint_full]
    row["per_joint_bff"] = [float(x) for x in per_joint_bff]
    row["joint_names"] = list(ts["body_names"])
    return row

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = {}
    for kind in KINDS:
        d = EV / "seed0" / kind
        names[kind] = sorted(p.stem.replace(".metrics", "") for p in d.glob("*.metrics.json"))
    rows = []
    for kind in KINDS:
        for name in names[kind]:
            for seed in SEEDS:
                rows.append(analyze_motion(seed, kind, name))
                print("analyzed", kind, name, "seed", seed, "fall", rows[-1]["fall"], flush=True)
    # determinism + noise floor
    determinism = {}
    for kind in KINDS:
        for name in names[kind]:
            shas = {r["metrics_sha256"] for r in rows if r["kind"] == kind and r["motion"] == name}
            determinism[kind + "/" + name] = ("identical" if len(shas) == 1 else "differing")
    def spread(key, sel):
        vals = [r[key] for r in rows if sel(r)]
        return dict(values=vals, mean=float(np.mean(vals)), std=float(np.std(vals)),
                    std_pct_of_mean=float(np.std(vals) / abs(np.mean(vals))) if np.mean(vals) else None)
    noise = dict(determinism=determinism, metrics={})
    for kind in KINDS:
        selk = lambda r, k=kind: r["kind"] == k
        if kind == "cloth":
            by_seed = []
            for seed in SEEDS:
                vs = [r["cloth6_full"] for r in rows if r["kind"] == kind and r["seed"] == seed]
                by_seed.append(float(np.mean(vs)))
            noise["metrics"]["cloth6_rmse_full"] = dict(per_seed_mean=by_seed,
                std_across_seeds=float(np.std(by_seed)), three_sigma=float(3 * np.std(by_seed)))
        per_motion = {}
        for name in names[kind]:
            per_motion[name] = dict(
                joint_rmse_full=spread("cloth6_full" if kind == "cloth" else "cloth6_full",
                                       lambda r, k=kind, n=name: r["kind"] == k and r["motion"] == n),
                wdr_full=spread("wdr_full", lambda r, k=kind, n=name: r["kind"] == k and r["motion"] == n),
                wdr_bff=spread("wdr_bff", lambda r, k=kind, n=name: r["kind"] == k and r["motion"] == n),
                guard_max_full=spread("guard_max_full", lambda r, k=kind, n=name: r["kind"] == k and r["motion"] == n),
                guard_max_bff=spread("guard_max_bff", lambda r, k=kind, n=name: r["kind"] == k and r["motion"] == n))
        noise["metrics"][kind] = per_motion
    # termination distribution + infeasible rule
    term = {}
    for kind in KINDS:
        for name in names[kind]:
            rs = [r for r in rows if r["kind"] == kind and r["motion"] == name]
            falls = [r for r in rs if r["fall"]]
            clusters = []
            for r in sorted(falls, key=lambda x: x["first_fall_s"]):
                if clusters and abs(r["first_fall_s"] - clusters[-1][-1]["first_fall_s"]) <= 2.0:
                    clusters[-1].append(r)
                else:
                    clusters.append([r])
            suspects = [c for c in clusters if len(c) >= 2]
            term[kind + "/" + name] = dict(
                first_fall_s=[r["first_fall_s"] for r in rs],
                fall_seeds=[r["seed"] for r in falls],
                clusters=[dict(seeds=[x["seed"] for x in c], window_s=[c[0]["first_fall_s"] - 2.0, c[-1]["first_fall_s"] + 2.0],
                               top_joints=c[0]["window"]["top_joints"] if c and "window" in c[0] else None,
                               ref_joint_speed_rms=c[0]["window"].get("ref_joint_speed_rms") if c and "window" in c[0] else None) for c in clusters],
                infeasible_suspect=bool(suspects))
    # per-joint baseline table (mean over seeds; full + bff)
    per_joint = {}
    for kind in KINDS:
        for name in names[kind]:
            rs = [r for r in rows if r["kind"] == kind and r["motion"] == name]
            per_joint[kind + "/" + name] = dict(
                joint_names=rs[0]["joint_names"],
                rmse_full_mean_over_seeds=[float(np.mean([r["per_joint_full"][i] for r in rs])) for i in range(29)],
                rmse_bff_mean_over_seeds=[float(np.mean([r["per_joint_bff"][i] for r in rs])) for i in range(29)])
    zzx = term.get("cloth/G_20251229_zzx__zzx_111505_walk")
    atomic = lambda p, o: Path(p).write_text(json.dumps(o, indent=2, default=str) + chr(10))
    atomic(OUT / "noise_floor.json", noise)
    atomic(OUT / "termination_distribution.json", term)
    atomic(OUT / "per_joint_baseline.json", per_joint)
    atomic(OUT / "rollout_rows.json", rows)
    verdict = dict(
        zzx=zzx,
        rule="infeasible_suspect = >=2 seeds (or cross-motion same segment) fall within same +-2s window",
        zzx_answer=("infeasible_suspect" if zzx and zzx["infeasible_suspect"] else
                    "not_infeasible_suspect (single-seed or non-clustered termination; E26 5.66s is a policy-side fall, not a reference infeasibility verdict)" if zzx else "missing"),
        disposition_recommendation="pending lead review",
    )
    atomic(OUT / "g0_verdict.json", verdict)
    print("G0 ANALYSIS DONE", flush=True)
    print(json.dumps(verdict["zzx_answer"]))

if __name__ == "__main__":
    main()
