#!/usr/bin/env python3
# p16_pool.py — build fgp_fitted pool (m2-schema isomorphic) + G0 audit + summary
# Provenance: poses=FGP(SEQ+shift3, wire FEED drop_u4); root_trans borrowed from GT annotation;
#             joints 24-31 zero-padded (FGP 24-joint chain); 151751/155435 placeholder-metadata noted.
import sys, os, json, glob, hashlib
import numpy as np, h5py, joblib, torch
sys.path.insert(0, "/opt/data/private/yuanqi_ops/script/tools/FGP_Full/FGP-main")
sys.path.insert(0, "/opt/data/private/yuanqi_ops/script/tools/FGP_Full/FGP-main/Aplus/tools")
from scipy.spatial.transform import Rotation as R
from smpl_light import SMPLight

PRE = "/opt/data/private/yuanqi_ops/data/humanplus_preview"
POOL = "/opt/data/private/yuanqi_ops/data/training/p2_main/fgp_fitted_motionlib"
EXP = "/opt/data/private/yuanqi_ops/data/experiments/p15b_fgp_pilot"
os.makedirs(POOL, exist_ok=True)
PILOT = "session_20260901_144355"
PLACEHOLDER_META = {"session_20260912_151751", "session_20260912_155435"}

sm = SMPLight()
scenes = sorted(os.path.basename(p) for p in glob.glob(os.path.join(PRE, "session_*")) if PILOT not in p)
rows, bad = [], []
for s in scenes:
    try:
        pf = os.path.join(PRE, s, "poses_fgp.npz"); af = os.path.join(PRE, s, "fgp_audit.json")
        aud = json.load(open(af)); T = int(aud["frames"])
        poses = np.load(pf)["poses"].astype(np.float32)
        assert poses.shape[0] == T, "frame mismatch %d!=%d" % (poses.shape[0], T)
        # root_trans from GT (borrowed, annotated)
        with h5py.File(os.path.join(PRE, s, "annotation.hdf5"), "r") as f:
            gt = f["body_motion/body_keypoints"][:T].astype(np.float32)
        root_trans = gt[:, 0] - gt[0, 0]
        # aa from rot mats: 24 body + 8 zero hands
        aa24 = R.from_matrix(poses.reshape(-1, 3, 3)).as_rotvec().reshape(T, 24, 3).astype(np.float32)
        pose_aa = np.concatenate([aa24, np.zeros((T, 8, 3), np.float32)], 1)
        root_rot = R.from_matrix(poses[:, 0]).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)  # xyzw->wxyz
        # FK for smpl_joints (root-relative), batch
        jf = np.zeros((T, 24, 3), np.float32)
        with torch.no_grad():
            for st in range(0, T, 4096):
                Rb = torch.from_numpy(poses[st:st+4096])
                _, j = sm.forward_kinematics(Rb, calc_joint=True)
                jf[st:st+len(j)] = j.numpy().astype(np.float32)
        jf = jf - jf[:, 0:1]
        payload = {s: {"root_trans_offset": root_trans.astype(np.float32),
                       "pose_aa": pose_aa, "dof": np.zeros((T, 29), np.float32),
                       "root_rot": root_rot, "smpl_joints": jf, "fps": 30}}
        pkl_path = os.path.join(POOL, s + ".pkl")
        joblib.dump(payload, pkl_path)
        sha = hashlib.sha256(open(pkl_path, "rb").read()).hexdigest()[:16]
        rows.append({"session": s, "pkl": pkl_path, "sha256": sha, "frames": T,
                     "mpjpe_mean_m": float(aud["mpjpe_mean_m"]), "scale": float(aud.get("scale", 0)),
                     "input_ble_sha16": aud.get("input_ble_sha16"),
                     "root_trans_source": "gt_annotation_borrowed", "hands_joints_24_31": "zero_padded",
                     "metadata_upstream": "placeholder_note" if s in PLACEHOLDER_META else "ok",
                     "wire_evidence": "experiments/p15b_fgp_pilot/WIRE_GRID.json"})
    except Exception as e:
        bad.append((s, "%s: %s" % (type(e).__name__, str(e)[:80])))

manifest = {"motionlib": POOL, "tier": "fgp_fitted", "provenance": {
    "recipe": "SEQ+shift3", "wire_feed": "drop_u4_l1st", "d2b": "identity",
    "mpjpe_mean_m": round(float(np.mean([r["mpjpe_mean_m"] for r in rows])), 4),
    "baseline_reference": 0.330, "wire_grid_evidence": EXP + "/WIRE_GRID.json",
    "notes": "root_trans borrowed from GT annotation; joints 24-31 zero-padded; see per-row fields"},
    "n": len(rows), "rows": rows}
mf = os.path.join(POOL, "fgp_fitted_manifest.json")
json.dump(manifest, open(mf, "w"), indent=1)

# ---- G0 audit: reload every pkl, verify schema/shape/NaN/frames/sha ----
g0 = {"pool": POOL, "n_expected": len(rows), "checked": 0, "fail": [], "total_frames": 0}
for r in rows:
    try:
        d = joblib.load(r["pkl"]); inner = d[r["session"]]
        assert hashlib.sha256(open(r["pkl"], "rb").read()).hexdigest()[:16] == r["sha256"]
        for k, exp_shape in [("root_trans_offset", (r["frames"], 3)), ("pose_aa", (r["frames"], 32, 3)),
                             ("dof", (r["frames"], 29)), ("root_rot", (r["frames"], 4)),
                             ("smpl_joints", (r["frames"], 24, 3))]:
            assert inner[k].shape == exp_shape, "%s %s" % (k, inner[k].shape)
            assert not np.isnan(inner[k]).any(), "%s NaN" % k
        assert int(inner["fps"]) == 30
        g0["checked"] += 1; g0["total_frames"] += r["frames"]
    except Exception as e:
        g0["fail"].append((r["session"], str(e)[:80]))
g0["pass"] = (g0["checked"] == g0["n_expected"] and not g0["fail"])
json.dump(g0, open(EXP + "/G0_fgp_fitted_report.json", "w"), indent=1, default=str)

# ---- full summary (directive 1) ----
mp = [r["mpjpe_mean_m"] for r in rows]
v = np.array(mp)
summary = {"tier": "fgp_fitted", "n_scenes": len(rows), "total_frames": g0["total_frames"],
           "mpjpe": {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
                     "p90": float(np.percentile(v, 90)), "max": float(v.max()), "min": float(v.min())},
           "redirect_triggered": 0, "redirect_records": [],
           "worst3": sorted([(r["session"], r["mpjpe_mean_m"]) for r in rows], key=lambda t: -t[1])[:3],
           "g0_pass": g0["pass"], "manifest": mf, "pool_build_fail": bad}
json.dump(summary, open(EXP + "/FGP_AUDIT_SUMMARY.json", "w"), indent=1)
print("rows=%d bad=%d g0_pass=%s frames=%d mean=%.4f" % (
    len(rows), len(bad), g0["pass"], g0["total_frames"], v.mean()))
print("worst3:", summary["worst3"])
