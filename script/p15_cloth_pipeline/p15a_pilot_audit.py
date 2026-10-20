"""P1.5a per-scene audit. argv: [scene_dir]. Writes audit_metrics.json + abnormal_frames.csv."""
import csv, json, sys
import numpy as np
from pathlib import Path

OUT = Path(sys.argv[1])
BODY_NAMES = ["pelvis", "spine3", "left_hip", "right_hip", "left_knee", "right_knee",
              "left_ankle", "right_ankle", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
with np.load(OUT / "kinematic_trace.npz") as d:
    q = d["qpos"]; jt = [str(x) for x in d["joint_names"].tolist()]
    err = d["position_error_m"]; fz = d["foot_geometry_min_z_m"]
summary = json.loads((OUT / "retarget_summary.json").read_text())
fps = summary.get("fps", 30)
j = q[:, 7:]
vel = np.diff(j, axis=0, prepend=j[:1]) * fps
vmax_joint = np.abs(vel).max(1)
frame_perr = err.max(1)
arm_idx = [BODY_NAMES.index(n) for n in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")]
leg_idx = [BODY_NAMES.index(n) for n in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")]
abnormal = []
for i in range(len(q)):
    reasons = []
    if vmax_joint[i] > 5.0:
        worst = int(np.argmax(np.abs(vel[i])))
        reasons.append("vel_%.1f_%s" % (vmax_joint[i], jt[7 + worst] if 7 + worst < len(jt) else "qpos_idx_%d" % (7 + worst)))
    if frame_perr[i] > 0.10:
        reasons.append("pos_err_%.3f" % frame_perr[i])
    if fz[i].min() < -0.05:
        reasons.append("foot_pen_%.3f" % fz[i].min())
    if reasons:
        abnormal.append((i, ";".join(reasons)))
metrics = {
    "frames": int(len(q)),
    "reachability": {"joint_limit_violations": summary.get("joint_limit_violations"), "finite": summary.get("finite")},
    "tracking": {"ik_position_p95_m": summary.get("ik_position_p95_m"),
                  "ik_position_max_m": summary.get("ik_position_max_m"),
                  "ik_orientation_p95_rad": summary.get("ik_orientation_p95_rad"),
                  "arm_pos_p95_m": round(float(np.percentile(err[:, arm_idx], 95)), 4),
                  "leg_pos_p95_m": round(float(np.percentile(err[:, leg_idx], 95)), 4),
                  "per_body_pos_p95_m": {BODY_NAMES[b]: round(float(np.percentile(err[:, b], 95)), 4)
                                          for b in np.argsort(-np.percentile(err, 95, axis=0))[:6]}},
    "smoothness": {"joint_velocity_rms_rad_s": round(float(np.sqrt((vel ** 2).mean())), 3),
                    "joint_velocity_p99_rad_s": round(float(np.percentile(np.abs(vel), 99)), 2),
                    "joint_velocity_max_rad_s": round(float(vmax_joint.max()), 2),
                    "frames_over_5_rad_s": int((vmax_joint > 5.0).sum())},
    "foot_contact": {"penetration_frames": int((fz.max(1) < 0).sum()),
                      "min_z_median_m": round(float(np.median(fz.min(1))), 4),
                      "min_z_worst_m": round(float(fz.min()), 4)},
    "abnormal_frames": {"count": len(abnormal), "ratio": round(len(abnormal) / len(q), 4)},
}
(OUT / "audit_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
with open(OUT / "abnormal_frames.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "reasons"])
    w.writerows(abnormal)
print(json.dumps(metrics, ensure_ascii=False))
