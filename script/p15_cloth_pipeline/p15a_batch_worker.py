"""P1.5a batch worker. argv: [shard_file] [out_base] [log_path].
Each scene runs in its own a3-sonic subprocess (isolation; vendor raise writes artifacts
before raising — recorded as-is per guardrail 2). Writes <scene>/status.json per scene."""
import json, subprocess, sys, time
from pathlib import Path
A3PY = "/opt/data/private/hechengzhu/ZHY/data/environments/a3-sonic/bin/python"
OPS = Path("/opt/data/private/yuanqi_ops")
PILOT = OPS / "p15a_retarget_pilot.py"
AUDIT = OPS / "p15a_pilot_audit.py"
shard, out_base, log_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
sessions = [l.strip() for l in open(shard) if l.strip()]
log = open(log_path, "a", buffering=1)
def say(msg):
    log.write(msg + "\n")
for n, session in enumerate(sessions):
    t0 = time.time()
    scene_dir = out_base / session
    scene_dir.mkdir(parents=True, exist_ok=True)
    entry = {"session": session, "rc": None, "seconds": None, "status": None, "error": None}
    try:
        p1 = subprocess.run([A3PY, str(PILOT), session, "0", str(out_base)],
                            capture_output=True, text=True, timeout=3600)
        entry["rc"] = p1.returncode
        entry["error"] = (p1.stderr or "")[-500:] if p1.returncode != 0 else None
        if (scene_dir / "retarget_summary.json").exists():
            p2 = subprocess.run([A3PY, str(AUDIT), str(scene_dir)],
                                capture_output=True, text=True, timeout=600)
        else:
            entry["error"] = (entry["error"] or "") + " | no retarget_summary.json"
    except subprocess.TimeoutExpired:
        entry["rc"] = -9
        entry["error"] = "timeout"
    except Exception as e:
        entry["rc"] = -1
        entry["error"] = type(e).__name__ + ": " + str(e)[:300]
    entry["seconds"] = round(time.time() - t0, 1)
    try:
        summary = json.loads((scene_dir / "retarget_summary.json").read_text())
        entry["status"] = "ok_with_exception" if entry["rc"] != 0 else "ok"
        entry["summary"] = {k: summary.get(k) for k in ("frames", "duration_s", "joint_limit_violations",
            "ik_position_p95_m", "ik_position_max_m", "ik_orientation_p95_rad",
            "joint_velocity_max_rad_s", "foot_geometry_penetration_frames")}
    except Exception:
        entry["status"] = "failed"
    (scene_dir / "status.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1))
    say(json.dumps({"done": n + 1, "of": len(sessions), "session": session,
                    "status": entry["status"], "seconds": entry["seconds"],
                    "pos_p95": entry.get("summary", {}).get("ik_position_p95_m")}))
say("SHARD_COMPLETE " + shard)
