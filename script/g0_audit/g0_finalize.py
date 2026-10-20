import json, hashlib, time
from pathlib import Path
OPS = Path("/opt/data/private/yuanqi_ops")
def sha(p): return hashlib.sha256((OPS / p).read_bytes()).hexdigest()[:16]
reg_path = OPS / "data/experiments/a3_zhy_reachability_audit_20261019_G0/registration.json"
reg = json.loads(reg_path.read_text())
reg["code_sha256"] = {p: sha(p) for p in ["g0_runner.py", "g0_analysis.py"]}
reg["finalized_at"] = time.time()
reg["protocol_fidelity"] = "cloth6 mean full = 0.1413798 (E26 registration value reproduced exactly)"
reg_path.write_text(json.dumps(reg, indent=2) + chr(10))
print("registration finalized")
