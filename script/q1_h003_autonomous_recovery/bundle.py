"""Build and exercise the self-contained advisory runtime with an isolated module path."""

import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

D = Path(__file__).resolve().parent
P = D.parent
B = Path(os.environ["Q1_V2B_BUNDLE"])


def digest(b):
    return hashlib.sha256(b).hexdigest()


def build():
    cfg = json.loads((B / "config/q1_deploy_config.json").read_text())
    files = {
        name: D / name for name in ("q1_autonomous_recovery.py", "check_ownership.py", "USAGE.md")
    }
    files.update(
        {name: P / name for name in ("q1_recovery_runtime.py", "leg_height.py", "recovery.py")}
    )
    files.update(
        {name: P / "dependencies" / name for name in ("q1_cgfa_obs_np.py", "q1_kinematics_np.py")}
    )
    for name in (
        "config/q1_deploy_config.json",
        "robot_model/supermimic.xml",
        "onnx/" + cfg["onnx_head_file"],
        "onnx/" + cfg["modal_tokenizer_files"]["robot_pose"],
    ):
        files[name] = B / name
    files["q1_v2b_runtime.py"] = B / "code/q1_v2b_runtime.py"
    files["onnx/q1_recovery_expert.onnx"] = P / "formal_03/q1_recovery_expert.onnx"
    path = D / "q1_autonomous_recovery.zip"
    hashes = {n: digest(p.read_bytes()) for n, p in files.items()}
    assert (
        hashes["onnx/q1_recovery_expert.onnx"]
        == "0231328d60f07297b38a4079e01167cf9fec65fdb2c89302144ee65e0011eb92"
    )
    with ZipFile(path, "w", ZIP_DEFLATED, compresslevel=5) as z:
        for n, p in sorted(files.items()):
            z.write(p, n)
        z.writestr("SHA256.json", json.dumps(hashes, indent=2))
    with tempfile.TemporaryDirectory() as tmp, ZipFile(path) as z:
        assert z.testzip() is None
        z.extractall(tmp)
        for n, v in hashes.items():
            assert digest((Path(tmp) / n).read_bytes()) == v
        code = """import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from q1_v2b_runtime import Q1V2bOnnxController
from check_ownership import check
p=Path.cwd()
print(json.dumps(check(Q1V2bOnnxController(p,("robot_pose",)),p/"onnx/q1_recovery_expert.onnx",p/"robot_model/supermimic.xml")))"""
        result = json.loads(
            subprocess.check_output([sys.executable, "-I", "-c", code], cwd=tmp, text=True)
        )
    (D / "bundle.json").write_text(
        json.dumps(
            dict(
                sha256=digest(path.read_bytes()),
                bytes=path.stat().st_size,
                components=hashes,
                isolated_checks=result,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    build()
