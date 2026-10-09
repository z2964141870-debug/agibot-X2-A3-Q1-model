"""Freeze this blocked full-chain stage using its actual artifacts and test receipts."""

from datetime import datetime, timezone
import json
import re

from script.a3.fullchain_support import DATA, LOGS, ROOT, digest, mark, write_json
from script.a3.host_health import command, sample


def main():
    transfer = json.loads((DATA / "official_download.json").read_text())
    if any(row["status"] != "deferred" for row in transfer):
        raise ValueError("Acquisition status changed; reassess remaining tasks before finalizing")
    job = json.loads((DATA / "finetune_R04/state.json").read_text())
    if job["attempts"] or job["reason"] != "official_pt_unavailable":
        raise ValueError("Trial status changed; reassess training results")
    parity = json.loads((DATA / "onnx_diagnostic_local2950_v2/parity.json").read_text())
    model = DATA / "onnx_diagnostic_local2950_v2/a3_fast.onnx"
    if not parity["allclose"] or digest(model) != parity["onnx_sha256"]:
        raise ValueError("Missing verified ONNX diagnostic")
    tests = (LOGS / "final_tests.log").read_text()
    match = re.search(r"Ran (\d+) tests", tests)
    if not match or not re.search(r"^OK$", tests, re.MULTILINE):
        raise ValueError("Tests did not complete successfully")
    converter_tests = (LOGS / "rknn_converter_tests.log").read_text()
    if "6 passed" not in converter_tests:
        raise ValueError("Converter contract tests did not pass")
    dependencies = json.loads((DATA / "dependencies.json").read_text())
    if dependencies["modules"]["rknn"] or dependencies["isaac_metrics_returncode"] == 0:
        raise ValueError("Dependency status changed; reassess deferred tasks")
    deferred = {
        "mujoco_baseline": ["official_pt", "Transfer official PT and matching configurations"],
        "isaac_baseline": ["official_pt_and_isaac_dependency", "Provide official PT and compatible smpl_sim metrics"],
        "paired_evaluation": ["official_baseline_and_finetune", "Complete official baseline and R04 before comparing"],
        "reference_buffer_policy": ["official_pt", "Run policy-level causal replay comparison after official PT arrives"],
        "onnx": ["official_pt_and_finetune", "Export and compare actual official/fine-tuned policy inputs"],
        "rknn": ["rknn_toolkit_and_official_onnx", "Supply compatible RKNN toolkit; board runtime is a separate future task"],
        "mocap_finetune": ["mocap_quality", "Resolve calibrated pose/root/contact quality before using clothing references for training"],
    }
    for name, (reason, user_action) in deferred.items():
        mark(name, "deferred", reason=reason, user_action=user_action)
    for name, path, scope in (
        ("mujoco_wrapper_check", "wrapper_abi_local2950/explicit_summary.json", "LOCAL2950_100_STEPS_NOT_OFFICIAL_BASELINE"),
        ("onnx_diagnostic", "onnx_diagnostic_local2950_v2/parity.json", "LOCAL2950_ACTUAL100_INPUT_PARITY_ONLY"),
    ):
        if not (DATA / path).is_file():
            raise ValueError("Missing result evidence")
        mark(name, "passed", evidence=str(DATA / path), scope=scope)
    mark("finetune_preflight", "passed", evidence=str(LOGS / "finetune_inputs_preflight.log"),
         scope="INPUT_ONLY_NO_OFFICIAL_WEIGHT_LOADING_OR_PPO")
    mark("mocap_quality", "failed", reason="ik_residual_and_foot_penetration",
         evidence=str(DATA / "retarget_summary.json"),
         user_action="Verify personal calibration, reconstruction, root trajectory, ground and foot contacts")
    ledger = json.loads((DATA / "tasks.json").read_text())
    unfinished = [key for key, row in ledger["tasks"].items() if row["status"] in ("pending", "running")]
    if unfinished:
        raise ValueError(f"Unresolved runnable task states: {unfinished}")
    service = command(["systemctl", "--user", "show", "yuanqi-a3-longtrain.service",
                       "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "UnitFileState"])
    gpu_processes = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    if service["returncode"] != 0 or "ActiveState=inactive" not in service["stdout"] or "MainPID=0" not in service["stdout"]:
        raise ValueError("Legacy training is not confirmed paused")
    if gpu_processes["returncode"] != 0 or gpu_processes["stdout"]:
        raise ValueError("GPU process state changed")
    host = {"health": sample(), "legacy_service": service, "gpu_processes": gpu_processes,
            "services": command(["systemctl", "--user", "list-units", "--all", "--no-pager", "yuanqi-a3-*"]),
            "q1_git": command(["git", "-C", "/media/yu/FAFF-E977/YuanQi_Q1", "status", "--short", "--branch"])}
    write_json(DATA / "final_host.json", host)
    records = []
    for directory in (DATA, LOGS):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix != ".lock":
                records.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                "sha256": digest(path), "backup_status": "LOCAL_ONLY"})
    for path in sorted((ROOT / "script/a3").glob("*.py")):
        records.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)})
    result = {"experiment": ledger["experiment"], "stage_status": "ALL_INDEPENDENT_TASKS_FINISHED_REMAINING_DEFERRED",
              "updated_utc": datetime.now(timezone.utc).isoformat(), "tasks": ledger["tasks"],
              "official_training_updates_this_stage": 0, "official_baseline_executed": False,
              "tests_passed": int(match.group(1)), "upstream_rknn_contract_tests_passed": 6,
              "backup_status": "LOCAL_ONLY", "hardware_control": False,
              "user_transfer": transfer, "final_host": host, "artifacts": records}
    destination = ROOT / "data/manifests/a3_fullchain_20261010.json"
    write_json(destination, result)
    print(json.dumps({"manifest": str(destination), "tasks": len(ledger["tasks"]),
                      "artifacts": len(records), "tests_passed": result["tests_passed"]}, indent=2))


if __name__ == "__main__":
    main()
