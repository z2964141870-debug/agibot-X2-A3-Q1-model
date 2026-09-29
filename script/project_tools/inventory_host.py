#!/usr/bin/env python3
"""Print a read-only host inventory without opening credentials or sessions."""

import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def command(args, cwd=None):
    try:
        result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": type(error).__name__}


def main():
    project = Path(__file__).resolve().parents[2]
    disk = Path("/media/yu/FAFF-E9771")
    home = Path.home()
    usage = shutil.disk_usage(disk) if disk.is_dir() else None
    executable = Path("/opt/baidunetdisk/baidunetdisk")
    cli_names = ["BaiduPCS-Go", "baidupcs-go", "BaiduPCS", "bypy", "rclone"]
    process_result = command(["ps", "-eo", "comm="])
    netdisk_processes = sorted({
        name.strip() for name in process_result.get("stdout", "").splitlines()
        if any(term in name.lower() for term in ("baidu", "netdisk", "bypy", "rclone"))
    })
    payload = {
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "project_root": str(project),
        "expected_training_root": "/media/yu/FAFF-E9771/YUANQI",
        "disk": {
            "path": str(disk), "exists": disk.is_dir(),
            "bytes": dict(usage._asdict()) if usage else None,
            "mount": command(["findmnt", "-n", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS", "--target", str(disk)]) if disk.is_dir() else None,
        },
        "gpu": command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "git": {"version": command(["git", "--version"]), "status": command(["git", "status", "--short", "--branch"], cwd=project)},
        "baidu_netdisk": {
            "executable": str(executable),
            "executable_present": executable.is_file() and os.access(executable, os.X_OK),
            "package": command(["dpkg-query", "-W", "-f=${Version}", "baidunetdisk"]),
            "desktop_entry_present": Path("/usr/share/applications/baidunetdisk.desktop").is_file(),
            "user_config_directory_present": (home / ".config/baidunetdisk").is_dir(),
            "running_process_names": netdisk_processes,
            "cli_paths": {name: shutil.which(name) for name in cli_names},
            "login_status": "NOT_VERIFIED",
            "upload_download_status": "NOT_VERIFIED",
            "credential_files_read": False,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
