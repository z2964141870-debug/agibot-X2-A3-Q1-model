"""Run the four fixed v0.1 policy tests sequentially without training."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path("/opt/data/private/ZHY")
HERE = Path(__file__).resolve().parent
REFERENCE = ROOT / "data/references/morphmimic_v01_x2_t2_20261102"
SOURCE = REFERENCE / "unpack/morphmimic-0.1"
DEPLOY = REFERENCE / "release_deploy/morphmimic_policy"
PYTHON = ROOT / "data/environments/morphmimic_original_20261029/bin/python"
CLIPS = ("walk_forward_loop_001__A022", "walk_forward_start_002__A023")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    if not output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        parser.error("Use the isolated real-reference evaluation root")
    progress = output / "policy_suite_progress.json"
    if progress.exists():
        raise FileExistsError("Policy suites are single-use; preserve prior attempts")
    os.environ.update(PYTHONPATH=str(SOURCE / "src") + os.pathsep + str(SOURCE),
        GIT_PYTHON_GIT_EXECUTABLE=str(ROOT / "data/environments/morphmimic_git_tools/usr/bin/git"),
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
        LP_NUM_THREADS="4", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
        EGL_PLATFORM="surfaceless", LIBGL_ALWAYS_SOFTWARE="1",
        __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
        PYTHONUNBUFFERED="1")
    cpus = sorted(os.sched_getaffinity(0))[:12]
    os.sched_setaffinity(0, cpus)
    record = dict(status="RUNNING", started_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), code_commit=args.code_commit, cpu_affinity=cpus,
        renderer="existing Mesa software EGL, LP_NUM_THREADS=4",
        interface="motion", optimizer_updates=0, replays=[], commands=[])
    child = None

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
        raise KeyboardInterrupt("Policy suite stopped")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for robot in ("x2", "t2"):
            for clip in CLIPS:
                label = robot + "_" + clip
                summary = output / (label + ".json")
                video = output / (label + ".mp4")
                inputs = [DEPLOY / "policy.onnx", DEPLOY / (robot + ".json"),
                    output / ("assets_" + robot) / (robot + ".xml"),
                    output / ("assets_" + robot) / "clips" / robot / (clip + ".npz"),
                    SOURCE / "deploy/run.py", SOURCE / "deploy/sim2sim/sim_node.py",
                    SOURCE / "deploy/sim2sim/policy_node.py", SOURCE / "deploy/sim2sim/obs.py",
                    HERE / "run_guarded.py", Path(__file__).resolve()]
                command = [sys.executable, str(HERE / "run_guarded.py"),
                    "--output", str(output / ("policy_" + label)), "--source", str(SOURCE),
                    "--cwd", str(SOURCE), "--code-commit", args.code_commit,
                    "--wall-seconds", "3600"]
                for path in inputs:
                    command += ["--input", str(path)]
                command += ["--", str(PYTHON), "-u", "-m", "deploy.run",
                    "--deploy", str(DEPLOY), "--assets", str(output / ("assets_" + robot)),
                    "--robot", robot, "--clip", clip, "--interface", "motion",
                    "--summary", str(summary), "--video", str(video)]
                record.update(phase=label)
                record["commands"].append(command)
                save(progress, record)
                with (output / ("policy_" + label + "_controller.log")).open("x") as stream:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    code = child.wait()
                if code:
                    raise RuntimeError(label + " command failed; retain its logs")
                result = json.loads(summary.read_text())
                record["replays"].append(dict(robot=robot, clip=clip,
                    upstream_summary=result, video=str(video), summary=str(summary)))
                save(progress, record)
        record.update(status="COMPLETED", phase="completed")
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save(progress, record)


if __name__ == "__main__":
    main()
