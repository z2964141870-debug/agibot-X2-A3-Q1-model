"""Wait for the existing validation, check saved-trace media, then train once."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "primebot_q1"))
from common import ROOT, identity, now, require, save
from validate import phase
from run_formal import run


def process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def execute(config_path, validation, smoke, output, commit, validation_pid):
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="WAITING_FOR_EXISTING_VALIDATION", started_utc=now(),
                  code_commit=commit, validation_pid=validation_pid,
                  validation_start_tick=process_start(validation_pid),
                  automatic_retry=False, user_accepted_checkpoint=None,
                  controller_pid=os.getpid(), controller_start_tick=process_start(os.getpid()))
    save(output / "pipeline.json", record)
    try:
        started = time.monotonic()
        while True:
            state = json.loads((validation / "pipeline.json").read_text())
            if state["status"] == "VALIDATED_READY_FOR_FRESH_FORMAL":
                break
            require(state["status"] == "RUNNING", "Existing validation failed; no dependent launch")
            require(record["validation_start_tick"] is not None and
                    process_start(validation_pid) == record["validation_start_tick"],
                    "Validation controller disappeared; preserve its evidence")
            require(time.monotonic() - started < 3 * 3600, "Bounded validation wait elapsed")
            time.sleep(10)
        record.update(status="CHECKING_SAVED_TRACE_MEDIA", validation=identity(validation / "pipeline.json"))
        save(output / "pipeline.json", record)
        graphics = Path(config["graphics_dir"])
        os.environ.update(Q1_REVIEW_GRAPHICS=str(graphics), LD_LIBRARY_PATH=str(graphics / "lib"),
            __EGL_VENDOR_LIBRARY_FILENAMES=str(HERE.parent / "morphmimic_cross_robot/nvidia_egl_595_20261102.json"),
            PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless")
        if config.get("ffmpeg"):
            os.environ["Q1_REVIEW_FFMPEG"] = config["ffmpeg"]
        inputs = [*HERE.glob("*.py"), smoke, config_path,
                  HERE.parent / "morphmimic_cross_robot/nvidia_egl_595_20261102.json"]
        if config.get("ffmpeg"):
            inputs.append(Path(config["ffmpeg"]))
        require(phase(config, config_path, output, "render_saved", [config["primebot_python"], "-u",
            str(HERE / "render_saved.py"), "--review", str(smoke), "--xml", config["xml"],
            "--output", str(output / "render_saved/media")], seconds=1800, inputs=inputs),
            "Saved-trace video validation failed; do not start the dependent run")
        rendered = json.loads((output / "render_saved/media/result.json").read_text())
        require(rendered["status"] == "PASSED" and len(rendered["videos"]) == 3,
                "Three original-speed fixed videos must render and decode")
        record.update(status="FORMAL_PIPELINE_RUNNING", render=identity(output / "render_saved/media/result.json"))
        save(output / "pipeline.json", record)
        run(config, config_path, validation, output / "formal", commit)
        record.update(status="FORMAL_SERVER_PIPELINE_FINISHED_PENDING_LOCAL_REVIEW",
                      formal=identity(output / "formal/pipeline.json"), ended_utc=now())
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        save(output / "pipeline.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "validation", "smoke", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--validation-pid", type=int, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    require(len(args.code_commit) == 40 and all(c in "0123456789abcdef" for c in args.code_commit),
            "Use an independently verified published commit")
    execute(args.config.resolve(), args.validation.resolve(), args.smoke.resolve(),
            args.output.resolve(), args.code_commit, args.validation_pid)
