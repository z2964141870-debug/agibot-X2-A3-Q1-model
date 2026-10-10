"""One durable evaluation phase, resumed by systemd after host interruptions."""

import argparse
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

from script.a3.fullchain_support import DATA, LOGS, ROOT, VENDOR, write_json


STAGES = ("finetuned_evaluation", "official_mocap", "official_buffer_selected20",
          "official_buffer_mocap", "finetuned_partial_evaluation")


def run(stage):
    with (DATA / "evaluation_supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        return run_locked(stage)


def run_locked(stage):
    if float(Path("/proc/uptime").read_text().split()[0]) < 300:
        return 75
    training = json.loads((DATA / "finetune_R05/state.json").read_text())
    if training["status"] not in ("complete", "blocked"):
        return 75
    output = DATA / stage
    output.mkdir(parents=True, exist_ok=True)
    control_path = output / "evaluation_control.json"
    control = json.loads(control_path.read_text()) if control_path.exists() else {"stage": stage, "status": "pending"}
    if control["status"] == "blocked":
        return 2
    if training["status"] != "complete" and stage == "finetuned_evaluation":
        control.update(status="blocked", reason="final200_checkpoint_unavailable")
        write_json(control_path, control)
        return 2
    from script.a3.evaluation_common import preflight
    from script.a3.evaluate_mujoco import evaluate_explicit
    from script.a3.finetune_job import select_trial_checkpoint
    from script.a3.recover_evaluation import recover
    preflight()
    checkpoint = ROOT / "data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt"
    if stage.startswith("finetuned"):
        step, checkpoint = select_trial_checkpoint(training["attempts"], "R05")
        if checkpoint is None or (stage == "finetuned_evaluation" and step != 200):
            raise ValueError("Missing independently verified evaluation checkpoint")
        control["verified_step"] = step
    motion = VENDOR / "a3_data/agibot_a3"
    stride = 4
    if "mocap" in stage:
        motion = DATA / "mocap_policy_inputs"
        motion.mkdir(parents=True, exist_ok=True)
        for name in ("cloth_walk", "cloth_oneleg", "recording_0923", "recording_0924"):
            source = DATA / "mocap" / name / "reference.csv"
            target = motion / f"{name}.csv"
            if not target.exists():
                target.symlink_to(source)
            elif target.resolve() != source.resolve():
                raise ValueError("Motion alias changed")
        stride = 1
    manifest = output / "explicit_manifest.json"
    if manifest.exists():
        recover(output)
    control.update(status="running", boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                   checkpoint=str(checkpoint), old_training_restart=False)
    write_json(control_path, control)
    args = SimpleNamespace(output=output, logs=LOGS / stage, checkpoint=checkpoint,
        motion=motion, reference_fps=30, frame_stride=stride, max_policy_steps=None,
        action_delay_ms=0, capture_inputs=True, summarize_only=False,
        reference_buffer_ms=180 if "buffer" in stage else 0)
    code = evaluate_explicit(args)
    control.update(status="complete" if code == 0 else "blocked", returncode=code)
    write_json(control_path, control)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    raise SystemExit(run(parser.parse_args().stage))
