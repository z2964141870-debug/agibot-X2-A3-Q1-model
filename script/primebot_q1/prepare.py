"""Convert existing paired V3 references and apply Primebot's original screen."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from common import FIXED, ROOT, identity, require, save


def read_yaml(path, python):
    # Use the existing YAML parser without changing the locked training environment.
    return json.loads(subprocess.check_output([str(python), "-c",
        "import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))",
        str(path)], text=True))


def convert(config, output):
    import mujoco
    import numpy as np
    import torch
    from supermimic.robot import Q1Robot

    output.mkdir(parents=True, exist_ok=False)
    q1 = read_yaml(config["q1_subset"], config["morph_python"])
    g1 = read_yaml(config["g1_subset"], config["morph_python"])
    expected = config.get("expected_candidates", 559)
    require(q1["clips"] == g1["clips"] and len(q1["clips"]) == expected,
            "The real paired candidate selection changed")
    robot = Q1Robot(Path(config["xml"]), reward_scaling="q1_proportional",
                    modeled_action_delay=True)
    model, data = robot.model, mujoco.MjData(robot.model)
    names = list(robot.joint_names)
    records = []
    for relative in q1["clips"]:
        source = Path(q1["root"]) / relative
        before = identity(source)
        with np.load(source, allow_pickle=False) as archive:
            repair_status = str(archive["reference_repair_status"].item()) if "reference_repair_status" in archive else None
            joints = [str(n) for n in archive["joint_names"]]
            bodies = [str(n) for n in archive["body_names"]]
            root = bodies.index("pelvis")
            require(float(archive["fps"]) == 50 and set(joints) == set(names),
                    "Source time base or native joint map changed")
            qpos = np.concatenate((archive["body_pos_w"][:, root],
                archive["body_quat_w"][:, root],
                archive["joint_pos"][:, [joints.index(n) for n in names]]), axis=1)
            require(qpos.shape[1] == 29 and np.isfinite(qpos).all()
                    and np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)) < 1e-4,
                    "Invalid physical pelvis/joint reference")
            errors = []
            for frame in sorted({0, len(qpos) // 2, len(qpos) - 1}):
                data.qpos[:] = qpos[frame]
                mujoco.mj_kinematics(model, data)
                actual = np.stack([data.xpos[model.body(n).id] for n in bodies])
                errors.append(float(np.max(np.abs(actual - archive["body_pos_w"][frame]))))
            require(max(errors) <= 2e-6, "Real V3 FK disagrees with stored body poses: " + relative)
        destination = output / "candidates" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, qpos=qpos.astype(np.float32),
            robot_joint_names=np.asarray(names), robot_name="q1_v3", root_quat_order="wxyz",
            fps=50., motion_key=Path(relative).stem, source_file=str(source),
            source_qpos_sha256=before["sha256"], import_method="existing_SOMA_pelvis_native22",
            resample_method="none_already_50Hz")
        require(identity(source) == before, "Original source changed during conversion")
        records.append(dict(relative=relative, source=before, converted=identity(destination),
                            frames=len(qpos), sampled_fk_max_error_m=max(errors), repair_status=repair_status))
        if len(records) % 50 == 0:
            print(json.dumps(dict(converted=len(records), total=expected)), flush=True)
    require(not torch.cuda.is_initialized(), "CPU conversion initialized CUDA")
    save(output / "conversion.json", dict(status="COMPLETED", records=records,
         asset_sha256=robot.asset_sha256, xml=identity(Path(config["xml"])),
         native_joint_names=names, cuda_initialized=False, new_optimizer_updates=0,
         source_subset=identity(Path(config["q1_subset"]))))
    return q1, g1, records


def prepare(config, output):
    from supermimic.filter_motions import FilterConfig, run_screen
    q1, g1, records = convert(config, output)
    # The exact upstream defaults decide accepted/review/rejected. No automatic
    # promotion of review clips, relabeling or threshold tuning is performed.
    result = run_screen(Path(config["xml"]), output / "candidates", output / "screen",
                        FilterConfig(), workers=4)
    rows = [json.loads(line) for line in (output / "screen/results.jsonl").read_text().splitlines()]
    expected = config.get("expected_candidates", 559)
    require(len(rows) == expected and result["phase"] == "complete"
            and result["completed_files"] == result["total_files"] == expected,
            "Upstream screening did not cover every candidate")
    repair_valid = {r["relative"] for r in records if r["repair_status"] in (None, "projected")}
    accepted = sorted(row["path"] for row in rows if row["status"] == "accepted" and row["path"] in repair_valid)
    save(output / "screening_outcome.json", dict(status="ADMISSIBLE" if accepted else "NO_ADMISSIBLE_DATA",
        candidates=expected, accepted=len(accepted), original_screen_counts=result["counts"],
        gate_reason=[] if accepted else ["no_motions_pass_original_screen_and_repair_audit"],
        original_thresholds_unchanged=True, formal_training_started=False,
        excluded_by_repair_audit=sorted(set(q1["clips"]) - repair_valid)))
    require(bool(accepted), "Original screening accepted no motions; preserve the negative result")
    for relative in accepted:
        src, dst = output / "candidates" / relative, output / "accepted" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
    for robot, original in (("q1", q1), ("g1", g1)):
        original["clips"] = accepted
        original["hours"] = sum(r["frames"] for r in records if r["relative"] in accepted) / 50 / 3600
        save(output / (robot + "_paired.yaml"), original)
    by_name = {Path(r["relative"]).stem: r for r in records}
    fixed = [dict(clip=name, relative=by_name[name]["relative"],
                  candidate=by_name[name]["converted"], in_training=by_name[name]["relative"] in accepted)
             for name in FIXED]
    record = dict(status="COMPLETED", candidates=expected, accepted=len(accepted),
                  rejected=sum(r["status"] == "rejected" for r in rows),
                  review=sum(r["status"] == "review" for r in rows),
                  accepted_relative_paths=accepted, fixed=fixed,
                  original_screening_thresholds=True, candidate_manifest=identity(output / "conversion.json"),
                  screening_results=identity(output / "screen/results.jsonl"),
                  optimizer_updates=0, storage="SERVER_LOCAL_ONLY")
    save(output / "result.json", record)
    print(json.dumps({k: record[k] for k in ("status", "candidates", "accepted", "rejected", "review")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.output.resolve())
