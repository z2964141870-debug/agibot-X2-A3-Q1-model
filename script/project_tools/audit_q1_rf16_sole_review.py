"""Measure actual collision-sole clearance in completed RF16 saved fixed clips."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from audit_q1_saved_sole_geometry import foot_geometry, measure
from prepare_q1_gait_review import FIXED, identity, write_csv


def valid_poses(trace, index):
    import numpy as np

    steps = int(trace["steps"][index])
    reference = int(trace["reference_steps"][index])
    states = trace["states"]
    if (states.ndim != 3 or states.shape[-1] != 57 or not 1 < steps <= reference
            or steps > states.shape[0]):
        raise ValueError("Saved pre-reset duration or state shape is invalid")
    selected = states[:steps, index]
    if not np.isfinite(selected).all():
        raise ValueError("Non-finite saved physical pose")
    poses = np.concatenate((selected[:, :7], selected[:, 13:35]), -1)
    return poses, steps, reference


def audit(args):
    import mujoco
    import numpy as np

    if args.output.exists():
        raise FileExistsError(args.output)
    if json.loads((args.trial / "status.json").read_text())["status"] != "COMPLETED":
        raise ValueError("Read only completed RF16 evidence")
    plan = json.loads((args.trial / "plan.json").read_text())
    stages = json.loads((args.trial / "run/stage_decisions.json").read_text())
    labels = ["initial_static075"] + [f"stage_{i:02d}" for i in range(1, len(stages) + 1)]
    library_path = args.trial / "evaluation/evaluation_library.json"
    library = json.loads(library_path.read_text())["splits"]["train"]
    if len(library) != 330 or len({row["name"] for row in library}) != 330:
        raise ValueError("Expected original330 evaluation library")
    indices = {name: next(i for i, row in enumerate(library) if row["name"] == name) for name in FIXED}
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    geometry = [foot_geometry(model, side) for side in ("left", "right")]
    floor = model.geom("floor").id
    if (int(model.geom_type[floor]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
            or not np.allclose(model.geom_pos[floor], 0.)
            or not np.allclose(model.geom_quat[floor], [1., 0., 0., 0.])):
        raise ValueError("Expected the actual horizontal z0 floor")
    with args.reference_csv.open(newline="") as stream:
        references = [row for row in csv.DictReader(stream)
                      if row["case"] == "reference" and row["clip"] in FIXED]
    if len(references) != 6 or {(row["clip"], row["side"]) for row in references} != {
            (name, side) for name in FIXED for side in ("left", "right")}:
        raise ValueError("Missing fixed reference collision-sole measurements")
    reference_receipt_path = args.reference_csv.parent / "summary.json"
    reference_receipt = json.loads(reference_receipt_path.read_text())
    if identity(args.scene) not in reference_receipt["sources"]:
        raise ValueError("Reference sole geometry used a different scene")
    sources = [identity(path) for path in
               (args.scene, args.reference_csv, reference_receipt_path, library_path,
                args.trial / "plan.json", Path(__file__))]
    rows, curves = [], []
    for label in labels:
        for repeat in range(3):
            path = args.trial / "evaluation" / f"{label}_repeat{repeat}.npz"
            result = json.loads(path.with_suffix(".json").read_text())
            if result["trace"] != identity(path):
                raise ValueError("Saved physical trace identity changed: " + str(path))
            sources.extend((identity(path), identity(path.with_suffix(".json"))))
            with np.load(path, allow_pickle=False) as trace:
                for name, index in indices.items():
                    poses, steps, reference_steps = valid_poses(trace, index)
                    clip = result["per_clip"][index]
                    if (clip["name"] != name or clip["steps"] != steps
                            or clip["reference_steps"] != reference_steps):
                        raise ValueError("Clip/step alignment differs from its saved JSON")
                    if any(int(row["frames"]) != reference_steps
                           for row in references if row["clip"] == name):
                        raise ValueError("Original reference sole duration differs")
                    height, speed = measure(model, data, poses, geometry)
                    force = trace["force"][:steps, index]
                    if force.shape != (steps, 2) or not np.isfinite(force).all():
                        raise ValueError("Missing saved native force")
                    for foot, side in enumerate(("left", "right")):
                        above, loaded = height[:, foot] > .02, force[:, foot] > 5.
                        runs = int(np.count_nonzero(above & np.r_[True, ~above[:-1]]))
                        rows.append(dict(label=label, repeat=repeat, clip=name, side=side,
                            training_seconds=result["training_seconds"], steps=steps,
                            reference_steps=reference_steps, failed=bool(clip["failed"]),
                            sole_height_p05_m=float(np.quantile(height[:, foot], .05)),
                            sole_height_p50_m=float(np.quantile(height[:, foot], .50)),
                            sole_height_p95_m=float(np.quantile(height[:, foot], .95)),
                            lowest_sole_height_m=float(height[:, foot].min()),
                            above2cm_fraction=float(above.mean()), above2cm_runs=runs,
                            native_loaded_fraction=float(loaded.mean()),
                            loaded_material_speed_mps=float(speed[loaded, foot].mean()) if loaded.any() else None,
                            near_floor5mm_fraction=float((height[:, foot] <= .005).mean())))
                        curves.extend(dict(label=label, repeat=repeat, clip=name, side=side,
                                           seconds=(i+1)*.02, sole_height_m=float(h),
                                           lowest_material_speed_mps=float(v), native_force_N=float(f))
                                      for i, (h, v, f) in enumerate(zip(height[:, foot], speed[:, foot], force[:, foot])))
            print(json.dumps(dict(label=label, repeat=repeat, saved_pose_analysis_complete=True)), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "fixed_actual_sole.csv", rows)
    write_csv(args.output / "fixed_actual_sole_timeseries.csv", curves)
    write_csv(args.output / "fixed_reference_sole.csv", references)
    value = dict(status="RF16_SAVED_FIXED_SOLE_REVIEW_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
        trial=str(args.trial), training_code_commit=plan["code_commit"], sources=sources, labels=labels,
        physical_foot_rows=len(rows), reference_foot_rows=len(references), timeseries_rows=len(curves),
        new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0,
        user_accepted_checkpoint=None, physical_success_claimed=False,
        limits=["Exact lowest active collision mesh vertex or sphere surface above the actual floor.",
                "Only saved states through physical termination are read; reset/hold suffixes are excluded.",
                "Above2cm runs are geometric episodes, not validated human step counts or a new acceptance gate.",
                "Material speed uses saved20ms pose differences; it differs from solver-reported native loaded slip.",
                "Saved native force is the final5ms substep, not integrated20ms impulse.",
                "Reference geometry comes from the prior frozen original330-reference audit.",
                "Full available trajectories may differ in duration; this table does not establish paired causality."])
    with (args.output / "summary.json").open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value[key] for key in
                      ("status", "physical_foot_rows", "reference_foot_rows", "timeseries_rows")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("trial", "scene", "reference-csv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    audit(parser.parse_args())
