"""Compare the inherited and pelvis-relative rest calibration on identical SOMA."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from retarget_soma import solver_module, write_json, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "pilot", "canonical", "gmr", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--existing",type=Path,help="Reuse D01 saved Q1 trajectories; no repeated IK")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selection = json.loads((args.pilot/"selection.json").read_text())
    chosen = []
    for group in ("walking", "gesture", "jogging", "turning", "dancing", "jumping"):
        candidates = sorted((r for r in selection["clips"] if r["group"] == group), key=lambda r:r["name"])
        ordinary = [r for r in candidates if r["name"].startswith("walk_ff_")]
        chosen.extend((ordinary+candidates)[:2] if ordinary else candidates[:2])
    module, _ = solver_module(args.runtime)
    from tools.soma_source import load_soma
    results = []
    for record in chosen:
        source_path = args.pilot/"source_bvh"/(record["name"]+".bvh")
        source = load_soma(source_path,args.canonical,args.gmr)
        row = {"name":record["name"],"group":record["group"],"source_sha256":digest(source_path)}
        corrected_module, _ = solver_module(args.runtime,True)
        shared_solver = corrected_module.Retargeter(args.runtime/"q1_assets/scene.xml")
        shared_target, _, _, _ = shared_solver.build_targets(source)
        for label, corrected in (("inherited",False),("pelvis_relative",True)):
            module, _ = solver_module(args.runtime,corrected)
            solver = module.Retargeter(args.runtime/"q1_assets/scene.xml")
            targets, _, _, _ = solver.build_targets(source)
            started = time.monotonic()
            if args.existing:
                qpos=np.load(args.existing/(record["name"]+"_"+label+".npz"))["qpos"]
                previous=json.loads((args.existing/"comparison.json").read_text())
                info=next(r[label] for r in previous["clips"] if r["name"]==record["name"])
                residuals=info["position_residual_p95_m"]
            else:
                qpos, info = solver.solve(source)
                residuals = dict(zip(info["position_task_order"],info["position_residual_p95_m"]))
            wrists = [f"{s}_wrist_yaw_link" for s in ("left","right")]
            actual, desired = [], []
            for frame, q in enumerate(qpos):
                solver.configuration.update(q)
                data = solver.configuration.data
                pelvis = data.xpos[solver.model.body("pelvis").id]
                actual.append([data.xpos[solver.model.body(w).id]-pelvis for w in wrists])
                desired.append([shared_target[w][frame]-shared_target["pelvis"][frame] for w in wrists])
            actual, desired = np.asarray(actual), np.asarray(desired)
            row[label] = {"seconds":time.monotonic()-started,"frames":len(qpos),"position_residual_p95_m":residuals,
                          "wrist_relative_root_rmse_m":float(np.sqrt(np.mean((actual-desired)**2))),
                          "wrist_peak_height_abs_error_m":np.abs(actual[...,2].max(axis=0)-desired[...,2].max(axis=0)).tolist(),
                          "root_height_range_m":[float(qpos[:,2].min()),float(qpos[:,2].max())],
                          "floor_correction_max_m":info["floor_correction_max_m"],"rest_pelvis_m":solver.rest["pelvis"].tolist()}
            np.savez_compressed(args.output/(record["name"]+"_"+label+".npz"),qpos=qpos)
        results.append(row)
        write_json(args.output/"comparison.json",{"status":"RUNNING","clips":results})
        print(json.dumps(row),flush=True)
    summary = {label:{"mean_pelvis_p95_m":float(np.mean([r[label]["position_residual_p95_m"]["pelvis"] for r in results])),
                     "mean_wrist_relative_root_rmse_m":float(np.mean([r[label]["wrist_relative_root_rmse_m"] for r in results])),
                     "mean_wrist_peak_height_abs_error_m":float(np.mean([r[label]["wrist_peak_height_abs_error_m"] for r in results]))}
               for label in ("inherited","pelvis_relative")}
    write_json(args.output/"comparison.json",{"status":"RETARGET_COMPARISON_COMPLETED","summary":summary,"clips":results,
               "interpretation":"Both variants use the SAME pelvis-relative anatomical wrist target for upper-limb metrics. Pelvis task residuals remain against each variant's own targets. D01's differing-target relative wrist metric is superseded. Zero physics/optimizer updates; existing IK may be reused. No new data admission threshold."})
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    main()
