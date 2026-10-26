"""Use HP's pinned SOMA loader and Mink solver with the supplied Q1 geometry."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys
import time
import types

import mujoco
import numpy as np

QUOTAS = {"walking": 28, "jogging": 22, "turning": 16, "gesture": 16, "dancing": 14, "jumping": 4}
VAL_QUOTAS = {"walking": 3, "jogging": 3, "turning": 2, "gesture": 3, "dancing": 3, "jumping": 1}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Q1RetargetTree(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace("wrist_roll_link", "wrist_yaw_link")
        return node

    def visit_Assign(self, node):
        # Q1 has no actuated head; HP's head-neutral assignment has no target.
        if any(isinstance(v, ast.Constant) and v.value == "head_yaw_joint" for v in ast.walk(node)):
            return None
        return self.generic_visit(node)


def solver_module(runtime, root_relative_rest=False):
    path = runtime / "tools/retarget_x2.py"
    sys.path[:0] = [str(runtime), str(runtime / "src")]
    tree = Q1RetargetTree().visit(ast.parse(path.read_text()))
    ast.fix_missing_locations(tree)
    module = types.ModuleType("q1_hp_retarget_runtime")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    robot = runpy.run_path(str(runtime / "src/unisupermimic/tasks/motion_tracking/q1/supermimic/robot.py"))
    module.X2_JOINT_NAMES = robot["Q1_JOINT_NAMES"]
    module.default_joint_pos_list = robot["default_joint_pos_list"]
    module.motor_of = robot["motor_of"]
    original = module.local_collision_vertices

    def foot_vertices(model, body_name):
        from tools.x2_geometry import CollisionGeometry
        import trimesh

        geometry = CollisionGeometry(model, (body_name,))
        if all(i in geometry.vertices for i in geometry.ids):
            return original(model, body_name)
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        body = model.body(body_name).id
        points = []
        for geom in geometry.ids:
            vertices = geometry.vertices.get(geom)
            if vertices is None:
                if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_SPHERE:
                    raise ValueError("Q1 foot support expects mesh, box or sphere")
                vertices = trimesh.creation.icosphere(subdivisions=3, radius=model.geom_size[geom, 0]).vertices
            world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            points.append((world - data.xpos[body]) @ data.xmat[body].reshape(3, 3))
        return np.concatenate(points)

    module.local_collision_vertices = foot_vertices
    if root_relative_rest:
        original_retargeter = module.Retargeter

        class RootRelativeRetargeter(original_retargeter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.rest_origin = self.rest["pelvis"].copy()
                # HP's targets add these offsets to the animated pelvis. Q1's
                # qpos0 pelvis is above zero, so world rest positions double it.
                self.rest = {name: value-self.rest_origin for name, value in self.rest.items()}

            def solve(self, source):
                qpos, info = super().solve(source)
                info["rest_coordinate_frame"] = "relative to qpos0 pelvis"
                info["removed_rest_origin_m"] = self.rest_origin.tolist()
                return qpos, info

        module.Retargeter = RootRelativeRetargeter
    return module, robot


def process(job):
    runtime, pilot, canonical, gmr, output, iterations, root_relative_rest, record = job
    started = time.monotonic()
    result = {"name": record["name"], "split": record["split"], "group": record["group"], "actor_uid": record["actor_uid"]}
    try:
        module, robot = solver_module(runtime, root_relative_rest)
        from tools.soma_source import load_soma

        source_path = pilot / "source_bvh" / (record["name"] + ".bvh")
        if digest(source_path) != record["sha256"]:
            raise ValueError("Selected source changed")
        source = load_soma(source_path, canonical, gmr)
        scene = runtime / "q1_assets/scene.xml"
        solver = module.Retargeter(scene, iterations=iterations)
        qpos, info = solver.solve(source)
        if qpos.shape[1] != 29 or len(qpos) < 2 or not np.isfinite(qpos).all():
            raise ValueError("Invalid Q1 trajectory dimensions or numbers")
        path = output / (record["name"] + ".csv")
        np.savetxt(path, np.c_[qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]], delimiter=",", fmt="%.8f")
        metadata = {"source_record": record, "source_path": str(source_path), "source_sha256": digest(source_path), "canonical_sha256": digest(canonical), "model_path": str(scene), "model_sha256": digest(scene), "fps": 30.0, "frames": len(qpos), "joint_names": robot["Q1_JOINT_NAMES"], "root_quat_order": "xyzw", "output_sha256": digest(path), "ik_iterations": iterations, **info}
        write_json(path.with_suffix(".json"), metadata)
        result.update(status="READABLE_FINITE_Q1", path=str(path), sha256=digest(path), frames_30hz=len(qpos), duration_seconds=(len(qpos)-1)/30, root_xy_net_m=float(np.linalg.norm(qpos[-1, :2]-qpos[0, :2])), floor_correction_max_m=info["floor_correction_max_m"])
    except Exception as error:
        result.update(status="PROCESSING_ERROR", error=f"{type(error).__name__}: {error}")
    result["seconds"] = time.monotonic() - started
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--gmr", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--root-relative-rest", action="store_true")
    p.add_argument("--quotas", type=Path, help="JSON with train/validation per-group counts")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selection = json.loads((args.pilot / "selection.json").read_text())
    results = []
    jobs = [(args.runtime, args.pilot, args.canonical, args.gmr, args.output, args.iterations, args.root_relative_rest, r) for r in selection["clips"]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(process, jobs):
            results.append(result)
            write_json(args.output / "retarget_report.json", results)
    splits = {}
    deficits = {}
    requested = json.loads(args.quotas.read_text()) if args.quotas else {"train": QUOTAS, "validation": VAL_QUOTAS}
    for split, quotas in requested.items():
        chosen = []
        for group, count in quotas.items():
            choices = [r for r in results if r["status"] == "READABLE_FINITE_Q1" and r["split"] == split and r["group"] == group]
            choices.sort(key=lambda r: hashlib.sha256(r["name"].encode()).hexdigest())
            chosen.extend(choices[:count])
            if len(choices) < count:
                deficits[f"{split}/{group}"] = count - len(choices)
        splits[split] = chosen
    actors = {s: sorted({r["actor_uid"] for r in rows}) for s, rows in splits.items()}
    if set(actors["train"]) & set(actors["validation"]):
        raise ValueError("Source actor split contract violated")
    manifest = {"status": "Q1_SOMA_LIBRARY_READY" if splits["train"] else "NO_READABLE_TRAINING_CLIPS", "upstream_commit": "310244b73e45775744a75b789eb25a6c9c5a81c9", "selection_note": "HP archive-order pilot and within-category SHA256(name) order; independent Q1 conversion. Not the labmate's exact QC-selected 100/15 source list. No X2 or agent-defined quality rejection gates.", "root_relative_rest": args.root_relative_rest, "requested_quotas": requested, "source_selection_sha256": digest(args.pilot / "selection.json"), "canonical_sha256": digest(args.canonical), "gmr_utility_sha256": digest(args.gmr / "general_motion_retargeting/utils/lafan_vendor/utils.py"), "adapter_sha256": digest(Path(__file__)), "packages": {n: importlib.metadata.version(n) for n in ("mujoco", "mink", "numpy", "scipy", "daqp", "bvh", "trimesh")}, "foot_sphere_support": "Trimesh icosphere subdivisions=3 for support-point selection; analytic primitive minimum retained in final floor projection", "actors": actors, "deficits": deficits, "counts": {s: len(rows) for s, rows in splits.items()}, "splits": splits, "processing_errors": [r for r in results if r["status"] != "READABLE_FINITE_Q1"]}
    write_json(args.output / "pilot_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "deficits": deficits}), flush=True)


if __name__ == "__main__":
    main()
