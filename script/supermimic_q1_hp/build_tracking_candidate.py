"""Materialize an independent candidate without changing the RF09 runtime."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
TASK = Path("src/unisupermimic/tasks/motion_tracking/q1/supermimic")
PINNED = "310244b73e45775744a75b789eb25a6c9c5a81c9"


def identity(path):
    value = Path(path).read_bytes()
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def reward_error_fields(source):
    """Expose existing computed errors, leaving every analytic formula intact."""
    tree = ast.parse(source)
    changed = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value if isinstance(k, ast.Constant) else None for k in node.keys]
        if "track" not in keys or "e_body_ori" not in keys:
            continue
        for key, variable in (("e_body_linvel", "error_linvel"), ("e_body_angvel", "error_angvel")):
            if key in keys:
                raise ValueError("Candidate reward errors already exist")
            node.keys.append(ast.Constant(key))
            node.values.append(ast.Name(id=variable, ctx=ast.Load()))
        changed += 1
    if changed != 1:
        raise ValueError("Expected exactly one HP reward term dictionary")
    return ast.unparse(ast.fix_missing_locations(tree))+"\n"


def relocate(value, source, destination):
    if isinstance(value, dict):
        return {k: relocate(v, source, destination) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, source, destination) for v in value]
    if isinstance(value, str):
        old = str(source)
        if value == old or value.startswith(old+"/"):
            return str(destination)+value[len(old):]
        # Configurations may spell the same root through a directory alias.
        if Path(value).is_absolute():
            try:
                relative = Path(value).resolve().relative_to(Path(source).resolve())
            except ValueError:
                return value
            return str(Path(destination)/relative)
    return value


def build_candidate(source, destination, config, variant="joint_tracking"):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if variant not in ("sampler_only", "joint_tracking"):
        raise ValueError("Unknown candidate variant")
    if destination == source or source in destination.parents:
        raise ValueError("Use a separate runtime outside the frozen source tree")
    if destination.exists():
        raise FileExistsError(destination)
    upstream = json.loads((source/"upstream_source.json").read_text())
    if upstream["commit"] != PINNED:
        raise ValueError("Unexpected HP source commit")
    for name, expected in upstream["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid upstream manifest path")
        if identity(source/name) != expected:
            raise ValueError("Pinned upstream source identity differs: "+name)
    for name in ("LICENSE", "NOTICE", "LICENSES/MIT-supermimic.txt", "q1_adapter.json", "q1_config.yaml", str(TASK/"mdp.py"), str(TASK/"motion.py")):
        if not (source/name).is_file():
            raise FileNotFoundError(source/name)
    adapter = json.loads((source/"q1_adapter.json").read_text())
    if len(adapter["joints"]) != 22:
        raise ValueError("Expected Q1's 22-joint runtime")
    config = relocate(copy.deepcopy(config), source, destination)
    if config["algo"]["num_envs"] != 4096:
        raise ValueError("This candidate retains the authorized 4096-environment configuration")
    original = {name: identity(source/TASK/name) for name in ("mdp.py", "motion.py", "control.json")}
    base_mdp = reward_error_fields((source/TASK/"mdp.py").read_text())
    shutil.copytree(source, destination, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    task = destination/TASK
    (task/"mdp_base.py").write_text(base_mdp)
    shutil.copy2(source/TASK/"motion.py", task/"motion_base.py")
    for new, old in (("mdp.py", "tracking_mdp.py"), ("motion.py", "tracking_motion.py"), ("transition_bins.py", "transition_bins.py")):
        shutil.copy2(HERE/old, task/new)
    options = {
        "variant": variant,
        "repair_transition_bins": True,
        "joint_error_observation": variant == "joint_tracking",
        "joint_tracking_reward": variant == "joint_tracking",
        "future_goal_padding_changed": False,
        "controller_or_reset_changed": False,
    }
    (task/"candidate_options.json").write_text(json.dumps(options, indent=2)+"\n")
    control = relocate(json.loads((task/"control.json").read_text()), source, destination)
    (task/"control.json").write_text(json.dumps(control, indent=2)+"\n")
    adapter = relocate(adapter, source, destination)
    adapter["tracking_candidate"] = {**options, "source_runtime": str(source), "source_q1": original}
    (destination/"q1_adapter.json").write_text(json.dumps(adapter, indent=2)+"\n")
    # JSON is valid YAML and avoids introducing another serializer into the
    # builder; the caller parses the original YAML using the existing OmegaConf.
    (destination/"q1_config.yaml").write_text(json.dumps(config, indent=2)+"\n")
    derived = {p.name: identity(p) for p in task.iterdir() if p.is_file()}
    result = {"status": "CANDIDATE_MATERIALIZED_NOT_PHYSICALLY_VALIDATED", "source_runtime": str(source), "destination": str(destination), "upstream_commit": PINNED, "source_q1": original, "options": options, "derived_q1": derived, "optimizer_updates": 0, "training_started": False}
    (destination/"candidate_source.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--source-runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=("sampler_only", "joint_tracking"), default="joint_tracking")
    args = p.parse_args()
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(OmegaConf.load(args.source_runtime/"q1_config.yaml"), resolve=True)
    result = build_candidate(args.source_runtime, args.output, config, args.variant)
    print(json.dumps({"status": result["status"], "output": result["destination"], "options": result["options"]}))


if __name__ == "__main__":
    main()
