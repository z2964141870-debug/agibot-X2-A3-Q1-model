"""Audit local copies against a saved native training authorization, without imports."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


UPSTREAM_PREFIX = Path("/opt/data/private/ZHY/data/references/morphmimic_v01_x2_t2_20261102/unpack/morphmimic-0.1")
SERVER_ROOT = Path("/opt/data/private/ZHY")
UPSTREAM_FILES = (
    "src/morphmimic/constants.py",
    "src/morphmimic/model/nets.py",
    "src/morphmimic/model/policy.py",
    "src/morphmimic/model/rewards_torch.py",
    "src/morphmimic/model/state.py",
    "src/morphmimic/model/world_model.py",
    "src/morphmimic/train/imagination.py",
    "src/morphmimic/train/normalizers.py",
    "src/morphmimic/train/trainer.py",
)
PROJECT_FILES = (
    "script/morphmimic_q1/assets.py",
    "script/morphmimic_q1/bootstrap.py",
    "script/morphmimic_q1/canonical.py",
    "script/morphmimic_q1/model_adapter.py",
    "script/q1_five_stage/inference.py",
    "script/q1_five_stage/q1_native_review.py",
)


def identity(path):
    raw = Path(path).read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def source_records(value):
    records = {}

    def visit(item):
        if isinstance(item, dict):
            if isinstance(item.get("path"), str) and "sha256" in item and "bytes" in item:
                key = item["path"]
                record = {name:item[name] for name in ("path", "bytes", "sha256")}
                if key in records and records[key] != record:
                    raise ValueError("Conflicting saved source identity: " + key)
                records[key] = record
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return records


def function_index(raw):
    result = []

    def visit(node, prefix=()):
        named = isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        current = prefix + (node.name,) if named else prefix
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result.append(dict(name=".".join(current), line=node.lineno, end_line=node.end_lineno,
                arguments=[arg.arg for arg in node.args.posonlyargs + node.args.args],
                decorators=[ast.unparse(item) for item in node.decorator_list]))
        for child in ast.iter_child_nodes(node):
            visit(child, current)

    visit(ast.parse(raw))
    return result


def verified_source(local, expected):
    raw = Path(local).read_bytes()
    if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError("Local source differs from saved server identity: " + str(local))
    functions = function_index(raw)
    return dict(local=str(Path(local).resolve()), server=expected, functions=functions)


def audit(authorization, upstream_root, project_root):
    auth_before = identity(authorization)
    value = json.loads(Path(authorization).read_text())
    if not all(name in value for name in ("evaluation_sources", "training_source_inputs", "controller_inputs")):
        raise ValueError("Require the saved training authorization with protected source identities")
    expected = source_records(value)
    records = []
    for prefix, local_root, names in ((UPSTREAM_PREFIX, upstream_root, UPSTREAM_FILES),
                                      (SERVER_ROOT, project_root, PROJECT_FILES)):
        for relative in names:
            server = str(prefix / relative)
            if server not in expected:
                raise ValueError("Saved authorization does not protect: " + server)
            records.append(verified_source(Path(local_root) / relative, expected[server]))
    adapter = next(row for row in records if row["local"].endswith("/model_adapter.py"))
    step = next((row for row in adapter["functions"]
                 if row["name"] == "make_world_model.NativeWorldModel.step"), None)
    if step is None or step["arguments"] != ["self", "state", "action", "ctx"] or step["decorators"]:
        raise ValueError("Missing undecorated native step override")
    for row in records:
        verified_source(row["local"], row["server"])
    if identity(authorization) != auth_before:
        raise ValueError("Saved authorization changed during source inspection")
    return dict(stage="LOCAL_FROZEN_WM_INTERFACE_AUDIT_PASSED", authorization=auth_before,
        authorization_stage=value.get("stage"), evaluation_commit=value.get("evaluation_commit"),
        source_files=len(records), source_bytes=sum(row["server"]["bytes"] for row in records),
        all_source_sizes_and_sha256_equal=True, sources=records,
        evidence_scope="historical server source identities; not a new live-server observation",
        modules_imported=False, checkpoint_loaded=False, runtime_action_gradient="NOT_EXECUTED",
        new_cuda_calls=0, new_physical_transitions=0, new_training_updates=0,
        reviewed_interface=dict(physical_state=240, effective_state=269, context=62,
            canonical_action=29, real_q1_joints=22, canonical_physical_features=268,
            critic_features_with_command_memory=297, morphology=154,
            state_command_memory_slice=[240, 269], deterministic_next_command_memory=True,
            context_layout=["friction1", "encoder_bias29", "torso_com3", "delay_seconds29"],
            action_units="raw canonical policy action; do not apply actuator scale twice",
            state_normalization="saved mean/std in canonical frame; absent features projected",
            output_scale="saved delta_scale; retain actual checkpoint configuration",
            effective_step="make_world_model.NativeWorldModel.step",
            freeze_parameters_without_no_grad=True),
        next_runtime_requirements=[
            "Recheck original host/root/boot/GPU, explicit user pause, locks and resource limits",
            "Load actual audited 32k wm buffers strictly after native bootstrap; never initialize a replacement WM",
            "Use physical saved states and recorded executed actions; teacher queries are different labels",
            "Check physical-only action gradients; command-memory identity gradients do not certify dynamics",
            "Require finite nonzero physical pose response on present joints and zero absent-joint response",
            "Keep next-reference labels offline, same frame/clip, episode masks and fixed-pair coordinates",
            "Calibrate saved-transition prediction before registering a derivative-loss training comparison",
        ], user_accepted_checkpoint=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("authorization", "upstream-root", "project-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.authorization, args.upstream_root, args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    print(json.dumps({key:result[key] for key in ("stage", "source_files", "source_bytes",
        "all_source_sizes_and_sha256_equal", "runtime_action_gradient", "new_training_updates")}))


if __name__ == "__main__":
    main()
