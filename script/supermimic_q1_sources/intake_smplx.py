#!/usr/bin/env python3
"""Verify a user-provided SMPL-X package and probe its AMASS loading contract."""

import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from inventory_amass import header, health, identity


GENDERS = ("female", "male", "neutral")
PACKAGE_ROOT = "models_lockedhead/smplx/"


def extract_models(archive_path, output):
    model_dir = output / "smplx"
    model_dir.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        names = [entry.filename for entry in archive.infolist()]
        expected = {PACKAGE_ROOT + f"SMPLX_{g.upper()}.npz" for g in GENDERS}
        expected.add(PACKAGE_ROOT + "md5sums.txt")
        if len(names) != len(set(names)) or {n for n in names if not n.endswith("/")} != expected:
            raise ValueError("Unexpected or duplicate package members")
        for name in expected:
            entry = archive.getinfo(name)
            if entry.file_size > 256 * 1024 ** 2 or entry.flag_bits & 1:
                raise ValueError("Oversized or encrypted package member")
        checksum_entry = archive.getinfo(PACKAGE_ROOT + "md5sums.txt")
        if checksum_entry.file_size > 4096:
            raise ValueError("Oversized checksum file")
        checksums = archive.read(checksum_entry).decode("ascii")
        expected_md5 = {}
        for line in checksums.splitlines():
            digest, name = line.split()
            name = name.removeprefix("*").removeprefix("./")
            if len(digest) != 32 or any(c not in "0123456789abcdefABCDEF" for c in digest):
                raise ValueError("Invalid package MD5")
            if name in expected_md5:
                raise ValueError("Duplicate package MD5")
            expected_md5[name] = digest.lower()
        if set(expected_md5) != {f"SMPLX_{g.upper()}.npz" for g in GENDERS}:
            raise ValueError("Package MD5 does not cover exactly the three models")
        files = []
        for gender in GENDERS:
            name = f"SMPLX_{gender.upper()}.npz"
            target = model_dir / name
            with archive.open(PACKAGE_ROOT + name) as source, target.open("xb") as dest:
                shutil.copyfileobj(source, dest, 1024 * 1024)
            md5 = hashlib.md5(target.read_bytes()).hexdigest()
            if md5 != expected_md5[name]:
                raise ValueError(f"Package MD5 mismatch: {name}")
            with zipfile.ZipFile(target) as model_archive:
                headers = {key: header(model_archive, key) for key in
                           ("shapedirs", "v_template", "J_regressor", "weights", "kintree_table")}
            if (headers["shapedirs"]["shape"][:2] != [10475, 3]
                    or headers["shapedirs"]["shape"][2] < 16
                    or headers["J_regressor"]["shape"] != [55, 10475]):
                raise ValueError("Unsupported SMPL-X geometry or shape space")
            files.append(dict(identity(target), gender=gender, md5=md5,
                              package_md5_matched=True, headers=headers))
        with (model_dir / "md5sums.txt").open("x") as output_stream:
            output_stream.write(checksums)
    return files


def select_samples(index_path):
    samples = {}
    with index_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if (row.get("kind") == "smplx_motion" and row["gender"] in {"male", "female"}
                    and row["gender"] not in samples and row["headers"]["betas"]["shape"] == [16]
                    and row["source_frames"] <= 2500 and row["bytes"] <= 32 * 1024 ** 2):
                samples[row["gender"]] = row
    if set(samples) != {"male", "female"}:
        raise ValueError("Missing bounded male/female AMASS samples")
    return samples


def probe_model(model_path, gender, sample, source_root):
    import smplx
    import torch
    from smplx.lbs import batch_rigid_transform, batch_rodrigues

    torch.set_num_threads(1)
    source_identity = None
    if sample is None:
        poses, translations, betas = np.zeros((3, 165)), np.zeros((3, 3)), np.zeros(16)
        frame_ids, source_hz = [0, 1, 2], None
    else:
        source = source_root / sample["relative_path"]
        source_identity = identity(source)
        with np.load(source, allow_pickle=False) as data:
            frame_ids = np.linspace(0, sample["source_frames"] - 1, 3, dtype=int)
            poses, translations = data["poses"][frame_ids], data["trans"][frame_ids]
            betas = data["betas"]
            source_hz = float(data[sample["source_hz_fields"][0]])
            if source_hz != sample["source_hz"] or source_hz != 120.:
                raise ValueError("Source timebase differs from admitted AMASS index")
            if data["gender"].item() != gender or data["surface_model_type"].item() != "smplx":
                raise ValueError("Source identity metadata mismatch")
            for key, start, stop in (("root_orient", 0, 3), ("pose_body", 3, 66),
                                     ("pose_jaw", 66, 69), ("pose_eye", 69, 75),
                                     ("pose_hand", 75, 165)):
                if not np.allclose(data[key][frame_ids], poses[:, start:stop], atol=1e-8, rtol=0):
                    raise ValueError(f"Source pose field disagreement: {key}")
    if (betas.shape != (16,) or poses.shape != (3, 165) or translations.shape != (3, 3)
            or not all(np.isfinite(a).all() for a in (betas, poses, translations))):
        raise ValueError("Invalid bounded AMASS input")
    model = smplx.SMPLX(str(model_path), num_betas=16, gender=gender, use_pca=False,
                       flat_hand_mean=True, batch_size=3)
    if model.shapedirs.shape[-1] != 16 or model.num_betas != 16:
        raise ValueError("SMPL-X loader truncated the 16-dimensional shape")
    with torch.no_grad():
        pose = torch.as_tensor(poses, dtype=torch.float32)
        beta = torch.as_tensor(betas, dtype=torch.float32).expand(3, -1).clone()
        trans = torch.as_tensor(translations, dtype=torch.float32)
        kwargs = dict(global_orient=pose[:, :3], body_pose=pose[:, 3:66],
                      jaw_pose=pose[:, 66:69], leye_pose=pose[:, 69:72],
                      reye_pose=pose[:, 72:75], left_hand_pose=pose[:, 75:120],
                      right_hand_pose=pose[:, 120:165], transl=trans, return_verts=False)
        joints = model(betas=beta, **kwargs).joints
        if not torch.isfinite(joints).all():
            raise ValueError("Nonfinite SMPL-X forward joints")
        shaped = model.v_template + torch.einsum("bl,vcl->bvc", beta, model.shapedirs)
        rest = torch.einsum("jv,bvc->bjc", model.J_regressor, shaped)
        rotations = batch_rodrigues(pose.reshape(-1, 3)).reshape(3, 55, 3, 3)
        fk, _ = batch_rigid_transform(rotations, rest, model.parents)
        error = float((fk[:, :22] + trans[:, None] - joints[:, :22]).abs().max())
        if error > 1e-5:
            raise ValueError(f"Body-joint FK disagrees with full forward: {error}")
        beta[:, 15] += .5
        changed = model(betas=beta, **kwargs).joints
        beta16_effect = float((changed[:, :22] - joints[:, :22]).abs().max())
        if not torch.isfinite(changed).all() or beta16_effect <= 1e-8:
            raise ValueError("Sixteenth beta does not produce finite body-joint changes")
    result = {"gender": gender, "source": source_identity, "source_hz": source_hz,
              "frame_ids": list(map(int, frame_ids)), "num_betas": model.num_betas,
              "joints_shape": list(joints.shape), "fk_body22_max_abs_error_m": error,
              "beta16_plus_half_max_body22_change_m": beta16_effect,
              "pose_fields_agree": sample is not None, "finite": True}
    del model, shaped, rest, rotations, fk, joints, changed, beta
    gc.collect()
    return result


def run(args):
    before = health(args.expected_oom_kill)
    started = time.monotonic()
    archive_identity = identity(args.archive)
    if (archive_identity["bytes"] != args.expected_bytes
            or archive_identity["sha256"] != args.expected_sha256):
        raise ValueError("Archive differs from the Windows source identity")
    if shutil.disk_usage(args.output_dir.parent).free < 100 * 1024 ** 3:
        raise RuntimeError("Less than 100 GiB free before extraction")
    args.output_dir.mkdir(exist_ok=False)
    files = extract_models(args.archive, args.output_dir)
    samples = select_samples(args.amass_index)
    probes = []
    for model in files:
        health(before.get("oom_kill"))
        probes.append(probe_model(Path(model["path"]), model["gender"],
                                  samples.get(model["gender"]), args.source_root))
        print(json.dumps({"probe": probes[-1]}), flush=True)
    result = {"status": "SMPLX_PACKAGE_AND_16_BETA_AMASS_FK_PASSED",
              "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "archive": archive_identity, "models": files, "probes": probes,
              "amass_index": identity(args.amass_index), "script": identity(Path(__file__)),
              "versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "smplx")},
              "health_before": before, "health_after": health(before.get("oom_kill")),
              "elapsed_s": time.monotonic() - started, "storage_status": "LOCAL_ONLY",
              "new_q1_references": 0, "new_physics_steps": 0, "new_optimizer_updates": 0,
              "limits": "User-provided lockedhead_20230207 variant, not redistributed. Two bounded AMASS samples and neutral zero-pose probe only. Numerical API/FK compatibility is not proof of the original AMASS fit-model identity or qualified Q1 reference quality."}
    with (args.output_dir / "verification.json").open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--amass-index", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int)
    run(parser.parse_args())
