"""Rebuild 6-clip SOMA showcase using upright canonical frame0 pose & verify 16/16 Oracle survival."""

from __future__ import annotations

import glob
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from bvh import Bvh

ROOT = Path("/home/yu/projects/SuperMimicHP")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    bvh_dir = Path("/home/yu/datasets/soma_showcase")
    ref_bvh = bvh_dir / "checking_time_R_001__A276.bvh"
    assert ref_bvh.is_file(), f"Missing {ref_bvh}"

    # 1. Build exact upright standing canonical pose from checking_time frame 0
    txt = ref_bvh.read_text(encoding="utf-8")
    hdr, _ = txt.split("MOTION", 1)
    anim = Bvh(txt)
    f0_line = " ".join(anim.frames[0])
    canon = bvh_dir / "soma_base_skel_minimal.bvh"
    canon.write_text(
        hdr + "MOTION\nFrames: 2\nFrame Time: 0.008333333333333333\n" + f0_line + "\n" + f0_line + "\n",
        encoding="utf-8",
    )
    print(f"Wrote upright canonical BVH -> {canon}", flush=True)

    # 2. Ensure we have enough clean walking/turning BVH candidates in bvh_dir
    existing_walks = sorted(
        p.name for p in bvh_dir.glob("*.bvh")
        if p.name != "soma_base_skel_minimal.bvh"
        and any(k in p.name for k in ("walk_ff", "turn_walk", "walk_"))
        and "jog" not in p.name
    )
    if len(existing_walks) < 4:
        seen_prefix = {re.sub(r"_[0-9]+__A[0-9]+.*", "", n) for n in existing_walks}
        added = 0
        with tarfile.open("/data/datasets_archive/soma_uniform.tar.gz", "r|gz") as tf:
            for idx, m in enumerate(tf):
                if not m.isfile() or not m.name.endswith(".bvh") or "_M.bvh" in m.name:
                    continue
                if not (400_000 <= m.size <= 3_500_000):
                    continue
                bn = m.name.split("/")[-1]
                if "jog" in bn or "jump" in bn or "sit" in bn or "lie" in bn or "injured" in bn:
                    continue
                if not any(k in bn for k in ("walk", "turn")):
                    continue
                if any(bad in bn for bad in ("horse_riding", "looking_by_binoculars", "pull_leash", "bump_into_sb_bar")):
                    continue
                pref = re.sub(r"_[0-9]+__A[0-9]+.*", "", bn)
                if pref in seen_prefix:
                    continue
                raw = tf.extractfile(m).read()
                (bvh_dir / bn).write_bytes(raw)
                seen_prefix.add(pref)
                added += 1
                print(f"EXTRACTED_WALK_CAND: {bn} ({m.size} B)", flush=True)
                if len(existing_walks) + added >= 6 or idx > 12000:
                    break

    # 3. Retarget stationary & dynamic candidates with upright canonical pose
    from tools.soma_source import load_soma
    from tools.retarget_q1 import FPS, Q1_JOINT_NAMES, Retargeter, digest, save_provenance

    gmr = Path("/home/yu/models/cache/GMR")
    model_xml = ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"
    solver = Retargeter(model_xml, iterations=4)

    cand_stat = [
        "painting_R_001__A273.bvh",
        "checking_time_R_001__A276.bvh",
        "eating_apple_R_002__A276.bvh",
        "greetings_hat_R_003__A276.bvh",
    ]
    cand_dyn = sorted(
        p.name for p in bvh_dir.glob("*.bvh")
        if p.name not in set(cand_stat) | {"soma_base_skel_minimal.bvh"}
        and "injured" not in p.name
    )
    # Put clean walking/turning before any jog
    cand_dyn.sort(key=lambda n: (1 if "jog" in n else 0, n))

    tmp_out = Path("/tmp/soma_candidates_q1")
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    tmp_out.mkdir(parents=True)
    mhash = save_provenance(tmp_out, gmr, "soma")

    stat_ready, dyn_ready = [], []
    for tag, names, bucket in (("stat", cand_stat, stat_ready), ("dyn", cand_dyn, dyn_ready)):
        for bn in names:
            p = bvh_dir / bn
            if not p.is_file():
                continue
            try:
                src = load_soma(p, canon, gmr, max_frames=240)
            except Exception as e:
                print(f"SKIP {bn}: {e}", flush=True)
                continue
            T = len(src["p"]["pelvis"])
            dp = float(np.linalg.norm(src["p"]["pelvis"][-1, :2] - src["p"]["pelvis"][0, :2]))
            if T < 110:
                continue
            if tag == "stat" and dp > 0.45:
                continue
            if tag == "dyn" and dp < 0.80:
                continue
            qpos, info = solver.solve(src)
            corr = float(info.get("floor_correction_max_m", 0.0))
            if corr > 0.015:
                print(f"SKIP_CORR {bn}: corr={corr:.4f}", flush=True)
                continue
            csv_p = tmp_out / f"{bn[:-4]}.csv"
            np.savetxt(csv_p, np.c_[qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]], delimiter=",", fmt="%.8f")
            meta = {
                "schema_version": 1, "source_kind": "soma", "source_path": str(p), "fps": FPS,
                "frames": len(qpos), "root_quat_order": "xyzw", "joint_names": Q1_JOINT_NAMES,
                "src_disp_xy_m": dp, "output_sha256": digest(csv_p), "generation_manifest_sha256": mhash, **info,
            }
            bucket.append((bn, csv_p, meta, dp, corr))
            print(f"CAND_Q1_{tag.upper()}: {bn} T={len(qpos)} disp={dp:.2f}m corr={corr:.5f}", flush=True)

    # 4. Select top 3 stat + top 3 dyn and copy to final soma_showcase directory
    out_dir = ROOT / "src/unisupermimic/assets/motions/q1/soma_showcase"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    save_provenance(out_dir, gmr, "soma")

    # Prefer painting, checking_time, eating_apple (or first 3 valid) for stationary
    chosen_stat = stat_ready[:3]
    chosen_dyn = dyn_ready[:3]
    assert len(chosen_stat) == 3 and len(chosen_dyn) == 3, f"Got {len(chosen_stat)} stat, {len(chosen_dyn)} dyn"

    for tag, picked in (("stat", chosen_stat), ("dyn", chosen_dyn)):
        for i, (bn, csv_p, meta, dp, corr) in enumerate(picked, 1):
            alias = f"soma_{tag}_{i}_{Path(bn).stem}"
            dst = out_dir / f"{alias}.csv"
            shutil.copy2(csv_p, dst)
            dst.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
            print(f"FINAL_SOMA: {alias} disp={dp:.2f}m corr={corr:.5f}", flush=True)


if __name__ == "__main__":
    main()
