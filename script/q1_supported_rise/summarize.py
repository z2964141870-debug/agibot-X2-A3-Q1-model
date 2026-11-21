"""Summarize paired recovery experiments without pooling synthetic crouches with falls."""

import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "q1_recovery_review_20261121"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(directory):
    rows = json.loads((directory / "results.json").read_text())
    inputs = json.loads((directory / "manifest.json").read_text())
    poses = sorted({r["pose"] for r in rows if r["pose"]})
    hashes = hashlib.sha256()
    initials = hashlib.sha256()
    for i, r in enumerate(rows):
        assert r["trial"] == i
        p = directory / f"trial_{i:02d}.npz"
        hashes.update(bytes.fromhex(digest(p)))
        with np.load(p) as sample:
            initials.update(sample["initial"].tobytes())
            assert len(sample["trace"]) == r["frames"]
    result = dict(
        n=len(rows),
        fell=sum(r["phases"][0]["fell"] for r in rows),
        success=sum(r["success"] for r in rows),
        wrong_release=sum(r["wrong_release"] for r in rows),
        pre_impact_fall=sum(r["pre_impact_fall"] for r in rows),
        pose_success={
            p: sum(r["pose"] == p and r["success"] for r in rows) for p in poses
        },
        peak_qd=max((r.get("peak_qd", 0.0) for r in rows)),
        peak_overlap_mm=max(r.get("penetration_mm", 0.0) for r in rows),
        seed=inputs["seed"],
        step_s=inputs["step_s"],
        delay_ms=inputs["delay_ms"],
        results_sha256=digest(directory / "results.json"),
        manifest_sha256=digest(directory / "manifest.json"),
        inputs=inputs["files"],
        ordered_npz_digest=hashes.hexdigest(),
        initial_sha256=initials.hexdigest(),
    )
    if (directory / "standing_phase.json").exists():
        phases = json.loads((directory / "standing_phase.json").read_text())
        assert len(phases) == 24
        for k in ("success", "fell", "pose_success"):
            del result[k]
        result["phase_only"] = True
        result["phase_success_by_time"] = {
            str(t): sum(v["stood_and_handed_back"] for v in phases if v["t"] == t)
            for t in sorted({v["t"] for v in phases})
        }
    return result


def main():
    paths = sorted(p.parent for p in ROOT.rglob("results.json"))
    groups = {str(p.relative_to(ROOT)): summarize(p) for p in paths}
    for stem in ["formal500", "native"]:
        for g in ["static", "push"]:
            groups["prior/" + stem + "/" + g] = summarize(PRIOR / stem / g)
    for geom in ["fast_catch", "limbs"]:
        old = (
            "matched/77331/fast_catch"
            if geom == "fast_catch"
            else "limbs/77331/fast_catch"
        )
        for group in ("static", "push"):
            a = groups[f"axis_safe/{geom}/{group}"]
            b = groups[f"{old}/{group}"]
            assert (a["success"], a["fell"], a["initial_sha256"]) == (
                b["success"],
                b["fell"],
                b["initial_sha256"],
            )
    inputs = [p for p in ROOT.glob("*.py") if p.name not in ("fit3.py", "fit_v2.py")]
    inputs += [
        *ROOT.glob("snapshots/*.npy"),
        ROOT / "search.cc",
        ROOT / "search.so",
        ROOT / "search_before_crouch.so",
        *ROOT.glob("*.jsonl"),
        *ROOT.glob("*/fit.npz"),
        *ROOT.glob("*/expert.onnx"),
        *ROOT.glob("axis_safe/*.onnx"),
        ROOT.parent / "h003_recovery_expert/ppo_residual_01/actor_00200.pt",
        ROOT.parent / "h003_gait_engineering_01/recovery.py",
        PRIOR / "replay_controls.py",
        ROOT / "axis_safe/parity.json",
    ]
    inputs += [
        ROOT / f
        for f in [
            "Q1_supine_1x.mp4",
            "Q1_crouch_1x.mp4",
            "Q1_prone_supported_1x.mp4",
            "Q1_training.png",
            "Q1_support_training.png",
            "official_geometry.json",
        ]
    ]
    for p in ROOT.glob("overlap*.json"):
        inputs.append(p)
    audit = {}
    for p in ROOT.glob("overlap*.json"):
        a = json.loads(p.read_text())["trials"]
        audit[p.stem] = dict(
            n=len(a),
            intersecting_trials=sum(bool(x["frames_intersecting"]) for x in a),
            deepest=max((x["worst"] for x in a), key=lambda x: x["mm"]),
        )
    ff = {
        p.name: json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,nb_frames,duration",
                    "-of",
                    "json",
                    str(p),
                ]
            )
        )["streams"]
        for p in ROOT.glob("Q1*_1x.mp4")
    }
    prone = {}
    for arm in (
        "limbs/77331/formal",
        "limbs/77331/fast_catch",
        "limbs/77331/paced",
        "toward_eval/77331",
        "transfer_eval/77331",
    ):
        path = ROOT / arm / "static"
        rows = json.loads((path / "results.json").read_text())[:8]
        assert all(r["pose"] == "prone" for r in rows)
        value = []
        for r in rows:
            with np.load(path / f"trial_{r['trial']:02d}.npz") as f:
                span = f["trace"][240:300]
                value.append(
                    [
                        r["peak_qd"],
                        r["penetration_mm"],
                        *np.median(span[:, :3], axis=0),
                        np.mean(span[:, 3]),
                    ]
                )
        prone[arm] = dict(
            columns=[
                "peak_qd",
                "peak_overlap_mm",
                "median_z_m",
                "median_cosine",
                "median_foot_N",
                "body_fraction",
            ],
            window="4.80s inclusive to 6.00s exclusive for last four columns",
            value=value,
        )
    report = dict(
        prone_breakdown=prone,
        root=str(ROOT),
        authority="simulation only",
        release_required="unchanged external owner, calm release, 1s upright",
        development=dict(train=[7521, 7522, 7523], selection=7524, search_rng=281103),
        experiments=groups,
        kinematic_overlap=audit,
        official_geometry=json.loads((ROOT / "official_geometry.json").read_text()),
        parity=json.loads((ROOT / "axis_safe/parity.json").read_text()),
        media=ff,
        sources={str(p): digest(p) for p in sorted(set(inputs))},
    )
    (ROOT / "q1_supported_rise_20261121.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
