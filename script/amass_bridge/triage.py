#!/usr/bin/env python3
"""triage.py -- AMASS corpus content triage into buckets for the A3 campaign.

Buckets (priority order when multiple match -- exercise is singled out first as
warm-up training ammo):
    exercise_stretch > dance > run > walk > daily > other
Signals:
  1) CMU numeric files: per-(subject,trial) motion names + categories from the official
     CMU index (data/cmu_index.json built by build_cmu_index.py from mocap.cs.cmu.edu).
  2) Everything else: keyword regex on the full relative path (dataset/subject/filename),
     which works for the descriptive datasets (ACCAD, EKUT, SFU, DanceDB, HDM05, ...).
Output: triage_report.json (+ stdout summary). Optional --scan-frames reads npz headers
for frame counts/duration (heavy IO on the full corpus; off by default).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

BUCKETS = ["exercise_stretch", "dance", "run", "walk", "daily", "other"]

PATTERNS = [
    ("exercise_stretch", re.compile(
        r"(?i)stretch|exercis|warm[\s_-]?up|work[\s_-]?out|yoga|gymnast|push[\s_-]?up|"
        r"sit[\s_-]?up|crunch|squat|lunge|jump(?:ing)?[\s_-]?jack|jump[\s_-]?rope|"
        r"chin[\s_-]?up|pull[\s_-]?up|calisthen|pilates|fitness|train(?:ing|er)?|"
        r"dumbbell|barbell|weightlift|aerobic|zumba|tae[\s_-]?bo|gym[\s_-]?|"
        r"(?<![a-z])jump|hop[\s_-]|leap|(?<![a-z])punch|kick[\s_-]|chop[\s_-]")),
    ("dance", re.compile(
        r"(?i)dance|ballet|tango|salsa|flamenco|break[\s_-]?dance|hip[\s_-]?hop|"
        r"\bEDB\b|folk|cha[\s_-]?cha|waltz")),
    ("run", re.compile(r"(?i)(?<![a-z])run|jog|sprint|marathon|hurdle|\brace[\s_-]")),
    ("walk", re.compile(r"(?i)walk|gait|stroll|treadmill")),
    ("daily", re.compile(
        r"(?i)household|chore|clean|sweep|vacuum|mop|wash|laundry|iron|cook|kitchen|"
        r"eat|drink|greet|conversat|talk|phone|smok|carry|lift|grab|object|assembl|"
        r"repair|screw|tool|dail|activity|activities|pick|throw|catch|open|close|door|"
        r"drawer|switch|button|sit[\s_-]|stand[\s_-]|box(?:es|ing)?[\s_-]")),
]

CMU_DIR_RE = re.compile(r"(?i)(?:^|[/\\])CMU(?:[/\\]|$)")
# bare AMASS CMU layout when triage root IS the CMU dir: 07/07_02_pos.npz
CMU_BARE_RE = re.compile(r"^\d{1,3}/\d{1,3}[ _-]\d{1,3}_pos\.npz$", re.I)


def load_cmu_index(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                idx = json.load(f)
            if isinstance(idx.get("subjects"), dict) and idx["subjects"]:
                return idx
        except Exception:
            pass
    return None


def cmu_subject_of(rel):
    parts = re.split(r"[/\\]", rel)
    for i, p in enumerate(parts):
        if p.upper() == "CMU":
            if i + 1 < len(parts) and re.fullmatch(r"\d{1,3}", parts[i + 1]):
                return int(parts[i + 1])
            m = re.match(r"CMU[_\-]?(\d{1,3})", parts[-1]) if parts else None
            if m:
                return int(m.group(1))
            return None
    m = re.match(r"CMU[_\-]?(\d{1,3})[_\.]", rel)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d{1,3})/\d{1,3}[ _-]\d{1,3}_pos\.npz$", rel, re.I)
    if m:
        return int(m.group(1))
    return None


def classify(rel, cmu_index):
    """-> (bucket, basis) ; basis explains the decision for the report."""
    text = rel.replace("\\", "/")
    if CMU_DIR_RE.search(text) or text.upper().startswith("CMU") or CMU_BARE_RE.match(text):
        subj = cmu_subject_of(text)
        if subj is not None:
            base = os.path.basename(text)
            m = re.match(r"(\d{1,3})[ _-](\d{1,3})", base)
            trial = (m.group(2).lstrip("0") or "0") if m else None
            if cmu_index:
                sdata = (cmu_index["subjects"].get("%02d" % subj)
                         or cmu_index["subjects"].get(str(subj)))
                if sdata:
                    name = sdata.get("motions", {}).get(trial, "") if trial else ""
                    label = sdata.get("label", "")
                    if name:  # trial-level motion name wins over subject label
                        for bucket, pat in PATTERNS:
                            if pat.search(name):
                                return bucket, "cmu_index_name:%s" % name
                    if label:
                        for bucket, pat in PATTERNS:
                            if pat.search("cmu " + label):
                                return bucket, "cmu_index_label:%s" % label
                    for bucket, pat in PATTERNS:
                        if pat.search(text):
                            return bucket, "cmu_path"
                    return "other", "cmu_index_no_keyword"
            return None, "cmu_no_index"
    for bucket, pat in PATTERNS:
        if pat.search(text):
            return bucket, "path_keyword"
    return "other", "no_match"


def collect_npz(root):
    out = []
    for dp, _, fns in os.walk(root):
        for f in fns:
            if f.endswith(".npz"):
                out.append(os.path.relpath(os.path.join(dp, f), root))
    out.sort()
    return out


def scan_frames(files, root, jobs):
    import numpy as np
    from joblib import Parallel, delayed

    def one(rel):
        try:
            d = np.load(os.path.join(root, rel), allow_pickle=True)
            T = int(d["poses"].shape[0])
            fps = float(d["mocap_framerate"]) if "mocap_framerate" in d.files else 30.0
            return T, fps
        except Exception as e:
            return None, str(e)[:80]

    return Parallel(n_jobs=jobs)(delayed(one)(r) for r in files)


def main():
    ap = argparse.ArgumentParser(description="AMASS corpus triage")
    ap.add_argument("--root", required=True, help="amass_raw root (searched recursively for .npz)")
    ap.add_argument("--out", default=None, help="report json path (default <script_dir>/triage_report.json)")
    ap.add_argument("--cmu-index", default=None, help="cmu_index.json path (default <script_dir>/data/cmu_index.json)")
    ap.add_argument("--limit-per-bucket", type=int, default=40, help="sample filenames kept per bucket")
    ap.add_argument("--scan-frames", type=int, default=0, help="if >0: read npz headers for first N files per bucket")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    idx_path = args.cmu_index or os.path.join(here, "data", "cmu_index.json")
    cmu_index = load_cmu_index(idx_path)

    files = collect_npz(args.root)
    buckets = {b: {"count": 0, "files": [], "datasets": defaultdict(int)} for b in BUCKETS}
    no_index_cmu = []
    bases = defaultdict(int)
    for rel in files:
        bucket, basis = classify(rel, cmu_index)
        if bucket is None:
            no_index_cmu.append(rel)
            bucket = "other"
            basis = "cmu_without_index"
        b = buckets[bucket]
        b["count"] += 1
        b["datasets"][rel.split("/")[0]] += 1
        if len(b["files"]) < args.limit_per_bucket:
            b["files"].append(rel)
        bases[basis] += 1

    report = {
        "root": os.path.abspath(args.root),
        "n_files": len(files),
        "cmu_index": {"used": cmu_index is not None, "path": idx_path},
        "buckets": {b: {"count": v["count"],
                        "datasets": dict(sorted(v["datasets"].items(), key=lambda t: -t[1])),
                        "files_sample": v["files"]} for b, v in buckets.items()},
        "decision_basis": dict(bases),
        "cmu_files_without_index": no_index_cmu[:50],
    }
    if args.scan_frames > 0:
        scan = {}
        for b in BUCKETS:
            fl = [f for f in files if classify(f, cmu_index)[0] in (b, None)][:args.scan_frames]
            res = scan_frames(fl, args.root, args.jobs)
            frames = [r[0] for r in res if r[0] is not None]
            def _f(x):
                try:
                    return round(float(x), 1)
                except (TypeError, ValueError):
                    return None
            fps = sorted(set(v for v in (_f(r[1]) for r in res) if v is not None))
            errs = [r[1] for r in res if r[0] is None][:10]
            scan[b] = {"scanned": len(fl), "total_frames": int(sum(frames)),
                       "fps_values": fps, "errors": errs}
        report["frame_scan"] = scan
        total = sum(s["total_frames"] for s in scan.values())
        report["frame_scan"]["hours_at_30fps"] = round(total / 30.0 / 3600.0, 1)

    out = args.out or os.path.join(here, "triage_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("triage: %d files -> %s" % (len(files), {b: buckets[b]["count"] for b in BUCKETS}))
    print("basis:", dict(bases))
    if no_index_cmu:
        print("WARNING: %d CMU files without index coverage" % len(no_index_cmu))
    print("report ->", out)


if __name__ == "__main__":
    main()
