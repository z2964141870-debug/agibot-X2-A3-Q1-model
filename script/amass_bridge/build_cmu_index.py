#!/usr/bin/env python3
"""build_cmu_index.py -- build data/cmu_index.json from the official CMU mocap site.

Per-subject file-index pages (search.php?subjectnumber=N&motion=%%%&maincat=%&subcat=%
&subtext=yes) look like:

    Subject #7 (walk)
    07_01  07_01.tvd 07_01.c3d 07_01.amc 07_01.avi  walk
    07_04  ...                                  slow walk

-> {"fetched": ..., "source": url, "subjects": {"07": {"label": "walk",
     "motions": {"1": "walk", "4": "slow walk", ...}}}}
Used by triage.py for CMU numeric files (subject label + per-trial motion name).
Run on huanyu (has internet); CPU only; ~144 small pages, <1 min.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import urllib.request

URL = ("http://mocap.cs.cmu.edu/search.php?subjectnumber=" )
HEAD_RE = re.compile(r"Subject\s*#(\d+)\s*\(([^)]*)\)")
ROW_RE = re.compile(r"^(\d{1,3})[ _-](\d{1,3})(?:\s|$)")
FILE_RE = re.compile(r"\.(tvd|c3d|amc|avi|mpg|txt|asf|trc)$|^(index|feedback|download)$", re.I)
ID_RE = re.compile(r"^\d{1,3}[_-]\d{1,3}$")


def fetch_subject(n):
    url = URL + str(n) + "&motion=%&maincat=%&subcat=%&subtext=yes"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "amass-bridge-index/1.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            html = r.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", html)
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]
        lines = [l for l in lines if l]
        label = ""
        m = HEAD_RE.search(" ".join(lines[:3]))
        if m:
            label = m.group(2).strip().lower()
        motions = {}
        for line in lines:
            rm = ROW_RE.match(line)
            if not rm:
                continue
            trial = str(int(rm.group(2)))
            toks = line.split()
            cand = [t for t in toks if not FILE_RE.search(t) and not ID_RE.match(t)]
            if cand:
                motions[trial] = " ".join(cand).strip().lower()
        return n, label, motions
    except Exception as e:
        return n, "", {"__error__": str(e)[:60]}


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cmu_index.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    subjects = {}
    errs = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        for n, label, motions in ex.map(fetch_subject, range(1, 145)):
            if "__error__" in motions:
                errs += 1
                motions = {}
            if label or motions:
                subjects["%02d" % n] = {"label": label, "motions": motions}
    idx = {"fetched": time.strftime("%F %T"), "source": URL + "N&motion=%&maincat=%&subcat=%&subtext=yes",
           "n_subjects": len(subjects), "subjects": subjects}
    with open(out_path, "w") as f:
        json.dump(idx, f, indent=0, ensure_ascii=False)
    n_trials = sum(len(s["motions"]) for s in subjects.values())
    print("cmu_index: subjects=%d trials=%d errors=%d seconds=%.0f -> %s"
          % (len(subjects), n_trials, errs, time.time() - t0, out_path))
    for k in ("07", "13", "16", "18", "19", "49", "86", "137"):
        if k in subjects:
            s = subjects[k]
            print("  %-4s label=%-24r sample=%s" % (k, s["label"], list(s["motions"].items())[:4]))


if __name__ == "__main__":
    main()
