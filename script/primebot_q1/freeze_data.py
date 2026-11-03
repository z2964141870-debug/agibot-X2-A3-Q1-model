"""Freeze the accepted paired references and preserve all admission exclusions."""

import argparse
import json
from pathlib import Path

from common import FIXED, identity, require, save


def freeze(config, dataset, output):
    result = json.loads((dataset / "result.json").read_text())
    conversion = json.loads((dataset / "conversion.json").read_text())
    q1 = json.loads((dataset / "q1_paired.yaml").read_text())
    g1 = json.loads((dataset / "g1_paired.yaml").read_text())
    accepted = result["accepted_relative_paths"]
    require(accepted and q1["clips"] == g1["clips"] == accepted, "Common paired selection differs")
    records = {r["relative"]: r for r in conversion["records"]}
    rows = []
    for relative in accepted:
        row = records[relative]
        require(row["repair_status"] == "projected", "A repair-review candidate reached training admission")
        require(identity(row["source"]["path"]) == row["source"] and
                identity(row["converted"]["path"]) == row["converted"], "Converted reference changed")
        rows.append(dict(relative=relative, frames=row["frames"], q1_primebot=row["converted"],
            q1_morph=row["source"], g1=identity(Path(g1["root"]) / relative),
            human=identity(Path(config["human_root"]) / relative)))
    screened = [json.loads(r) for r in (dataset / "screen/results.jsonl").read_text().splitlines()]
    from collections import Counter
    counts = dict(Counter(r["status"] for r in screened))
    require(len(screened) == len(records) == 559, "Full candidate screening is incomplete")
    require(set(FIXED) <= {Path(r).stem for r in accepted}, "A frozen fixed motion was excluded")
    payload = dict(status="FROZEN", candidates=559, admitted=len(rows),
        admitted_seconds=sum(r["frames"] - 1 for r in rows) / 50.,
        original_screen_counts=counts,
        original_screen_accepted_but_repair_audit_excluded=counts.get("accepted", 0) - len(rows),
        excluded_total=559 - len(rows), all_exclusion_records=identity(dataset / "screen/results.jsonl"),
        original_thresholds_unchanged=True, original_sources_edited=False,
        fixed_motion_names=list(FIXED), q1_subset=identity(dataset / "q1_paired.yaml"),
        g1_subset=identity(dataset / "g1_paired.yaml"), xml=identity(config["xml"]),
        conversion=identity(dataset / "conversion.json"), rows=rows,
        human_pairing="same original clip and50Hz frame index; no temporal warping",
        user_accepted_checkpoint=None, weights_and_npz_storage="SERVER_LOCAL_ONLY")
    save(output, payload)
    print(json.dumps({k: payload[k] for k in ("status", "candidates", "admitted", "admitted_seconds", "original_screen_counts", "excluded_total")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "dataset", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    freeze(json.loads(args.config.read_text()), args.dataset, args.output)
