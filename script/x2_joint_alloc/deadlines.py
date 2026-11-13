"""Saved-data deadline sensitivity; never reclassify actual native acceptances."""
import argparse
import json
import math
from pathlib import Path

from script.x2_fgp_realtime.assets import file_digest
from script.x2_joint_latency.analyze import load_lines
from script.x2_joint_latency.timing import statistics


def profile(rows):
    samples = [r for r in rows if r.get("kind") == "policy_sample"]
    ready = [r for r in samples if r["reply"].get("ready")]
    if not ready or any(not math.isfinite(r["rpc_ms"]) or r["rpc_ms"] < 0 for r in ready):
        raise ValueError("Require finite ready-request RPC observations")
    return dict(requests=len(samples), ready=len(ready),
        actual_native_accepted=sum(bool(r["accepted"]) for r in samples),
        hypothetical_deadline_only=[statistics([r["rpc_ms"] for r in ready], b)
                                    for b in (20., 25., 30., 1000/30)],
        actual_policy_rate_hz=50,
        caution="Post-hoc deadline-only sensitivity, not observed 30Hz or permission to use stale feedback; other validity gates still apply")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Preserve prior analysis")
    before = file_digest(a.input)
    result = profile(load_lines(a.input))
    if file_digest(a.input) != before:
        raise ValueError("Source changed during analysis")
    result.update(status="SAVED_DEADLINE_SENSITIVITY_PASSED", source=str(a.input),
                  source_bytes=a.input.stat().st_size, source_sha256=before,
                  new_model_calls=0, hardware_commands=0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
