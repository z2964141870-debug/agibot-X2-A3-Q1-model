"""Final library, fixed-motion media and saved-checkpoint gate before Morph."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "primebot_q1"))
from common import evaluation_gate, identity, matched_checkpoint, now, require, save
from audit import audit
from evaluate import execute
from metrics import complete_library, compare


def deliver(run, iteration, output):
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="RUNNING", started_utc=now(), gate_reason=["delivery_incomplete"],
                 user_accepted_checkpoint=None, morph_training_started=False)
    try:
        initial_path, final_path = run / "checkpoint_0.pt", run / f"checkpoint_{iteration}.pt"
        saved = dict(initial=audit(run, 0), final=audit(run, iteration))
        save(output / "checkpoint_audit.json", dict(status="PASSED", **saved))
        fixed = {}
        for stage, checkpoint in (("initial", initial_path), ("final", final_path)):
            fixed[stage] = execute(checkpoint, output / (stage + "_fixed"),
                                  baseline=(output / "initial_fixed/result.json") if stage == "final" else None, render=True)
        library = {}
        for stage, checkpoint in (("initial", initial_path), ("final", final_path)):
            library[stage] = execute(checkpoint, output / (stage + "_library"), library=True)
        selected = [row["relative"] for row in library["initial"]["cases"]]
        import torch
        checkpoint = torch.load(final_path, map_location="cpu", weights_only=True, mmap=True)
        expected = [str(Path(p).relative_to(checkpoint["config"]["motion_dir"])) for p in checkpoint["identity"]["motion_paths"]]
        del checkpoint
        all_library = complete_library(library["initial"]["cases"], expected) and complete_library(library["final"]["cases"], expected)
        before = {(r["clip"], r["seed"]): r for r in fixed["initial"]["cases"]}
        pairs = [compare(before[(r["clip"], r["seed"])], r) for r in fixed["final"]["cases"]]
        media = all(r.get("video", {}).get("status") == "PASSED" for stage in fixed.values() for r in stage["cases"] if r["seed"] == 0)
        gate = evaluation_gate(pairs, all_library, media, True)
        # Cross-route exposure is factual, never a reason to continue an early-
        # stopped run beyond the user's requested quality endpoint.
        marker = run / "exposure_snapshot.json"
        exposure = dict(status="UNAVAILABLE_EARLY_STOP_BEFORE_TARGET")
        if marker.exists():
            exposed = json.loads(marker.read_text())
            match = matched_checkpoint([dict(checkpoint=exposed["checkpoint"], global_env_steps=exposed["global_env_steps"])])
            cp = Path(match["checkpoint"]["path"])
            expose_iteration = int(cp.stem.split("_")[-1])
            exposure = dict(status="AUDITED_WITHIN_ONE_PERCENT", match=match, audit=audit(run, expose_iteration))
        save(output / "paired_fixed_metrics.json", dict(pairs=pairs, gate=gate))
        save(output / "exposure_comparison_checkpoint.json", exposure)
        state.update(status="COMPLETED", **gate, ended_utc=now(),
            final_iteration=iteration, truncated_recipe=iteration < 250000,
            complete_library=all_library, local_mac_delivery_required_before_morph=True,
            media=media, library_count=len(selected), checkpoint_audit=identity(output / "checkpoint_audit.json"),
            limitation="fixed/library are training-set evaluations, not unseen-motion generalization")
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        save(output / "result.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deliver(args.run, args.iteration, args.output)
