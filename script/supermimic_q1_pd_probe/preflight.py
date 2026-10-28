"""Verify copied leg gains on both actual CPU and Warp models, no optimizer."""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

from probe import ROOT, prepare_scene, identity, save, stamp, Q1Budget
from pd_control import scene_variant, verify_native_gains


def main(a):
    import torch
    torch.set_num_threads(8)
    a.output.mkdir(parents=True, exist_ok=False)
    budget = Q1Budget(SimpleNamespace(output=a.output, runtime=a.runtime, mode="diagnosis", expected_oom_kill=0))
    source = identity(a.runtime / "q1_assets/scene.xml")
    rows = json.loads(a.library.read_text())["splits"]["validation"][:16]
    library = a.output / "library.json"
    save(library, {"splits": {"train": rows}})
    results = []
    with torch.no_grad():
        for candidate in ("baseline", "damping125"):
            case = a.output / candidate
            case.mkdir()
            scene = case / "scene.xml"
            contract = scene_variant(a.runtime / "q1_assets/scene.xml", scene, candidate)
            env = None
            try:
                budget.check()
                env = prepare_scene(a, library, case, scene, 32000, 16)
                env.reset_to_frames(env.supermimic_mdp.motion.clip_start.clone())
                before = verify_native_gains(env, contract)
                env.step(torch.zeros((16, 22), device=env.device))
                after = verify_native_gains(env, contract)
                results.append(dict(candidate=candidate, before=before, after_step=after))
            finally:
                if env is not None:
                    env.close()
                    env.backend.close()
                del env
                gc.collect()
                torch.cuda.empty_cache()
    if identity(a.runtime / "q1_assets/scene.xml") != source:
        raise AssertionError("Original scene changed")
    receipt = dict(status="PREFLIGHT_PASSED", ended_utc=stamp(), optimizer_steps=0, original_scene=source,
                   actual_16_environment_variants=results, final_resources=budget.check())
    save(a.output / "receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT / "data/supermimic_for_q1"):
        p.error("Use sole experiment root")
    main(a)
