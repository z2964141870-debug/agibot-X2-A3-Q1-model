"""Tune the finite-difference control at equal gains and test duration transfer without refitting."""

from protocol import ROOT, EVALUATION
from run import read, run, sha
import json
import sys


def linear_selection():
    arms = ["linear_0.5", "linear", "linear_1.5"]
    run(["tune"], ["lag", "rough"], arms)
    scores = {}
    for arm in arms:
        static = []
        relay = []
        releases = []
        for f in ("lag", "rough"):
            p = ROOT / "runs/tune" / f / arm
            static += read(p / "static/trials.json")
            relay += read(p / "relay/relay.json")
            releases += [x["released"] for x in read(p / "relay/receipt.json")["episodes"]]
        a = sum(x["success"] for x in static)
        b = sum(x["success"] for x in relay)
        value = (
            0.45 * a / len(static)
            + 0.45 * b / len(relay)
            + 0.05 * sum(min(x["terminal_stable_ticks"], 50) / 50 for x in static) / len(static)
            + 0.05 * sum(releases) / len(relay)
        )
        scores[arm] = dict(static=a, relay=b, score=value)
    best = max(arms, key=lambda a: (scores[a]["score"], a == "linear"))
    (ROOT / "linear_selection.json").write_text(
        json.dumps(dict(selected=best, scores=scores), indent=2)
    )
    if best != "linear":
        run(["test0", "test1"], EVALUATION, [best])


def transfer():
    arms = ["published"] + [
        read(ROOT / f)["selected"]
        for f in ("linear_selection.json", "selection.json", "motion_selection.json")
    ]
    run(["test2"], ["half", "lag", "long"], arms)


if __name__ == "__main__":
    linear_selection() if sys.argv[1] == "linear" else transfer()
