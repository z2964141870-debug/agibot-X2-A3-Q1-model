"""Independent processes each simulate one arm, one scene and one episode set."""

from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json, subprocess, sys, fcntl
from design import D, PACES, GAINS, destination


def batch(jobs):
    def one(job):
        out = destination(*job)
        root = D / "logs"
        root.mkdir(exist_ok=True)
        name = "_".join(job)
        with (root / (name + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (out / "receipt.json").exists():
                return
            with (root / (name + ".log")).open("w") as f:
                subprocess.run(
                    [sys.executable, str(D / "worker.py"), *job],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, jobs))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "data":
        batch(
            product(
                ("fit", "val"),
                ("daily", "half", "lag", "long", "trainvar"),
                ("published",),
                ("static", "relay"),
            )
        )
    elif stage == "pace":
        batch(
            [("parity", "daily", "published", "relay")]
            + list(product(("dev",), ("daily",), PACES, ("relay",)))
        )
    elif stage == "gains":
        batch(product(("dev",), ("half", "lag", "long", "trainvar"), GAINS, ("static", "relay")))
    elif stage == "controls":
        jobs = list(
            product(
                ("hold0", "hold1"),
                ("daily",),
                ("published",),
                ("static", "relay", "double", "flat"),
            )
        )
        for scene in ("half", "lag", "long", "variable", "slick", "payload", "combined"):
            jobs += list(
                product(
                    ("hold0",),
                    (scene,),
                    ("published", "linear", "fixed", "motion"),
                    ("static", "relay"),
                )
            )
        batch(jobs)
    elif stage == "test":
        s = json.loads((D / "selection.json").read_text())
        a, b = s["pace"], s["gain"]
        arms = list(dict.fromkeys(["published", a, b, a + "+" + b]))
        jobs = list(product(("hold0", "hold1"), ("daily",), arms, ("relay", "double", "flat")))
        jobs += list(product(("hold0", "hold1"), ("daily",), ("published", b), ("static",)))
        for scene in ("half", "lag", "long", "variable", "slick", "payload", "combined"):
            comparators = ["linear", "fixed", "motion"]
            jobs += list(product(("hold0",), (scene,), arms + comparators, ("relay",)))
            jobs += list(product(("hold0",), (scene,), ["published", b] + comparators, ("static",)))
        batch(jobs)
