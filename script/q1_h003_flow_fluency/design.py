from pathlib import Path

D = Path(__file__).resolve().parent
OLD = D.parent / "delay_learning"
SCENES = {
    "daily": (0.6, 0, "0"),
    "half": (0.6, 0, "10"),
    "lag": (0.6, 0, "20"),
    "long": (0.6, 0, "30"),
    "variable": (0.6, 0, "unseen"),
    "trainvar": (0.6, 0, "train_var"),
    "slick": (0.47, 0, "20"),
    "payload": (0.6, 0.7, "20"),
    "combined": (0.51, 0.7, "20"),
}
SEEDS = {
    "fit": 6112101,
    "val": 6112117,
    "dev": 6112131,
    "hold0": 6112163,
    "hold1": 6112181,
    "parity": 230111,
}
PACES = {
    "published": (1.0, 1),
    "p80_30": (0.80, 30),
    "p80_50": (0.80, 50),
    "p90_30": (0.90, 30),
    "p90_50": (0.90, 50),
}
GAINS = {"b05": 0.5, "b10": 1.0, "b15": 1.5}


# Candidate ranking on development only. Prefer less intervention on equal scores.
def destination(split, surface, arm, kind):
    return D / "runs" / split / surface / arm / kind
