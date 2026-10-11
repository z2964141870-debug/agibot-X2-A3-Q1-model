"""Create a small deterministic detection dataset for hardware probes."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def make_split(root: Path, split: str, count: int, seed: int) -> None:
    rng = random.Random(seed)
    images = root / "images" / split
    labels = root / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        w = h = 640
        im = Image.new("RGB", (w, h), (rng.randrange(10, 35), rng.randrange(10, 35), rng.randrange(10, 35)))
        draw = ImageDraw.Draw(im)
        rows = []
        for _ in range(rng.randint(1, 5)):
            cls = rng.randrange(3)
            bw = rng.randint(45, 220)
            bh = rng.randint(45, 220)
            x1 = rng.randint(0, w - bw - 1)
            y1 = rng.randint(0, h - bh - 1)
            x2, y2 = x1 + bw, y1 + bh
            color = [(235, 80, 70), (80, 210, 110), (80, 130, 235)][cls]
            draw.rectangle((x1, y1, x2, y2), fill=color, outline=(255, 255, 255), width=2)
            rows.append(f"{cls} {(x1+x2)/(2*w):.6f} {(y1+y2)/(2*h):.6f} {bw/w:.6f} {bh/h:.6f}")
        stem = f"synthetic_{split}_{i:05d}"
        im.save(images / f"{stem}.jpg", quality=90)
        (labels / f"{stem}.txt").write_text("\n".join(rows) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--train", type=int, default=1800)
    p.add_argument("--val", type=int, default=200)
    args = p.parse_args()
    make_split(args.root, "train", args.train, 20261011)
    make_split(args.root, "val", args.val, 20261012)
    data = {"path": str(args.root), "train": "images/train", "val": "images/val", "names": {0: "red", 1: "green", 2: "blue"}}
    (args.root / "data.yaml").write_text(json.dumps(data, indent=2) + "\n")
    (args.root / "manifest.json").write_text(json.dumps({"train": args.train, "val": args.val, "classes": 3, "generator": "make_synthetic_yolo.py", "seed_train": 20261011, "seed_val": 20261012}, indent=2) + "\n")


if __name__ == "__main__":
    main()
