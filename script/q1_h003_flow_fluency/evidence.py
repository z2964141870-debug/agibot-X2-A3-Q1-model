"""Package trial-level outcomes and hash the original arrays, models and media."""

import gzip, hashlib, json
from design import D
from assess import read, put


def digest(f):
    h = hashlib.sha256()
    with f.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package():
    files = []
    outcomes = {}
    for receipt in sorted((D / "runs").glob("*/*/*/*/receipt.json")):
        root = receipt.parent
        if root.name == "repeat":
            continue
        result = root / ("trials.json" if root.name == "static" else "relay.json")
        rows = read(result)
        assert len(rows) == read(receipt)["episodes"]
        outcomes[str(root.relative_to(D))] = dict(receipt=read(receipt), rows=rows)
        files.extend(p for p in root.iterdir() if p.is_file())
    trial_file = D / "episodes.json.gz"
    trial_file.write_bytes(
        gzip.compress(json.dumps(outcomes, separators=(",", ":")).encode(), mtime=0)
    )
    files.extend(sorted((D / "entry_replay").rglob("*.npz")))
    files.extend((D / "age.onnx", D / "age.pt"))
    raw = D / "raw_sha256.txt"
    raw.write_text("".join(digest(f) + "  " + str(f.relative_to(D)) + "\n" for f in sorted(files)))
    artifacts = {}
    for f in sorted(D.iterdir()):
        if f.is_file() and (
            f.suffix in (".py", ".onnx", ".pt", ".mp4", ".pdf", ".png", ".json", ".gz", ".txt")
            and f.name != "manifest.json"
        ):
            artifacts[f.name] = {"bytes": f.stat().st_size, "sha256": digest(f)}
    put(
        D / "manifest.json",
        dict(
            server="yu-3090",
            root=str(D),
            files_hashed=len(files),
            complete_runs=len(outcomes),
            episode_rows=sum(len(s["rows"]) for s in outcomes.values()),
            artifacts=artifacts,
        ),
    )


if __name__ == "__main__":
    package()
