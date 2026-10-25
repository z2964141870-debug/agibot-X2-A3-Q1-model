"""Explicit, content-checked views of receipts after the recorded M01 migration."""

import hashlib
import json
from pathlib import Path


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


class RelocatedView:
    def __init__(self, root, mapping):
        self.root = Path(root).resolve(strict=True)
        self.mapping = {Path(old): Path(new) for old, new in mapping.items()}
        self.translations = {}
        self.verified = {}

    def path(self, value):
        path = Path(value)
        for old in sorted(self.mapping, key=lambda item: len(item.parts), reverse=True):
            if path == old or old in path.parents:
                destination = (self.mapping[old] / path.relative_to(old)).resolve()
                if not destination.is_relative_to(self.root):
                    raise ValueError("A migrated path escapes the unified project root")
                self.translations[str(path)] = str(destination)
                return destination
        if path.is_relative_to(self.root):
            destination = path.resolve()
            if not destination.is_relative_to(self.root):
                raise ValueError("A project symlink escapes the unified root")
            return destination
        return path

    def rebase(self, value):
        if isinstance(value, dict):
            return {self.rebase(key): self.rebase(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self.rebase(item) for item in value)
        if isinstance(value, str) and value.startswith("/"):
            return str(self.path(value))
        return value

    def verify(self, record):
        expected = {key: record[key] for key in ("path", "bytes", "sha256")}
        expected["path"] = str(self.path(record["path"]))
        actual = identity(expected["path"])
        if actual != expected:
            raise ValueError("Migrated content differs from its original receipt: " + record["path"])
        self.verified[record["path"]] = {"original": record, "current": actual}
        return actual

    def verify_records(self, value):
        if isinstance(value, dict):
            if {"path", "bytes", "sha256"} <= value.keys():
                self.verify(value)
                return
            for item in value.values():
                self.verify_records(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.verify_records(item)

    def load(self, path):
        return self.rebase(json.loads(Path(path).read_text()))

    def evidence(self):
        return {"protocol": "M01_EXPLICIT_IN_MEMORY_PATH_REBINDING",
                "canonical_root": str(self.root),
                "mapping": {str(k): str(v) for k, v in self.mapping.items()},
                "translations": self.translations, "verified_files": list(self.verified.values()),
                "original_receipts_modified": False, "checkpoint_bytes_modified": False,
                "limits": "A new relocated execution, not an original-path replay. SHA and size checks remain mandatory."}


def migration_view(root):
    root = Path(root).resolve(strict=True)
    folder = root / "data/storage/M01_20261025"
    inventory = json.loads((folder / "inventory.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    if (result["status"] != "PASSED" or result["failures"] or result["copy_operations"] != 0
            or result["canonical_root"] != str(root)
            or result["manifest_sha256"] != inventory["manifest_sha256"]
            or identity(folder / "before.jsonl.gz")["sha256"] != inventory["manifest_sha256"]):
        raise ValueError("The completed M01 migration evidence is required")
    view = RelocatedView(root, inventory["mapping"])
    for old, new in result["aliases"].items():
        if Path(old).resolve(strict=True) != Path(new).resolve(strict=True):
            raise ValueError("A recorded compatibility alias changed")
    view.verify_records(inventory["frozen"])
    view.verify_records(inventory["critical"])
    view.verify(identity(folder / "inventory.json"))
    view.verify(identity(folder / "result.json"))
    return view


class JsonView:
    def __init__(self, view):
        self.view = view

    def loads(self, value, *args, **kwargs):
        return self.view.rebase(json.loads(value, *args, **kwargs))

    def __getattr__(self, name):
        return getattr(json, name)


class TorchView:
    def __init__(self, torch, view):
        self.torch, self.view = torch, view

    def load(self, *args, **kwargs):
        checkpoint = self.torch.load(*args, **kwargs)
        checkpoint["manifest"] = self.view.rebase(checkpoint["manifest"])
        return checkpoint

    def __getattr__(self, name):
        return getattr(self.torch, name)
