"""The strategy-neutral frame contract exposed by the FGP preview."""
from __future__ import annotations
from dataclasses import dataclass
import json
import urllib.request


@dataclass(frozen=True)
class SmplFrame:
    sequence: int
    received_ns: int
    pose_mat: list
    root_translation_m: list
    source_clock: str
    inference_ms: float | None

    @classmethod
    def from_sample(cls, sample):
        return cls(int(sample["sequence"]), int(sample["received_ns"]),
                   sample["pose_mat"], sample["root_translation_m"],
                   str(sample.get("source_clock", "fgp_receiver_monotonic")),
                   sample.get("inference_ms"))


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def read_latest(preview_url, timeout, *, fetch=get_json):
    response = fetch(preview_url.rstrip("/") + "/api/smpl", timeout)
    sample = response.get("sample")
    if response.get("status") != "receiving" or not sample or not response.get("realtime_valid", True):
        return None
    return SmplFrame.from_sample(sample)
