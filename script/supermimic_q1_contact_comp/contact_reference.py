"""Use immutable retarget contact intentions for lower-body phase signals."""

import hashlib
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"supermimic_q1_gait_comp"))
from rewards import FootReference

_LIBRARY = None


def configure(library):
    global _LIBRARY
    _LIBRARY = Path(library)


def phase_map(contact, count):
    if contact.dtype != torch.bool or contact.ndim != 2 or contact.shape[1] != 2 or len(contact) < 2:
        raise ValueError("Expected original two-foot boolean contact schedule")
    ticks = torch.arange(count, device=contact.device, dtype=torch.long)*3
    lo = (ticks//5).clamp(max=len(contact)-1)
    hi = ((ticks+4)//5).clamp(max=len(contact)-1)
    stance = contact[lo] & contact[hi]
    swing = ~(contact[lo] | contact[hi])
    return stance, swing


class ContactReference(FootReference):
    def __init__(self, mdp):
        if _LIBRARY is None:
            raise RuntimeError("Explicit original reference library required")
        super().__init__(mdp)
        source = json.loads(_LIBRARY.read_text())
        rows = [r for values in source["splits"].values() for r in values]
        by_name = {r["name"]:r for r in rows}
        if len(by_name) != len(rows):
            raise ValueError("Ambiguous original motion identity")
        stance, swing, identities = [], [], []
        for index, name in enumerate(self.motion.names):
            row = by_name[name]
            path = Path(row["path"])
            metadata_path = path.with_suffix(".json")
            raw = metadata_path.read_bytes()
            metadata = json.loads(raw)
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"] or metadata["output_sha256"] != row["sha256"]:
                raise ValueError("Original reference or retarget identity changed")
            values = metadata["contact"]
            if len(values) != metadata["frames"] or any(len(v) != 2 or any(type(x) is not bool for x in v) for v in values):
                raise ValueError("Malformed original contact intentions")
            if metadata["fps"] != 30.:
                raise ValueError("Original30Hz contact schedule required")
            contact = torch.tensor(values, device=self.motion.device, dtype=torch.bool)
            s, w = phase_map(contact, int(self.motion.clip_len[index]))
            stance.append(s); swing.append(w)
            identities.append(dict(path=str(metadata_path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        self.stance, self.swing = torch.cat(stance), torch.cat(swing)
        if bool((self.stance & self.swing).any()) or self.stance.shape != (self.motion.num_frames,2):
            raise AssertionError("Contact cache frame/phase contract")
        self.contract = dict(source_library=str(_LIBRARY),metadata=identities,
            frames=self.motion.num_frames, source_clips=len(identities),
            stance_fraction=float(self.stance.float().mean()),swing_fraction=float(self.swing.float().mean()),
            ambiguous_fraction=float((~(self.stance|self.swing)).float().mean()),
            mapping="Exact rational3/5; both source neighbors on=stance, both off=swing, mixed=neutral",
            meaning="Saved retarget intentions, not measured native force labels; original references unchanged")

    def targets(self, frames):
        p,v,h,_,_,_ = super().targets(frames)
        stance,swing = self.stance[frames],self.swing[frames]
        still = (self.motion.root_v[frames,:2].norm(dim=-1) < .05) & (self.motion.root_w[frames].norm(dim=-1) < .25) & stance.all(-1)
        return p,v,h,stance,swing,still

