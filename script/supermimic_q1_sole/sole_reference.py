"""Independent RF16 reward candidate: replace only the swing world-Z score."""

import hashlib
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_contact_comp")]
from contact_reference import ContactReference
from sole_geometry import SoleGeometry
from tensor_geometry import TensorSole, swing_z_correction


class SoleReference(ContactReference):
    def __init__(self, mdp, scene):
        super().__init__(mdp)
        scene = Path(scene)
        before = hashlib.sha256(scene.read_bytes()).hexdigest()
        geometry = SoleGeometry.from_scene(scene)
        self.sole = TensorSole(geometry, mdp.kin.body_names, self.motion.device)
        heights = []
        with torch.no_grad():
            for start in range(0, self.motion.num_frames, 1024):
                frames = torch.arange(start, min(start + 1024, self.motion.num_frames),
                                      device=self.motion.device)
                fk = mdp.kin.forward(*self.motion.ref_state(frames), all_bodies=True)
                heights.append(self.sole(fk[0], fk[1]))
            self.reference_sole = torch.cat(heights)
        if not bool(torch.isfinite(self.reference_sole).all()):
            raise ValueError("Nonfinite immutable reference collision-sole cache")
        if hashlib.sha256(scene.read_bytes()).hexdigest() != before:
            raise ValueError("Scene changed while loading collision geometry")
        self.contract = {**self.contract, "sole": self.sole.contract,
                         "scene": dict(path=str(scene), bytes=scene.stat().st_size, sha256=before),
                         "reward_change": "Only swing world-Z half: ankle-roll origin -> collision sole",
                         "features_and_reference_contact_phase": "Unchanged RF16",
                         "reference_sole_frames": len(self.reference_sole)}

    def reward(self, original, state, frames, targets, previous_targets, residual, force, slip):
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state

        total, parts = super().reward(original, state, frames, targets, previous_targets,
                                     residual, force, slip)
        fk = self.mdp.kin.forward(*unflatten_state(state), all_bodies=True)
        sole = self.sole(fk[0], fk[1])
        _, _, _, _, swing, _ = self.targets(frames)
        correction = swing_z_correction(fk[0][:, self.ids, 2], self.positions[frames, :, 2],
                                         sole, self.reference_sole[frames], swing)
        return total + correction, {**parts, "swing_bonus": parts["swing_bonus"] + correction}
