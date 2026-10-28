"""Reference-conditioned physical foot, smoothness and stationary objectives."""

from __future__ import annotations

import torch

FEATURES = 1420
REWARD_MAXIMUM = .15


class FootReference:
    def __init__(self, mdp):
        self.mdp, self.motion = mdp, mdp.motion
        self.ids = [mdp.kin.body_names.index(side + "_ankle_roll_link") for side in ("left", "right")]
        positions, velocities = [], []
        with torch.no_grad():
            for start in range(0, self.motion.num_frames, 8192):
                end = min(start + 8192, self.motion.num_frames)
                frames = torch.arange(start, end, device=self.motion.device)
                out = mdp.kin.forward(*self.motion.ref_state(frames), all_bodies=True)
                positions.append(out[0][:, self.ids])
                velocities.append(out[2][:, self.ids])
            self.positions, self.velocities = torch.cat(positions), torch.cat(velocities)
            self.floor = torch.zeros(len(self.motion.clip_start), 2, device=self.motion.device)
            for clip, (start, end) in enumerate(zip(self.motion.clip_start.tolist(), self.motion.clip_end.tolist())):
                self.floor[clip] = torch.quantile(self.positions[start:end+1, :, 2], .05, dim=0)

    def targets(self, frames):
        position, velocity = self.positions[frames], self.velocities[frames]
        height = position[..., 2] - self.floor[self.motion.clip_id[frames]]
        stance = (height < .025) & (velocity[..., :2].norm(dim=-1) < .2)
        swing = height >= .025
        still = (self.motion.root_v[frames, :2].norm(dim=-1) < .05) & (self.motion.root_w[frames].norm(dim=-1) < .25) & stance.all(-1)
        return position, velocity, height, stance, swing, still

    def measured(self, state):
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
        out = self.mdp.kin.forward(*unflatten_state(state), all_bodies=True)
        return out[0][:, self.ids], out[2][:, self.ids]

    def features(self, state, frames):
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state, yaw_mat
        actual_p, actual_v = self.measured(state)
        reference_p, reference_v, _, stance, _, still = self.targets(frames)
        s = unflatten_state(state)
        actual_heading = yaw_mat(s[1]).transpose(-1, -2).unsqueeze(1)
        reference_heading = yaw_mat(self.motion.root_R[frames]).transpose(-1, -2).unsqueeze(1)
        local = lambda rotation, value: (rotation @ value.unsqueeze(-1)).squeeze(-1)
        actual = local(actual_heading, actual_p-s[0][:, None])
        target = local(reference_heading, reference_p-self.motion.root_p[frames, None])
        actual_velocity = local(actual_heading, actual_v-s[2][:, None])
        target_velocity = local(reference_heading, reference_v-self.motion.root_v[frames, None])
        return torch.cat(((target-actual).flatten(1) / .1,
                          (target_velocity-actual_velocity).flatten(1) / .5,
                          stance.float(), still[:, None].float()), -1).clamp(-10, 10)

    def reward(self, original, state, frames, targets, previous_targets, residual, force, slip):
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state, yaw_mat
        s = unflatten_state(state)
        actual, _ = self.measured(state)
        reference, _, _, stance, swing, still = self.targets(frames)
        local = lambda rotation, value: (rotation.transpose(-1,-2).unsqueeze(1) @ value.unsqueeze(-1)).squeeze(-1)
        local_actual = local(yaw_mat(s[1]), actual-s[0][:, None])
        local_reference = local(yaw_mat(self.motion.root_R[frames]), reference-self.motion.root_p[frames, None])
        stance_score = torch.exp(-(slip / .15).square()) * (force > 5.)
        stance_bonus = (stance_score * stance).sum(-1) / stance.sum(-1).clamp(min=1)
        xy_error = ((local_actual-local_reference)[..., :2] / .1).square().sum(-1)
        z_error = ((actual[..., 2]-reference[..., 2]) / .04).square()
        swing_score = .5 * torch.exp(-xy_error) + .5 * torch.exp(-z_error)
        swing_bonus = (swing_score * swing).sum(-1) / swing.sum(-1).clamp(min=1)
        smooth_cost = ((targets-previous_targets) / .1).square().mean(-1).clamp(max=25.)
        # World XY is deliberately absent: stationarity is velocity and tilt.
        gravity_error = (s[1][:, 2] - self.motion.root_R[frames, 2]).square().sum(-1)
        stationary = torch.exp(-s[2][:, :2].square().sum(-1) / .1**2
                               -s[3].square().sum(-1) / .5**2 -gravity_error / .2**2) * still
        residual_cost = (residual / .15).square().mean(-1)
        parts = {"stance_bonus": .02*stance_bonus, "swing_bonus": .02*swing_bonus,
                 "stationary_bonus": .01*stationary, "smooth_cost": .002*smooth_cost,
                 "residual_cost": .002*residual_cost}
        total = original + parts["stance_bonus"] + parts["swing_bonus"] + parts["stationary_bonus"]
        total -= parts["smooth_cost"] + parts["residual_cost"]
        return total, parts
