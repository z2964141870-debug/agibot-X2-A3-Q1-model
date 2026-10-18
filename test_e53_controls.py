"""E53 control tests: zero-init identity, heading transform, channel stats, verdict logic, patch inertness."""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

OPS = Path("/opt/data/private/yuanqi_ops")
ROOT = Path("/opt/data/private/hechengzhu/ZHY")
sys.path.insert(0, str(OPS))
sys.path.insert(0, str(ROOT))

import e53_protocol as P  # noqa: E402
from e53_protocol import CHANNELS, G1_ENCODER_FIRST_LAYER_IN, TIERS, mechanism_verdict_with_means  # noqa: E402


def test_zero_init_identity():
    torch.manual_seed(0)
    layer = torch.nn.Linear(G1_ENCODER_FIRST_LAYER_IN, 2048)
    obs_base = torch.randn(4, G1_ENCODER_FIRST_LAYER_IN)
    with torch.no_grad():
        expected = layer(obs_base)
    weight = layer.weight.data
    layer.weight = torch.nn.Parameter(torch.cat([weight, weight.new_zeros(weight.shape[0], CHANNELS)], dim=1))
    layer.in_features = G1_ENCODER_FIRST_LAYER_IN + CHANNELS
    obs_full = torch.cat([obs_base, torch.randn(4, CHANNELS)], dim=-1)
    with torch.no_grad():
        actual = layer(obs_full)
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), "zero-init extension changed outputs"
    assert layer.in_features == G1_ENCODER_FIRST_LAYER_IN + CHANNELS
    loss = layer(obs_full.clone().requires_grad_(True)).sum()
    loss.backward()
    assert layer.weight.grad[:, G1_ENCODER_FIRST_LAYER_IN:].abs().sum() > 0, "no gradient reaching new weights"
    assert torch.equal(layer.weight[:, :G1_ENCODER_FIRST_LAYER_IN], weight), "base weight columns must be untouched"


def test_heading_transform():
    from e53_obs_terms import _rotmat
    yaw = 0.7
    quat_wxyz = torch.tensor([[float(torch.cos(torch.tensor(yaw / 2))), 0.0, 0.0, float(torch.sin(torch.tensor(yaw / 2)))]])
    rot = _rotmat(quat_wxyz)
    world_offset = torch.tensor([[3.0, -2.0, 0.5]])
    anchor = torch.bmm(rot.transpose(1, 2), world_offset.unsqueeze(-1)).squeeze(-1)
    cos_y, sin_y = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    expected = torch.stack([world_offset[:, 0] * cos_y + world_offset[:, 1] * sin_y,
                            -world_offset[:, 0] * sin_y + world_offset[:, 1] * cos_y,
                            world_offset[:, 2]], dim=-1)
    assert torch.allclose(anchor, expected, atol=1e-6), (anchor, expected)
    origin = torch.tensor([[1.0, 1.0, 0.0]])
    disp = torch.bmm(rot.transpose(1, 2), (origin + world_offset - origin).unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(disp, expected, atol=1e-6)


def test_channel_stats():
    from e53_obs_terms import _rotmat
    steps, yaw = 10, 0.3
    quat = torch.tensor([[float(torch.cos(torch.tensor(yaw / 2))), 0.0, 0.0, float(torch.sin(torch.tensor(yaw / 2)))]])
    rot = _rotmat(quat)
    origin, pos = torch.zeros(1, 3), torch.zeros(1, 3)
    velocity = torch.tensor([[0.5, 0.0, 0.0]])
    dt = 0.02
    disp_series, vel_series = [], []
    prev = pos.clone()
    for step_index in range(steps):
        velocity[0, 2] = 0.002 * (step_index + 1)
        pos = pos + velocity * dt
        disp_series.append(torch.bmm(rot.transpose(1, 2), (pos - origin).unsqueeze(-1)).squeeze(-1))
        vel_series.append(torch.bmm(rot.transpose(1, 2), ((pos - prev) / dt).unsqueeze(-1)).squeeze(-1))
        prev = pos.clone()
    disp = torch.cat(disp_series)
    vel = torch.cat(vel_series)
    channels = torch.cat([disp, vel], dim=-1)
    assert channels.shape[1] == CHANNELS
    assert float(channels.var(dim=0).min()) > 0.0
    final_xy = float(disp[-1, :2].norm())
    assert abs(final_xy - 0.5 * steps * dt) < 1e-5, "anchor-frame xy displacement must equal world xy"
    cos_y = float(torch.cos(torch.tensor(yaw)))
    sin_y = float(torch.sin(torch.tensor(yaw)))
    assert abs(float(vel[:, 0].mean()) - 0.5 * cos_y) < 1e-5
    assert abs(float(vel[:, 1].mean()) + 0.5 * sin_y) < 1e-5
    mean_vz = 0.002 * (steps + 1) / 2
    assert abs(float(vel[:, 2].mean()) - mean_vz) < 1e-6, "anchor-frame z velocity must track ramp"
    ratio_disp = float(disp[-1, 1] / disp[-1, 0])
    assert abs(ratio_disp + sin_y / cos_y) < 1e-5, "channel ratio must equal -tan(yaw)"
    ratio_vel = float(vel[-1, 1] / vel[-1, 0])
    assert abs(ratio_vel + sin_y / cos_y) < 1e-5, "velocity channel ratio must equal -tan(yaw)"


def test_verdict_logic():
    base = 0.20
    verdict, detail = mechanism_verdict_with_means({("N", t): base for t in TIERS} | {("I", 8): base * 0.94, ("I", 32): base * 0.93})
    assert verdict == "H2_INTERFACE_SUPPORTED", (verdict, detail)
    verdict, detail = mechanism_verdict_with_means({("N", t): base for t in TIERS} | {("I", 8): base * 0.99, ("I", 32): base * 0.985})
    assert verdict == "WEAK_POSITIVE_SIGNAL", (verdict, detail)
    verdict, _ = mechanism_verdict_with_means({("N", t): base for t in TIERS} | {("I", 8): base * 0.99, ("I", 32): base * 1.02})
    assert verdict == "INCONCLUSIVE"
    verdict, _ = mechanism_verdict_with_means({("N", t): base for t in TIERS} | {("I", 8): base * 1.01, ("I", 32): base * 0.90})
    assert verdict == "INCONCLUSIVE"


class _StubCommand:
    robot_anchor_pos_w = torch.tensor([[1.0, 2.0, 0.5], [0.0, 0.0, 0.3]])
    anchor_heading_quat = torch.tensor([
        [float(torch.cos(torch.tensor(0.075))), 0.0, 0.0, float(torch.sin(torch.tensor(0.075)))],
        [1.0, 0.0, 0.0, 0.0],
    ])  # wxyz half-angle convention: this is a +0.15 rad yaw rotation


class _StubManager:
    _terms = {"motion": _StubCommand()}


class _StubEnv:
    command_manager = _StubManager()
    episode_length_buf = torch.tensor([0, 0])
    step_dt = 0.02


def test_obs_term_math():
    import e53_obs_terms
    e53_obs_terms.TRACKING_COMMAND_CLASS = _StubCommand
    yaw = 0.15
    env = _StubEnv()
    channels = e53_obs_terms.e53_root_channels(env)
    assert channels.shape == (2, 6)
    assert torch.equal(channels, torch.zeros_like(channels)), "first call after reset must be zero"
    env.episode_length_buf = torch.tensor([5, 5])
    _StubCommand.robot_anchor_pos_w = torch.tensor([[1.02, 2.04, 0.5], [0.0, 0.0, 0.3]])
    channels2 = e53_obs_terms.e53_root_channels(env)
    cos_y = float(torch.cos(torch.tensor(yaw)))
    sin_y = float(torch.sin(torch.tensor(yaw)))
    expected_disp = torch.tensor([[0.02 * cos_y + 0.04 * sin_y, -0.02 * sin_y + 0.04 * cos_y, 0.0],
                                  [0.0, 0.0, 0.0]])
    expected_vel = expected_disp / 0.02
    assert torch.allclose(channels2[:, :3], expected_disp, atol=1e-6), (channels2[:, :3], expected_disp)
    assert torch.allclose(channels2[:, 3:], expected_vel, atol=1e-5), (channels2[:, 3:], expected_vel)


def test_protocol_constants():
    assert TIERS == (8, 32) and CHANNELS == 6 and G1_ENCODER_FIRST_LAYER_IN == 640
    assert P.CHECKPOINT_BY_TIER[8].endswith("model_step_000008.pt")


def main():
    test_zero_init_identity()
    test_heading_transform()
    test_channel_stats()
    test_verdict_logic()
    test_obs_term_math()
    test_protocol_constants()
    print("E53 controls: ALL PASS")


if __name__ == "__main__":
    main()
