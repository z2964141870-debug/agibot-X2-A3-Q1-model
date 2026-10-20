"""E55 controls - tests DERIVED FROM contract text (v1.2 + Amendment A1), not from the
implementation's self-description (lead requirement: no shared blind spots with the code).

Lessons encoded: B.2 (WDR/per_joint are guards + cloth6 tie-break layer, never the rank key),
3-shutter liveness semantics, slot-major channel layout (evidence e53_trainer.py L103-114
view(n_slots, base_per_slot)+zeros concat, E54 2432 incident 30=6x5), entry build whitelist.
"""

import json
import os
import sys
import types

import torch

OPS = "/opt/data/private/yuanqi_ops"
ROOT = "/opt/data/private/hechengzhu/ZHY"
sys.path.insert(0, OPS)
sys.path.insert(0, ROOT)

FAILURES = []


def check(name, condition, detail=""):
    print("controls %s %s %s" % ("pass" if condition else "FAIL", name, detail))
    if not condition:
        FAILURES.append(name)


class _Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(700, 29)
        self.tokenizer_obs_specs = [("motion_anchor_ori_b_mf_nonflat", 0, 700, None)]

    def forward(self, net_input):
        tensor = net_input["tokenizer"] if isinstance(net_input, dict) else net_input
        return self.linear(tensor.reshape(tensor.shape[0], -1))


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_module = _Actor()


def test_head_basics():
    from e55_residual import E55ResidualHead, install_injected
    head = E55ResidualHead()
    out = head(torch.randn(4, 10, 58), torch.randn(4, 10, 6))
    check("A1 head zero-init exact", float(out.abs().max()) == 0.0)
    try:
        head(torch.randn(2, 10, 64), torch.randn(2, 10, 6))
        check("head rejects 64-wide anchor", False)
    except ValueError:
        check("head rejects 64-wide anchor", True)
    policy = _Policy()
    try:
        policy_p = _Policy()
        install_injected(policy_p, anchor_start=1762)
        check("default flat_base is production 2402 (anchor group at 1762)", True)
    except ValueError:
        check("default flat_base is production 2402 (anchor group at 1762)", False)


def test_install_order_and_ori_blind():
    from e55_residual import install_injected
    policy = _Policy()
    install_injected(policy, anchor_start=0, flat_base=640)
    with torch.no_grad():
        policy.e55_head.head.weight.add_(0.01)
    flat = torch.randn(3, 700)
    out = policy.actor_module.forward(flat)
    lin = policy.actor_module.linear
    base = torch.nn.functional.linear(flat, lin.weight, lin.bias)
    check("residual follows head weights", float((out - base).abs().max()) > 0)
    anchor = torch.stack([flat[:, s * 70:s * 70 + 58] for s in range(10)], 1)
    chans = torch.stack([flat[:, s * 70 + 64:s * 70 + 70] for s in range(10)], 1)
    check("injection order unit (slot[0:58]+slot[64:70])",
          torch.allclose(out, base + policy.e55_head(anchor, chans), atol=1e-5))
    idx = [s * 70 + 58 + k for s in range(10) for k in range(6)]
    flat_s = torch.randn(2, 700)
    flat_s[:, idx] = 7.777
    out_s = policy.actor_module.forward(flat_s)
    flat_z = flat_s.clone()
    flat_z[:, idx] = 0.0
    out_z = policy.actor_module.forward(flat_z)
    d = (out_s - torch.nn.functional.linear(flat_s, lin.weight, lin.bias))
    e = (out_z - torch.nn.functional.linear(flat_z, lin.weight, lin.bias))
    check("ori-blind sentinel (residual path; leak would be ~7.8)",
          float((d - e).abs().max()) <= 1e-5, "max|diff|=%g" % float((d - e).abs().max()))
    try:
        policy.actor_module.forward(torch.randn(2, 641))
        check("junk width rejected", False)
    except ValueError:
        check("junk width rejected", True)


class _MotionLib:
    _motion_data_keys = []


class _Command:
    def __init__(self, pos, heading):
        self.robot_anchor_pos_w = pos
        self.anchor_heading_quat = heading
        self.motion_lib = _MotionLib()
        self.motion_ids = [0, 0]
        self.anchor_pos_w = pos
        self.anchor_ori_w = heading
        self.anchor_lin_vel_w = torch.zeros_like(pos)
        self.prev_anchor_pos_w = pos.clone()


def test_obs_layout_nonzero():
    """Contract A1: injected slot = [64 base | 6 channels] x10 slot-major.
    Evidence: e53_trainer.py L103-114 (view(n_slots, base_per_slot)+zeros interleave),
    E54 2432 incident (30 = 6x5 slots). Nonzero channels - zero-only checks are vacuous."""
    import e54_obs_terms as E54
    import e55_obs_terms as E55
    env = _Env()
    command = _Command(torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                       torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))
    E54.TRACKING_COMMAND_CLASS = type(command)
    env.command_manager = type("_CM", (), {"_terms": {"motion": command}})()
    base = torch.randn(2, 10, 6)   # vendor native: (B,10,6) ori, g1 stays 640
    stub_pkg = types.ModuleType("gear_sonic.envs.manager_env.mdp")
    stub_obs = types.ModuleType("gear_sonic.envs.manager_env.mdp.observations")
    def _stub_ori(env, command_name="motion", non_flatten=True):
        E55.CHANNEL_CELL["vec"] = torch.randn(2, 6)
        return base
    stub_obs.motion_anchor_ori_b = _stub_ori
    stub_pkg.observations = stub_obs
    sys.modules["gear_sonic.envs.manager_env.mdp"] = stub_pkg
    sys.modules["gear_sonic.envs.manager_env.mdp.observations"] = stub_obs
    os.environ["E54_CHANNEL_STATS"] = os.path.join(OPS, "data/experiments/a3_zhy_channel_probe_20261019_E54/e54_channel_stats.json")
    E54._MANIFEST = None
    E54._TENSOR_CACHE.clear()
    import e55_protocol as P
    manifest = json.loads(open(os.environ["E54_CHANNEL_STATS"]).read())
    _MotionLib._motion_data_keys = [sorted(manifest.keys())[0] + ".csv"]
    env.episode_length_buf = torch.tensor([1, 1], dtype=torch.long)
    E55.e55_root_channels_a0(env)   # native passthrough + stash
    check("native passthrough width (10,6)", E55.e55_root_channels_a0(env).shape == (2, 10, 6))
    env.episode_length_buf = torch.tensor([2, 2], dtype=torch.long)
    command.robot_anchor_pos_w = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    command.anchor_pos_w = command.robot_anchor_pos_w * 2
    E55.e55_root_channels_a0(env)
    check("stash cell filled (nonzero at step2)", float(E55.CHANNEL_CELL["vec"].abs().max()) > 0)
    for name in ("p", "q"):
        fn = getattr(E55, "e55_root_channels_" + name)
        check(name + " native passthrough", fn(env).shape == (2, 10, 6))


class _Env:
    episode_length_buf = torch.zeros(2, dtype=torch.long)
    step_dt = 0.02


def test_parity():
    import e54_obs_terms as E54
    from e55_channels import entry_channels
    torch.manual_seed(7)
    steps = 12
    pos = torch.cumsum(torch.randn(steps, 2, 3) * 0.01, dim=0)
    heading = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(steps, 2, 4)
    command = _Command(pos[-1], heading[-1])
    E54.TRACKING_COMMAND_CLASS = type(command)
    env = _Env()
    env.command_manager = type("_CM", (), {"_terms": {"motion": command}})()
    worst = 0.0
    for step in range(steps):
        env.episode_length_buf = torch.tensor([step], dtype=torch.long).expand(2)
        command.robot_anchor_pos_w = pos[step]
        command.anchor_heading_quat = heading[step]
        raw = E54._raw_channels(command, env)
        entry = entry_channels(pos[step], pos[step - 1] if step else pos[0], heading[step],
                               heading[step], dt=0.02, origin=pos[0],
                               prev=pos[step - 1] if step else pos[0])
        worst = max(worst, float((raw - entry).abs().max()))
    check("channel formula parity (contract: <=1e-6)", worst <= 1e-6, "max|diff|=%g" % worst)


def test_entry_build_paths():
    from pathlib import Path
    import e55_frozen_entry as entry
    state = {"e55_head.trunk.0.weight": torch.randn(256, 64), "e55_head.trunk.0.bias": torch.randn(256),
             "e55_head.trunk.2.weight": torch.randn(256, 256), "e55_head.trunk.2.bias": torch.randn(256),
             "e55_head.head.weight": torch.zeros(29, 256), "e55_head.head.bias": torch.zeros(29),
             "actor.enc.weight": torch.randn(4, 4)}
    head = entry._build_head([k for k in state if k.startswith("e55_head.")], state, torch.device("cpu"))
    check("entry builds head from e55_head.*", head is not None)
    check("entry head last layer zero", float(head.head.weight.abs().max()) == 0.0)
    check("entry bare path -> None", entry._build_head([], {"a": 1}, torch.device("cpu")) is None)
    bad = dict(state, **{"e55_head.junk": {"not": "a tensor"}})
    try:
        entry._build_head(["e55_head.junk"], bad, torch.device("cpu"))
        check("entry whitelist rejects non-tensor", False)
    except ValueError as exc:
        check("entry whitelist rejects non-tensor", "expected keys" in str(exc))
    try:
        entry._build_head(["e55_head.missing"], state, torch.device("cpu"))
        check("entry missing key raises with diff", False)
    except ValueError as exc:
        check("entry missing key raises with diff", "expected keys ['missing']" in str(exc))


def test_stats_and_bands():
    import e55_protocol as P
    from script.a3.checkpoint_store import sha256
    reg = json.loads(P.E54_REG.read_text())
    full = sha256(P.STATS)
    check("stats sha pinned (A1: actual registration file; conflict source=d34df4e was the E54 commit tag)",
          reg.get("stats_sha256") == full and full.startswith("ed1867b8"), full[:12])
    check("grouped bands v1.2 literals", P.BAND_PERJOINT_CLOTH == 0.281 and P.BAND_PERJOINT_OFFICIAL == 0.340)


def _cand(Z, H, N, c8, c64, wdr_ok=True):
    z_wdr, h_wdr, pj = (1.2, 1.0, 0.25) if wdr_ok else (1.9, 1.9, 0.9)
    return dict(shutter={"cloth": dict(wdr_max=h_wdr, per_joint_bff=pj),
                         "official": dict(wdr_max=z_wdr, per_joint_bff=pj)},
                cloth6={8: c8, 32: (c8 + c64) / 2, 64: c64},
                rank_metrics=dict(Z=Z, H=H, N=N))


def test_verdict_contract_derived():
    """From contract section 4 / B.2: (Z,H,-N) lexicographic; cloth6 only breaks key ties."""
    import e55_protocol as P
    a = _cand(Z=2, H=500, N=0, c8=0.10, c64=0.0900)
    b = _cand(Z=1, H=900, N=0, c8=0.10, c64=0.0882)   # keys DIFFER; cloth6 rel diff = 2.0% <= 2%
    passes, sel = P.verdict(dict(A0=a, P=b, Q=_cand(0, 0, 9, 0.1, 0.3, wdr_ok=False)), {})
    check("COUNTEREXAMPLE: key order beats cloth6 near-tie (no A0 flip)",
          sel["winner"] == "A0" and "lexicographic" in sel["rule"], str(sel))
    _, sel_c1b = P.verdict(dict(A0=_cand(1, 400, 0, 0.10, 0.0900), P=_cand(2, 100, 0, 0.10, 0.0882),
                                Q=_cand(0, 0, 9, 0.1, 0.3, False)), {})
    check("C1b: P key-better + cloth6 near-tie 2% -> P stays (no flip)",
          sel_c1b["winner"] == "P" and "lexicographic" in sel_c1b["rule"], str(sel_c1b))
    a2, b2 = _cand(2, 500, 0, 0.10, 0.09), _cand(2, 500, 0, 0.10, 0.0895)
    _, sel2 = P.verdict(dict(A0=a2, P=b2, Q=_cand(0, 0, 9, 0.1, 0.3, False)), {})
    check("key equal + cloth6 rel<=2% -> A0", sel2["winner"] == "A0", str(sel2))
    a3, b3 = _cand(2, 500, 0, 0.10, 0.09), _cand(2, 500, 0, 0.10, 0.05)
    _, sel3 = P.verdict(dict(A0=a3, P=b3, Q=_cand(0, 0, 9, 0.1, 0.3, False)), {})
    check("key equal + cloth6 rel>2% -> smaller cloth6 wins", sel3["winner"] == "P", str(sel3))
    live = {"P": "ACTIVE"}
    _, sel4 = P.verdict(dict(A0=_cand(1, 1, 5, 0.1, 0.09), P=_cand(2, 100, 0, 0.1, 0.095),
                             Q=_cand(0, 0, 9, 0.1, 0.3, False)), live)
    check("winner ACTIVE -> CLEAN conclusion", sel4["winner_conclusion"] == "CLEAN", str(sel4))
    _, sel5 = P.verdict(dict(A0=_cand(1, 1, 5, 0.1, 0.09), P=_cand(2, 100, 0, 0.1, 0.095),
                             Q=_cand(0, 0, 9, 0.1, 0.3, False)), {})
    check("liveness missing -> PENDING, no conclusion", sel5["winner_conclusion"] == "PENDING", str(sel5))
    _, sel6 = P.verdict(dict(A0=_cand(1, 1, 5, 0.1, 0.09), P=_cand(2, 100, 0, 0.1, 0.095),
                             Q=_cand(0, 0, 9, 0.1, 0.3, False)), {"P": "CHANNEL_INERT"})
    check("CHANNEL_INERT -> INCONCLUSIVE field", sel6["winner_conclusion"] == "INCONCLUSIVE", str(sel6))


def main():
    test_head_basics()
    test_install_order_and_ori_blind()
    test_obs_layout_nonzero()
    test_parity()
    test_entry_build_paths()
    test_stats_and_bands()
    test_verdict_contract_derived()
    if FAILURES:
        raise SystemExit("E55 controls FAILED: " + ", ".join(FAILURES))
    print("controls all pass")


if __name__ == "__main__":
    main()
