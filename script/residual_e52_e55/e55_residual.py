"""E55 residual head over the E53 injected geometry.

Flat tokenizer = 2462: anchor group starts at anchor_start, 10 slots x 70 wide
(58 future q/qd + 6 ori + 6 arm channels). The head consumes per slot the first
58 dims plus the trailing 6 channel dims, shares an MLP across slots, sums the
slot outputs, and adds a zero-initialized residual on the actor output. Inputs
arriving at the native 2402 width (E11 guard replays) are padded to the
injected geometry with the SAME interleave the E53 wrapper uses, so the guard
sees the zero-channel form and the head stays silent at t=0.

Capacity discipline (lead-ratified): the head only sees per-slot reference
conditioning + channels - never the full observation.
"""

import os
import torch
from torch import nn


class E55ResidualHead(nn.Module):
    def __init__(self, slot_anchor=58, channels=6, hidden=(256, 256), out_dim=29):
        super().__init__()
        self.slot_anchor = int(slot_anchor)
        self.channels = int(channels)
        layers = []
        prev = self.slot_anchor + self.channels
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ELU()]
            prev = width
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, out_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, anchor_slots, channels):
        # anchor_slots: (B, S, 58); channels: (B, S, 6).
        if anchor_slots.shape[-1] != self.slot_anchor or channels.shape[-1] != self.channels:
            raise ValueError("E55 head input geometry mismatch: %s/%s"
                             % (tuple(anchor_slots.shape), tuple(channels.shape)))
        x = torch.cat([anchor_slots, channels], dim=-1)
        return self.head(self.trunk(x).sum(dim=1))


def install_injected(policy, anchor_start, slots=10, slot_anchor=58, ori_width=6, channels=6,
                     hidden=(256, 256), out_dim=29, flat_base=2402, cmd_start=None, mf_start=None,
                     cmd_group_name="command_multi_future_nonflat"):
    """Specs mode (cmd_start+mf_start given): the tokenizer group already carries the
    channels (E53 L88 mirror, native 2402 with g1 frame 70) - no zeros injection; the
    head extracts per-slot [58 cmd | 6 chan] by term start offsets and adds residual."""
    if hasattr(policy, "e55_head"):
        raise ValueError("E55 head already installed")
    if cmd_start is not None:
        import e55_obs_terms as _cellmod
        cmd_group_name_local = cmd_group_name
        head = E55ResidualHead(slot_anchor=slot_anchor, channels=channels, hidden=hidden, out_dim=out_dim)
        policy.add_module("e55_head", head)
        actor = policy.actor_module
        original_forward = actor.forward
        cmd_group_name = cmd_group_name_local
        def _rows(net_input):
            """Return (rows, base_offset): rows=(N,width) and slot-0 offset in that row
            space. net_input.tokenizer is (B,T,2402) with global offsets; a group dict
            (future-proofing) uses the cmd group at offset 0."""
            tok = net_input
            if not isinstance(tok, torch.Tensor):
                got = None
                try:
                    got = tok["tokenizer"]
                except Exception:
                    got = None
                if got is None and hasattr(tok, "get"):
                    try:
                        got = tok.get("tokenizer")
                    except Exception:
                        got = None
                tok = got if got is not None else tok
            if isinstance(tok, dict) or tok.__class__.__name__ == "TensorDict":
                try:
                    g = tok[cmd_group_name]
                    return g.reshape(-1, g.shape[-1]), 0
                except Exception:
                    pass
            if not isinstance(tok, torch.Tensor):
                raise ValueError("E55 head: tokenizer not a tensor: tok_type=%s top_keys=%s" % (
                    type(tok).__name__,
                    list(tok.keys()) if isinstance(tok, dict) else str(tok)[:120]))
            if tok.dim() == 1:
                tok = tok.unsqueeze(0)
            if tok.dim() < 2:
                raise ValueError("E55 head: tokenizer tensor too flat")
            return tok.reshape(-1, tok.shape[-1]), cmd_start
        def wrapped_forward(net_input, *args, **kwargs):
            flat, base = _rows(net_input)
            anchor = torch.stack([flat[..., base + s * slot_anchor:
                                            base + (s + 1) * slot_anchor] for s in range(slots)], dim=1)
            action = original_forward(net_input, *args, **kwargs)
            is_container = isinstance(action, dict) or action.__class__.__name__ == "TensorDict"
            if is_container:
                try:
                    a_t = action["action"]
                except Exception:
                    a_t = next(v for v in (action.values() if hasattr(action, "values") else []) if isinstance(v, torch.Tensor))
            else:
                a_t = action
            batch, tokens = (a_t.shape[0], a_t.shape[1]) if a_t.dim() >= 3 else (a_t.shape[0], 1)
            vec = _cellmod.CHANNEL_CELL.get("vec")
            if vec is None or vec.shape[0] != batch:
                # batch-safe fallback (identity replay / synthetic forwards): zero channels
                # keep the base path bit-equal; head is zero-init so residual adds 0
                chans = flat.new_zeros(flat.shape[0], slots, channels)
            else:
                chans = vec.to(flat.device).to(flat.dtype).unsqueeze(1).unsqueeze(1) \
                    .expand(batch, tokens, slots, channels).reshape(flat.shape[0], slots, channels)
            delta = head(anchor, chans).view(a_t.shape)
            if is_container:
                out = dict(action) if isinstance(action, dict) else action
                out["action"] = a_t + delta
                return out
            return a_t + delta
        def _compose_delta(a_t, anchor, chan):
            delta = head(anchor, chan).view(a_t.shape)
            return delta
        orig_fwet = actor.forward_with_external_tokens
        print("[e55-fwet] installed actor_id=%d cls=%s" % (id(actor), type(actor).__name__), flush=True)
        def wrapped_fwet(input_data, external_tokens, *args, **kwargs):
            wrapped_fwet._e55_wrapped = True
            if os.environ.get("YUANQI_E55_PROBE") and not getattr(wrapped_fwet, "_enter", False):
                wrapped_fwet._enter = True
                print("[e55-fwet] ENTERED", flush=True)
            out = orig_fwet(input_data, external_tokens, *args, **kwargs)
            is_c = isinstance(out, dict) or out.__class__.__name__ == "TensorDict"
            a_t = out["action"] if is_c else out
            if not isinstance(a_t, torch.Tensor):
                return out
            tok = None
            if isinstance(input_data, dict):
                tok = input_data.get("tokenizer")
            if not isinstance(tok, torch.Tensor):
                if os.environ.get("YUANQI_E55_PROBE") and not getattr(wrapped_fwet, "_probed", False):
                    wrapped_fwet._probed = True
                    print("[e55-fwet] no tokenizer tensor: top=%s" % (list(input_data.keys()) if isinstance(input_data, dict) else type(input_data).__name__), flush=True)
                return out
            flat = tok.reshape(-1, tok.shape[-1])
            anchor = torch.stack([flat[..., cmd_start + s * slot_anchor:
                                            cmd_start + (s + 1) * slot_anchor] for s in range(slots)], dim=1)
            batch, tokens = (a_t.shape[0], a_t.shape[1]) if a_t.dim() >= 3 else (a_t.shape[0], 1)
            vec = _cellmod.CHANNEL_CELL.get("vec")
            if vec is None or vec.shape[0] != batch:
                chans = flat.new_zeros(flat.shape[0], slots, channels)
            else:
                chans = vec.to(flat.device).to(flat.dtype).unsqueeze(1).unsqueeze(1) \
                    .expand(batch, tokens, slots, channels).reshape(flat.shape[0], slots, channels)
            delta = _compose_delta(a_t, anchor, chans)
            if os.environ.get("YUANQI_E55_PROBE") and not getattr(wrapped_fwet, "_probed", False):
                wrapped_fwet._probed = True
                print("[e55-fwet] anchor%s chan%s delta_norm=%s rows=%d at=%d" % (
                    tuple(anchor.shape), tuple(chans.shape), round(float(delta.float().norm()), 8),
                    flat.shape[0], cmd_start), flush=True)
            if is_c:
                out2 = dict(out) if isinstance(out, dict) else out
                out2["action"] = a_t + delta
                return out2
            return a_t + delta
        actor.forward_with_external_tokens = wrapped_fwet
        actor.forward = wrapped_forward
        return head
    if cmd_start is not None and mf_start is not None:
        head = E55ResidualHead(slot_anchor=slot_anchor, channels=channels, hidden=hidden, out_dim=out_dim)
        policy.add_module("e55_head", head)
        actor = policy.actor_module
        original_forward = actor.forward
        def wrapped_forward(net_input, *args, **kwargs):
            tensor = net_input if isinstance(net_input, torch.Tensor) else net_input.get("tokenizer")
            if tensor.dim() < 2:
                raise ValueError("E55 head: tokenizer tensor too flat")
            flat = tensor.reshape(-1, tensor.shape[-1])
            anchor = torch.stack([flat[..., cmd_start + s * slot_anchor:
                                            cmd_start + (s + 1) * slot_anchor] for s in range(slots)], dim=1)
            chans = torch.stack([flat[..., mf_start + s * (ori_width + channels) + ori_width:
                                            mf_start + (s + 1) * (ori_width + channels)] for s in range(slots)], dim=1)
            action = original_forward(net_input, *args, **kwargs)
            return action + head(anchor, chans)
        actor.forward = wrapped_forward
        return head
    base_flat = int(flat_base)
    injected_flat = base_flat + slots * channels                       # 2402 -> 2462 in production
    slot_width = slot_anchor + ori_width + channels                    # 70
    if base_flat != anchor_start + slots * (slot_anchor + ori_width):
        raise ValueError("E55: flat_base %d inconsistent with anchor geometry" % base_flat)
    head = E55ResidualHead(slot_anchor=slot_anchor, channels=channels, hidden=hidden, out_dim=out_dim)
    policy.add_module("e55_head", head)
    actor = policy.actor_module
    original_forward = actor.forward

    def _to_injected(flat):
        width = flat.shape[-1]
        if width == injected_flat:
            return flat
        if width != base_flat:
            raise ValueError("E55 head: flat tokenizer width %d unsupported" % width)
        pieces = [flat[..., :anchor_start]]
        for s in range(slots):
            lo = anchor_start + s * (slot_anchor + ori_width)
            pieces.append(flat[..., lo:lo + slot_anchor + ori_width])
            pieces.append(flat.new_zeros(*flat.shape[:-1], channels))
        return torch.cat(pieces, dim=-1)

    def wrapped_forward(net_input, *args, **kwargs):
        tensor = net_input if isinstance(net_input, torch.Tensor) else net_input.get("tokenizer")
        if tensor.dim() < 2:
            raise ValueError("E55 head: tokenizer tensor too flat")
        flat = tensor.reshape(-1, tensor.shape[-1])
        injected = _to_injected(flat)
        if not isinstance(net_input, torch.Tensor):
            patched = dict(net_input)
            patched["tokenizer"] = injected.reshape(*tensor.shape[:-1], injected.shape[-1])
            action = original_forward(patched, *args, **kwargs)
        else:
            action = original_forward(injected.reshape(*tensor.shape[:-1], injected.shape[-1]),
                                      *args, **kwargs)
        flat2 = injected
        anchor = torch.stack([flat2[..., anchor_start + s * slot_width:
                                        anchor_start + s * slot_width + slot_anchor]
                              for s in range(slots)], dim=1)
        chans = torch.stack([flat2[..., anchor_start + s * slot_width + slot_anchor + ori_width:
                                             anchor_start + (s + 1) * slot_width]
                             for s in range(slots)], dim=1)
        return action + head(anchor, chans)

    actor.forward = wrapped_forward
    return head
