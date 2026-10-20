"""E55 MuJoCo eval entry: bare-035 path plus the residual head from e55_head.* keys.

Launch word-swap target (E55LaunchProxy). Contract:
- the vendor encoder path is untouched; the head is PURELY ADDITIVE;
- with the residual forced to zero the policy must equal bare 035 bit-for-bit
  at the same frames (eval t=0 identity);
- channels are computed entry-side with the e54_obs_terms math on
  (reference, robot anchor state); parity vs the training-side math is pinned
  by test_e55_controls fixture (max abs diff <= 1e-6);
- YUANQI_E55_VARIANT in {normal, zero_channels, zero_head} drives the liveness
  shutters; under normal a missing provider REFUSES to run (no silent zeros).
"""

import os
import sys

import json
import numpy as np
import torch


def main():
    from script.a3.reference_contract import load_sim
    sim = load_sim()
    variant = os.environ.get("YUANQI_E55_VARIANT", "normal")
    original_policy = sim.A3Policy

    class E55Policy(original_policy):
        def __init__(self, checkpoint, device, encoder="a3_fast"):
            state = _payload_state(checkpoint)
            head_keys = [k for k in state.keys() if k.startswith("e55_head.")]
            self.e55_variant = variant
            import sys as _sys
            super().__init__(checkpoint, device, encoder=encoder)   # v1.2 native-640: untouched
            self.e55_head = _build_head(head_keys, state, device)
            if self.e55_head is None:
                print("[e55-entry] ENTRY_VARIANT=bare_035 (no e55_head.* keys)", flush=True)
                if variant == "normal":
                    raise ValueError("E55 entry: no e55_head.* in checkpoint; refusing silent bare-035")
            if variant != "normal":
                print("[e55-entry] ENTRY_VARIANT=" + variant, flush=True)

        def act(self, encoder_input, actor_obs, device):
            if self.e55_variant == "normal" and not _channels_for.installed:
                raise ValueError("E55 entry: channel provider not installed; wire the loop hook first")
            enc = torch.as_tensor(np.asarray(encoder_input)).reshape(1, -1)
            frames = enc.reshape(1, 10, -1)
            if frames.shape[-1] != 64:
                raise ValueError("E55 entry: frame width %d != 64" % frames.shape[-1])
            channels = _channels_for.current()
            if self.e55_head is not None and self.e55_variant != "zero_head":
                if channels is None:
                    raise ValueError("E55 entry: provider produced no channel vector this step")
            action = super().act(encoder_input, actor_obs, device)
            if self.e55_head is None or self.e55_variant == "zero_head":
                return action
            if channels is None:
                raise ValueError("E55 entry: provider produced no channel vector this step")
            chan_vec = torch.as_tensor(np.asarray(channels), dtype=torch.float32, device=device)
            if not torch.isfinite(chan_vec).all():
                raise ValueError("E55 entry: non-finite channels from provider")
            if self.e55_variant == "zero_channels":
                chan_vec = torch.zeros_like(chan_vec)
            anchor = frames[:, :, :58].to(dtype=torch.float32, device=device)
            chan = chan_vec.view(1, 1, 6).expand(1, 10, 6)
            if isinstance(action, torch.Tensor):
                base = action.detach().to(dtype=torch.float32, device=device)
            else:
                base = torch.as_tensor(np.asarray(action), dtype=torch.float32, device=device)
            adjusted = base + self.e55_head(anchor, chan)
            return adjusted.detach().cpu().numpy().reshape(-1)

    sim.A3Policy = E55Policy

    loop_cls = next(cls for cls in vars(sim).values()
                    if isinstance(cls, type) and "_step_policy" in vars(cls))
    body_quat = sim.body_quat_wxyz
    st = {"prev": None, "origin": None, "last_frame": None}

    def _step_channels(self, ref_frame, *args, **kwargs):
        anchor_id = self.runtime.mapping.anchor_body_id
        pos = np.asarray(self.data.xpos[anchor_id], dtype=np.float64).copy()
        restart = st["origin"] is None or (st["last_frame"] is not None and ref_frame < st["last_frame"])
        if restart:
            st["origin"], st["prev"] = pos.copy(), pos.copy()
        quat = body_quat(self.data, anchor_id)
        w, x, y, z = quat
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        disp = rot.T @ (pos - st["origin"])
        vel_a = rot.T @ ((pos - st["prev"]) / 0.02)
        channels = np.concatenate([disp, vel_a]).astype(np.float32)
        _channels_for.install(lambda: channels)
        if os.environ.get("YUANQI_E55_SMOKE"):
            with open(os.environ["YUANQI_E55_SMOKE"], "a") as handle:
                handle.write(json.dumps({"ref_frame": int(ref_frame), "channels": channels.tolist()}) + "\n")
        st["prev"] = pos.copy()
        st["last_frame"] = ref_frame
        return original_step(self, ref_frame, *args, **kwargs)

    original_step = loop_cls._step_policy
    loop_cls._step_policy = _step_channels
    rest = sys.argv[1:]
    if "--" in rest:
        i = rest.index("--")
        rest = rest[:i] + rest[i + 1:]
    drop, out = {"--capture", "--capture-count", "--reference-buffer-ms", "--buffer-trace",
                 "--joint-reference-mode", "--joint-noise-std", "--joint-noise-seed",
                 "--joint-trace", "--joint-predictor", "--causal-orientation"}, []
    skip = False
    for word in rest:
        if skip:
            skip = False
            continue
        if word in drop:
            skip = not word.startswith("--causal")
            continue
        out.append(word)
    sys.argv = [str(sim.__file__), *out]
    return sim.main()


def _payload_state(checkpoint):
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = state.get("policy_state_dict", state) if isinstance(state, dict) else state
    if not isinstance(state, dict):
        raise ValueError("E55 entry: checkpoint payload is not a state dict")
    return state


def _build_head(head_keys, state, device):
    """Whitelist: only plain tensor arrays under e55_head.* are accepted."""
    if not head_keys:
        return None
    try:
        flat = {k[len("e55_head."):]: state[k] for k in head_keys}
        for key, value in flat.items():
            if not isinstance(value, (np.ndarray, torch.Tensor, list)):
                raise ValueError("non-tensor payload under e55_head.%s: %r" % (key, type(value)))
        from e55_residual import E55ResidualHead
        head = E55ResidualHead()
        head.load_state_dict({k: torch.as_tensor(np.asarray(v), dtype=torch.float32)
                              for k, v in flat.items()})
        print("[e55-entry] ENTRY_VARIANT=normal (head rebuilt from %d tensors)" % len(flat), flush=True)
        return head.to(device).eval()
    except Exception as exc:
        expected = sorted(k[len("e55_head."):] for k in head_keys)
        raise ValueError("E55 entry: head build FAILED; expected keys %s; cause: %s" % (expected, exc))


def install_channel_provider(fn):
    """Loop-hook seam (single entry): fn() returns the CURRENT-STEP 6-dim channel
    vector for ONE env (NOT a batch); dt is fixed 0.02 at the assembly site."""
    _channels_for.install(fn)


class _channels_for:
    step_fn = None
    installed = False

    @classmethod
    def current(cls):
        return cls.step_fn() if cls.step_fn is not None else None

    @classmethod
    def install(cls, fn):
        cls.step_fn = fn
        cls.installed = True




if __name__ == "__main__":
    raise SystemExit(main())
