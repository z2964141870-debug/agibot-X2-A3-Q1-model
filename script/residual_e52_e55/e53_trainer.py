"""E53 trainers: plain passthrough and injected (anchor-term channel concat)."""

import torch

from script.a3.cloth_constrained_trainer import ClothConstrainedTrainer, parameter_digest
from script.a3.cloth_curve_trainer import ClothCurveTrainer

from e53_protocol import CHANNELS, CHANNEL_GROUP, G1_ENCODER_NAME, G1_ENCODER_FIRST_LAYER_IN


class E53RecordOnlyDriftMixin:
    """E53 amendment: the E-series 0.05/0.5 anchor drift budget becomes
    record-only. E52 established the interception gate does not protect task
    performance and the lead mandated teacher MSE record-only for E53; plain
    AdamW crosses 0.05 near optimizer step 23 (E52 arm-A phenomenon, now a
    fourth reproduction), so interception would make both arms untrainable at
    tier 8/32. Frozen-parameter/buffer integrity checks are kept."""

    def _after_optimizer_step(self, optimizer, args, kwargs):
        super(ClothConstrainedTrainer, self)._after_optimizer_step(optimizer, args, kwargs)
        self._assert_frozen()
        policy = self.accelerator.unwrap_model(self.model).policy
        with torch.no_grad():
            means = torch.cat([policy.forward({n: v[i:i + 4] for n, v in self._cloth_obs.items()},
                                              is_training=False) for i in range(0, 16, 4)], dim=0)
        difference = means - self._cloth_targets
        row = dict(optimizer_step=self._optimizer_step_ordinal,
                   teacher_action_rmse=float(difference.square().mean().sqrt()),
                   teacher_action_max_abs=float(difference.abs().max()),
                   retention_loss_before_step=self._cloth_last_retention_loss,
                   frozen_parameter_digest=parameter_digest(self._cloth_frozen))
        self._cloth_records.append(row)
        self._save_cloth_receipt()


class E53PlainTrainer(E53RecordOnlyDriftMixin, ClothCurveTrainer):
    """Arm N: identical vendor warm-start flow; kept explicit for registration."""

    pass


class E53InjectedTrainer(E53RecordOnlyDriftMixin, ClothCurveTrainer):
    """Arm I: the tokenizer term motion_anchor_ori_b_mf_nonflat is overridden to
    e53_obs_terms.e53_anchor_ori_b_mf_with_channels (vendor anchor block + the
    6 root channels repeated across the 10 future slots), so each g1 encoder
    slot is 580 + 120 = 700 wide and the first Linear auto-builds at 700 input
    columns. The 035 warm start (inside PPOTrainer.__init__, ppo_trainer.py
    line 442) carries a (2048, 640) weight; this override wraps torch.load
    during the chained load to right-pad that checkpoint weight with 60 zero
    columns before the vendor load, so live and source stay bit-equal (the
    Finetune exact-weights guard holds) and the appended columns are exactly
    zero: the injected channels are silent at t=0 regardless of slot layout
    (0 x 0 contributions are exact in fp32). The load-time anchor identity
    check replays the E11 rollout obs dict (flat tokenizer 2402 while the arm
    I env expects 2462); the policy forward is wrapped to insert the 60 zero
    channels at the anchor block tail (anchor_start + anchor_width - delta)
    so the actor parses it and every downstream slice stays aligned."""

    def load_checkpoint(self, *args, **kwargs):
        policy = self._e53_actor_module()
        name, layer, built_in = self._e53_g1_first_linear(policy)
        if built_in <= G1_ENCODER_FIRST_LAYER_IN:
            raise ValueError(
                "E53 injection: g1 first Linear in_features %d does not include the channel block"
                % built_in)
        delta = built_in - G1_ENCODER_FIRST_LAYER_IN
        if delta % CHANNELS != 0:
            raise ValueError(
                "E53 injection: channel block %d not a multiple of %d" % (delta, CHANNELS))
        weight_key = None
        for key in policy.state_dict().keys():
            if key.endswith("encoders." + G1_ENCODER_NAME + "." + name + ".weight"):
                weight_key = key
                break
        if weight_key is None:
            raise ValueError("E53 injection: g1 first Linear weight key not found")
        decoder_last = None
        g1_kin = policy.get_submodule("actor_module.decoders.g1_kin")
        for d_name, d_module in g1_kin.named_modules():
            if isinstance(d_module, torch.nn.Linear):
                decoder_last = d_name
        decoder_weight_key = "actor_module.decoders.g1_kin." + decoder_last + ".weight"
        decoder_bias_key = "actor_module.decoders.g1_kin." + decoder_last + ".bias"

        anchor_start = anchor_width = None
        spec_total = 0
        for spec_name, start, end, _dims in policy.actor_module.tokenizer_obs_specs:
            spec_total += end - start
            if spec_name == "motion_anchor_ori_b_mf_nonflat":
                anchor_start, anchor_width = start, end - start
        if anchor_start is None:
            raise ValueError("E53 injection: anchor term spec not found in tokenizer specs")
        if anchor_width <= delta:
            raise ValueError(
                "E53 injection: anchor spec width %d does not include the channel block" % anchor_width)
        expected_flat = spec_total

        orig_torch_load = torch.load

        n_slots = delta // CHANNELS
        base_per_slot = G1_ENCODER_FIRST_LAYER_IN // n_slots

        def _interleave_cols(weight):
            out_features = weight.shape[0]
            return torch.cat(
                [weight.view(out_features, n_slots, base_per_slot),
                 weight.new_zeros(out_features, n_slots, CHANNELS)], dim=2).view(
                out_features, built_in)

        def _interleave_rows(weight):
            in_features = weight.shape[1]
            return torch.cat(
                [weight.view(n_slots, base_per_slot, in_features),
                 weight.new_zeros(n_slots, CHANNELS, in_features)], dim=1).view(
                built_in, in_features)

        def _e53_padded_torch_load(*t_args, **t_kwargs):
            loaded = orig_torch_load(*t_args, **t_kwargs)
            if isinstance(loaded, dict) and "policy_state_dict" in loaded and "state" in loaded:
                source = loaded["policy_state_dict"]
                weight = source.get(weight_key)
                if weight is not None and weight.dim() == 2 and weight.shape[1] == G1_ENCODER_FIRST_LAYER_IN:
                    source[weight_key] = _interleave_cols(weight)
                d_weight = source.get(decoder_weight_key)
                if d_weight is not None and d_weight.dim() == 2 and d_weight.shape[0] == G1_ENCODER_FIRST_LAYER_IN:
                    source[decoder_weight_key] = _interleave_rows(d_weight)
                    d_bias = source.get(decoder_bias_key)
                    if d_bias is not None and d_bias.shape[0] == G1_ENCODER_FIRST_LAYER_IN:
                        source[decoder_bias_key] = torch.cat(
                            [d_bias.view(n_slots, base_per_slot),
                             d_bias.new_zeros(n_slots, CHANNELS)], dim=1).reshape(built_in)
            return loaded

        torch.load = _e53_padded_torch_load
        actor_module = policy.actor_module
        orig_forward = actor_module.forward
        orig_fwt = actor_module.forward_with_external_tokens
        slots = delta // CHANNELS
        per_orig = (anchor_width - delta) // slots
        probe_state = {"n": 0}

        def _e53_pad_tokenizer(tok):
            if tok.shape[-1] != expected_flat - delta:
                return tok
            pieces = []
            for k in range(slots):
                pieces.append(tok[..., anchor_start + k * per_orig:
                                  anchor_start + (k + 1) * per_orig])
                pieces.append(tok.new_zeros(*tok.shape[:-1], CHANNELS))
            return torch.cat(
                [tok[..., :anchor_start]] + pieces +
                [tok[..., anchor_start + (anchor_width - delta):]], dim=-1)

        def _e53_pad_obs(obs):
            if isinstance(obs, dict) and "tokenizer" in obs:
                tok = obs["tokenizer"]
                if isinstance(tok, torch.Tensor) and tok.shape[-1] == expected_flat - delta:
                    patched = dict(obs)
                    patched["tokenizer"] = _e53_pad_tokenizer(tok)
                    if probe_state["n"] < 8:
                        probe_state["n"] += 1
                        print("E53 DBG pad#%d width %d->%d" %
                              (probe_state["n"], int(tok.shape[-1]),
                               int(patched["tokenizer"].shape[-1])), flush=True)
                    return patched
            return obs

        def _e53_padded_forward(net_input, *f_args, **f_kwargs):
            if isinstance(net_input, torch.Tensor):
                net_input = _e53_pad_tokenizer(net_input)
            elif isinstance(net_input, dict):
                net_input = _e53_pad_obs(net_input)
            return orig_forward(net_input, *f_args, **f_kwargs)

        def _e53_padded_fwt(obs, *f_args, **f_kwargs):
            return orig_fwt(_e53_pad_obs(obs), *f_args, **f_kwargs)

        def _probe_hook(module, inputs):
            x = inputs[0]
            print("E53 DBG lin", tuple(x.shape), flush=True)

        handle = layer.register_forward_pre_hook(_probe_hook)
        actor_module.forward = _e53_padded_forward
        actor_module.forward_with_external_tokens = _e53_padded_fwt
        try:
            result = super().load_checkpoint(*args, **kwargs)
        finally:
            handle.remove()
            torch.load = orig_torch_load
        # The padding wrappers stay installed for the whole run: the cloth
        # retention/anchor replays feed the stored E11 obs dict (2402-wide)
        # through the live 2462-spec module at every optimizer step. The
        # wrappers are width-gated (2402 only) so live 2462 observations pass
        # through untouched.
        with torch.no_grad():
            w = layer.weight.detach()
            slot_tails = w.view(w.shape[0], n_slots, base_per_slot + CHANNELS)[
                :, :, base_per_slot:].abs().max().item()
        if slot_tails != 0.0:
            raise ValueError("E53 injection: appended weight columns are not zero (%g)" % slot_tails)
        self._e53_extended = True
        self._e53_touched = ["actor_module.encoders." + G1_ENCODER_NAME + "." + name + ".weight"]
        self._e53_pad_wrappers = (orig_forward, orig_fwt)
        print("E53 injection padded g1 first Linear weight %d->%d with zero columns (%s); "
              "anchor block at %d width %d, per-slot zero insert (%d slots), flat %d->%d; "
              "padding wrappers stay installed"
              % (G1_ENCODER_FIRST_LAYER_IN, built_in, name, anchor_start, anchor_width,
                 delta // CHANNELS, expected_flat - delta, expected_flat), flush=True)
        if hasattr(self, "_assert_frozen"):
            self._assert_frozen()
        return result

    def _e53_actor_module(self):
        return self.accelerator.unwrap_model(self.model).policy

    def _e53_g1_first_linear(self, actor):
        encoder = actor.get_submodule("actor_module.encoders." + G1_ENCODER_NAME)
        for name, module in encoder.named_modules():
            if isinstance(module, torch.nn.Linear):
                return name, module, module.in_features
        raise ValueError("E53 injection: g1 encoder has no Linear layer")
