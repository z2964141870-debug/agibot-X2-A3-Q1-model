"""Saved-tensor probability checks without importing any training module."""

def compare(mean, old_mean, std, old_std, actions, old_logprobs):
    import torch
    delta = mean - old_mean
    logprobs = torch.distributions.Normal(mean, std).log_prob(actions).sum(-1)
    ratio = (logprobs - old_logprobs).exp()
    if not torch.isfinite(ratio).all() or not torch.isfinite(delta).all():
        raise ValueError("Nonfinite replay")
    return dict(action_rmse=delta.square().mean().sqrt().item(), action_max_abs=delta.abs().max().item(),
                changed_frames_over_1e_4=(delta.abs().amax(-1) > 1e-4).sum().item(),
                ratio_mean=ratio.mean().item(), ratio_min=ratio.min().item(), ratio_max=ratio.max().item(),
                max_abs_ratio_minus_one=(ratio-1).abs().max().item())


def initial_probability(result, batch):
    valid = ~batch['mb_padding_mask']
    if not valid.any():
        raise ValueError('Initial PPO batch has no valid transitions')
    record = compare(result['action_mean'][valid], batch['mb_old_mu'][valid],
        result['action_std'][valid], batch['mb_old_sigma'][valid],
        batch['mb_actions'][valid], batch['mb_logprobs'][valid])
    if record['action_rmse'] > 1e-5 or record['max_abs_ratio_minus_one'] > .001:
        raise ValueError('Initial clothing likelihood contract failed: ' + repr(record))
    return record

