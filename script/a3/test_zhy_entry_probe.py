"""The entry gate must distinguish a changed input from a changed output."""

import unittest
import torch

from script.a3.zhy_entry_probe import evaluate_capture


def example():
    mean = torch.zeros(4, 3, 2)
    std = torch.ones_like(mean)
    obs = {'actor_obs':torch.zeros(4, 3, 5)}
    state = dict(parameters='weights', buffers='buffers')
    record = dict(inputs={key:value.clone() for key,value in obs.items()}, mean=mean.clone(),
        flags=dict(matmul_tf32=False, cudnn_tf32=False), input_metadata={}, options={})
    return dict(indices=torch.arange(4), entry=record, wrapper_mean=mean.clone(), wrapper_std=std,
        mean_at_loss_entry=mean.clone(), batch_input_before=obs,
        batch_input_after={key:value.clone() for key,value in obs.items()},
        live_input_after={key:value.clone() for key,value in obs.items()},
        rollout=dict(all_obs_dict=obs, old_mu_batch=mean.clone(), old_sigma_batch=std,
            actions=mean.clone(), logprobs=torch.distributions.Normal(mean,std).log_prob(mean).sum(-1)),
        actor_is_sampling_actor=True, rollout_actor_state=dict(state),
        wrapper_actor_state_before=dict(state), wrapper_actor_state_after=dict(state),
        actor_state_after_replays=dict(state), accelerator_mixed_precision='no',
        parameter_changes_from_load=[], replays=dict(same_live_inputs_and_options=mean.clone(),
            stored_inputs_same_options=mean.clone(), stored_inputs_without_episode_mask=mean.clone()))


class EntryEvidenceTests(unittest.TestCase):
    def test_exact_input_and_distribution_pass(self):
        capture = example()
        capture['entry'].update(state_before=dict(capture['rollout_actor_state']),
                                state_after=dict(capture['rollout_actor_state']))
        self.assertTrue(evaluate_capture(capture)['probability_contract_passed'])

    def test_changed_input_is_not_accepted_even_with_equal_outputs(self):
        capture = example()
        capture['entry'].update(state_before=dict(capture['rollout_actor_state']),
                                state_after=dict(capture['rollout_actor_state']))
        capture['entry']['inputs']['actor_obs'][0,0,0] = 1
        result = evaluate_capture(capture)
        self.assertFalse(result['flags']['actual_input_matches_stored'])
        self.assertFalse(result['probability_contract_passed'])

    def test_stable_weights_do_not_hide_distribution_mismatch(self):
        capture = example()
        capture['entry'].update(state_before=dict(capture['rollout_actor_state']),
                                state_after=dict(capture['rollout_actor_state']))
        capture['wrapper_mean'].add_(1)
        result = evaluate_capture(capture)
        self.assertTrue(result['flags']['actor_parameters_unchanged'])
        self.assertFalse(result['flags']['raw_actor_matches_wrapper'])
        self.assertFalse(result['probability_contract_passed'])


if __name__ == '__main__':
    unittest.main()
