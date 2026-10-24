"""Check action-domain equivalence and refusal before invalid optimizer steps."""

import copy
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from bounded_policy import ACTION_LIMIT, BoundedPolicy, bounded_action, reject_nonfinite_gradients
    from supermimic.policy.nets import Policy, PolicyCfg
    from supermimic.robot_spec import ACT_DIM, GOAL_DIM, OBS_DIM


@unittest.skipIf(torch is None, "PyTorch is required; run these controls on the training server")
class BoundedPolicyTests(unittest.TestCase):
    def test_nominal_forward_and_checkpoint_layout_are_unchanged(self):
        torch.manual_seed(20261025)
        cfg = PolicyCfg(encoder_hidden=(8,), decoder_hidden=(8,))
        original, bounded = Policy(cfg), BoundedPolicy(cfg)
        bounded.load_state_dict(original.state_dict(), strict=True)
        obs, goal = torch.randn(3, OBS_DIM * cfg.history), torch.randn(3, GOAL_DIM)
        self.assertTrue(torch.equal(original(obs, goal), bounded(obs, goal)))
        self.assertEqual(original.state_dict().keys(), bounded.state_dict().keys())
        for key, value in bounded.state_dict().items():
            self.assertTrue(torch.equal(value, original.state_dict()[key]))

    def test_extreme_actions_and_gradients_match_real_clipping(self):
        raw = torch.tensor([-200., -1., 1., 200.], requires_grad=True)
        actual = bounded_action(raw)
        self.assertTrue(torch.equal(actual, torch.tensor([-20., -1., 1., 20.])))
        actual.sum().backward()
        self.assertTrue(torch.equal(raw.grad, torch.tensor([0., 1., 1., 0.])))

    def test_raw_nonfinite_values_are_rejected_before_they_can_be_hidden(self):
        for value in (float("inf"), -float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                bounded_action(torch.tensor([value]))

    def test_action_rate_has_the_same_finite_bound_as_real_collection(self):
        previous = bounded_action(torch.full((2, ACT_DIM), -200.))
        current = bounded_action(torch.full((2, ACT_DIM), 200.))
        penalty = -.1 * .02 * ((current - previous) ** 2).sum(-1)
        self.assertTrue(torch.allclose(penalty, torch.full((2,), -70.4)))
        self.assertEqual(ACTION_LIMIT, 20.)

    def test_gradient_hook_preserves_finite_adam_and_rejects_nan_without_mutation(self):
        left = torch.nn.Parameter(torch.tensor([2.]))
        right = torch.nn.Parameter(left.detach().clone())
        original, guarded = torch.optim.Adam([left]), torch.optim.Adam([right])
        handle = reject_nonfinite_gradients(guarded, "fixture")
        for parameter in (left, right):
            parameter.grad = torch.tensor([.3])
        original.step()
        guarded.step()
        self.assertTrue(torch.equal(left, right))
        before = copy.deepcopy(guarded.state_dict())
        value_before = right.detach().clone()
        right.grad = torch.tensor([float("nan")])
        with self.assertRaisesRegex(FloatingPointError, "before Adam step"):
            guarded.step()
        self.assertTrue(torch.equal(right, value_before))
        after = guarded.state_dict()
        self.assertEqual(before["param_groups"], after["param_groups"])
        for index, state in before["state"].items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, after["state"][index][key]))
        handle.remove()


if __name__ == "__main__":
    unittest.main()
