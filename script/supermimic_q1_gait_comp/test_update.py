"""CPU synthetic optimizer check; no simulator or physical training evidence."""

import unittest

import torch
from tensordict import TensorDict

from gait_adapter import Actor, make_ppo


class UpdateTests(unittest.TestCase):
    def test_critic_only_then_actor_and_adaptive_cap(self):
        torch.set_num_threads(4)
        torch.manual_seed(7)
        obs = TensorDict({"features": torch.randn(16, 1420)*.1,
                          "privileged": torch.randn(16, 62)*.1}, batch_size=[16], device="cpu")
        actor = Actor()
        algorithm = make_ppo(actor, obs, 16)
        original = {k:v.clone() for k,v in actor.state_dict().items()}
        actor.head.requires_grad_(False)
        for warmup in (True, False):
            if not warmup:
                actor.head.requires_grad_(True)
            with torch.no_grad():
                for _ in range(24):
                    action = algorithm.act(obs)
                    reward = -10*(action[:, 0]-.2).square()
                    done = torch.zeros(16, dtype=torch.bool)
                    algorithm.process_env_step(obs, reward, done, {})
                algorithm.compute_returns(obs)
            losses = algorithm.update()
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
            if warmup:
                for key,value in actor.state_dict().items():
                    self.assertTrue(torch.equal(value, original[key]))
        self.assertTrue(any(not torch.equal(v, original[k]) for k,v in actor.state_dict().items()))
        algorithm.learning_rate = .01
        algorithm.optimizer.zero_grad()
        (algorithm.critic(obs).square().mean()+actor(obs).square().mean()).backward()
        algorithm.optimizer.step()
        self.assertEqual(algorithm.learning_rate, 5e-5)
        self.assertTrue(all(g["lr"] ==5e-5 for g in algorithm.optimizer.param_groups))
        self.assertEqual(algorithm.schedule, "adaptive")
        self.assertEqual(algorithm.desired_kl, .01)


if __name__ == "__main__":
    unittest.main()
