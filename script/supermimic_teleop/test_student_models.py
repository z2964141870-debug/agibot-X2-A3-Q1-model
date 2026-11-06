"""CPU synthetic gradient contracts, not training or inference of the 32k policy."""

import unittest

import torch
from torch import nn

from student_inputs import FRAME_DIM, GEOMETRY_DIM, HISTORY
from student_models import (CurrentHistoryEncoder, FrozenDecoderStudent,
    action_distillation_loss, masked_preview_goal)


class DecoderFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4+6, 29)

    def forward(self, z, history):
        return self.net(torch.cat((z, history), -1))


class StudentModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_temporal_mask_blocks_hidden_pose_heading_and_gradients(self):
        torch.manual_seed(7)
        features = torch.randn(5, 2160, requires_grad=True)
        heading = torch.randn(5, 60, requires_grad=True)
        horizons = torch.tensor((1, 2, 4, 5, 10))
        actual = masked_preview_goal(features, heading, horizons)
        self.assertEqual(actual.shape, (5, 2254))
        alternate, alternate_heading = features.detach().clone(), heading.detach().clone()
        for i, horizon in enumerate(horizons.tolist()):
            alternate[i, horizon*216:] = 1e6
            alternate_heading[i, horizon*6:] = -1e6
        self.assertTrue(torch.equal(actual,
            masked_preview_goal(alternate, alternate_heading, horizons)))
        actual.sum().backward()
        for i, horizon in enumerate(horizons.tolist()):
            self.assertEqual(torch.count_nonzero(features.grad[i, horizon*216:]).item(), 0)
            self.assertEqual(torch.count_nonzero(heading.grad[i, horizon*6:]).item(), 0)
            self.assertTrue(torch.equal(actual[i, 2160:2184], torch.ones(24)))

    def test_encoder_updates_through_frozen_decoder_without_mutating_teacher(self):
        torch.manual_seed(8)
        encoder, decoder = nn.Sequential(nn.Linear(8, 4), nn.Tanh()), DecoderFixture()
        teacher_encoder = {k: v.clone() for k, v in encoder.state_dict().items()}
        teacher_decoder = {k: v.clone() for k, v in decoder.state_dict().items()}
        student = FrozenDecoderStudent(encoder, decoder).train()
        self.assertFalse(student.decoder.training)
        goal, history = torch.randn(3, 8), torch.randn(3, 6)
        label = torch.randn(3, 29, requires_grad=True)
        optimizer = torch.optim.SGD(student.trainable_parameters(), lr=.01)
        action = student(history, goal)
        loss = action_distillation_loss(action, label, torch.ones(29, dtype=torch.bool))
        loss.backward()
        optimizer.step()
        self.assertIsNone(label.grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in student.decoder.parameters()))
        self.assertTrue(any(not torch.equal(v, teacher_encoder[k]) for k, v in student.encoder.state_dict().items()))
        self.assertTrue(all(torch.equal(v, teacher_encoder[k]) for k, v in encoder.state_dict().items()))
        self.assertTrue(all(torch.equal(v, teacher_decoder[k]) for k, v in decoder.state_dict().items()))
        self.assertTrue(all(torch.equal(v, teacher_decoder[k]) for k, v in student.decoder.state_dict().items()))

    def test_absent_joints_have_no_training_target_or_gradient(self):
        action = torch.zeros(2, 29, requires_grad=True)
        labels = torch.ones_like(action)
        present = torch.arange(29) < 22
        labels[:, ~present] = 1e6
        loss = action_distillation_loss(action, labels, present)
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        self.assertEqual(torch.count_nonzero(action.grad[:, ~present]).item(), 0)

    def test_f0_encoder_zeroes_padding_after_train_only_normalization(self):
        statistics = dict(schema="q1_current_geometry_stats_v1", fitted_split="train",
            count=9, geometry_dim=GEOMETRY_DIM, clipping=10.,
            mean=[1.]*GEOMETRY_DIM, scale=[.2]*GEOMETRY_DIM)
        encoder = CurrentHistoryEncoder(4, statistics)
        values = torch.zeros(2, HISTORY, FRAME_DIM)
        values[:, -1, GEOMETRY_DIM:GEOMETRY_DIM+2] = 1.
        altered = values.clone()
        altered[:, :-1, :GEOMETRY_DIM] = 1e6
        self.assertTrue(torch.equal(encoder(values), encoder(altered)))
        output = encoder(values)
        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool((output.abs() <= 1.).all()))
        self.assertEqual(encoder.mean.requires_grad, False)
        for split in ("validation", "test"):
            with self.assertRaises(ValueError):
                CurrentHistoryEncoder(4, dict(statistics, fitted_split=split))

    def test_nonfinite_labels_and_invalid_horizons_do_not_produce_a_fit_loss(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            action_distillation_loss(torch.zeros(2, 29), torch.full((2, 29), float("nan")),
                                     torch.ones(29, dtype=torch.bool))
        for horizon in (0, 11):
            with self.assertRaises(ValueError):
                masked_preview_goal(torch.zeros(1, 2160), torch.zeros(1, 60), torch.tensor([horizon]))


if __name__ == "__main__":
    unittest.main()
