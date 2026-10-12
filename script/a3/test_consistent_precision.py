"""Regression checks for the TRL default that overrides launch precision."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from trl import PPOConfig

from script.a3.consistent_precision import require_fp32


class PrecisionContractTests(unittest.TestCase):
    def test_explicit_bf16_false_preserves_no_mixed_precision(self):
        with patch.dict(os.environ, ACCELERATE_MIXED_PRECISION='no'):
            default = PPOConfig(output_dir='/tmp/yuanqi-precision-config', use_cpu=True, report_to=[])
            self.assertTrue(default.bf16)
            self.assertEqual(os.environ['ACCELERATE_MIXED_PRECISION'], 'bf16')
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
            fixed = PPOConfig(output_dir='/tmp/yuanqi-precision-config', use_cpu=True,
                              report_to=[], bf16=False, fp16=False)
            self.assertFalse(fixed.bf16)
            self.assertEqual(os.environ['ACCELERATE_MIXED_PRECISION'], 'no')

    def test_guard_rejects_native_amp(self):
        accelerator = SimpleNamespace(mixed_precision='bf16', native_amp=True)
        with self.assertRaisesRegex(ValueError, 'FP32 contract violated'):
            require_fp32(accelerator, SimpleNamespace(bf16=False, fp16=False))

    def test_actual_autocast_rejected_even_when_config_says_no(self):
        accelerator = SimpleNamespace(mixed_precision='no', native_amp=False)
        args = SimpleNamespace(bf16=False, fp16=False)
        previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.assertFalse(require_fp32(accelerator, args)['cpu_autocast'])
            with torch.autocast('cpu', dtype=torch.bfloat16), self.assertRaisesRegex(ValueError, 'FP32 contract violated'):
                require_fp32(accelerator, args)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


if __name__ == '__main__':
    unittest.main()
