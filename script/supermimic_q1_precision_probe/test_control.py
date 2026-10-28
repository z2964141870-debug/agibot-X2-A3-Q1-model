"""CPU contract checks for the inference-only checkpoint view."""

import unittest

from precision_control import precision_checkpoint


class PrecisionTest(unittest.TestCase):
    def test_fp32_changes_only_copied_args(self):
        weights = object()
        source = {"args": {"amp": "bf16", "hist": 10}, "pi": weights}
        result = precision_checkpoint(source, "fp32")
        self.assertEqual(result["args"], {"amp": "none", "hist": 10})
        self.assertEqual(source["args"], {"amp": "bf16", "hist": 10})
        self.assertIs(result["pi"], weights)
        self.assertIsNot(result["args"], source["args"])

    def test_bf16_preserves_metadata(self):
        source = {"args": {"amp": "bf16"}, "pi": object()}
        self.assertEqual(precision_checkpoint(source, "bf16")["args"], source["args"])

    def test_rejects_unmatched_or_invalid_mode(self):
        with self.assertRaises(ValueError):
            precision_checkpoint({"args": {"amp": "none"}}, "fp32")
        with self.assertRaises(ValueError):
            precision_checkpoint({"args": {"amp": "bf16"}}, "fp16")


if __name__ == "__main__":
    unittest.main()
