"""Reject unrelated delivery failures and media identity mismatches."""

import unittest

from audit_saved import content_match, delivery_scope


class DeliveryContract(unittest.TestCase):
    def test_normal_render(self):
        guard = dict(status="WORKER_COMPLETED", cleanup=dict(worker_return_code=0))
        self.assertEqual(delivery_scope(guard, {}, "", False), "SERVER_RENDER_COMPLETED")

    def test_only_exact_known_missing_tool_can_complete_on_mac(self):
        guard = dict(status="WORKER_FAILED", cleanup=dict(worker_return_code=1))
        controller = dict(status="FAILED", phase="delivery")
        error = "FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe'"
        self.assertIn("PRESERVED", delivery_scope(guard, controller, error, True))
        for enabled, phase, message in ((False, "delivery", error), (True, "training", error),
                                        (True, "delivery", "unknown error")):
            with self.assertRaises(ValueError):
                delivery_scope(guard, dict(controller, phase=phase), message, enabled)

    def test_mirrored_paths_do_not_replace_content_identity(self):
        self.assertTrue(content_match(dict(path="mac", bytes=1, sha256="a"),
                                      dict(path="server", bytes=1, sha256="a")))
        self.assertFalse(content_match(dict(bytes=1, sha256="a"), dict(bytes=1, sha256="b")))


if __name__ == "__main__":
    unittest.main()
