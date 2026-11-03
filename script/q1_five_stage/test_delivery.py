"""Delivery receipts cannot accept swapped media, paths, or missing visual checks."""

from pathlib import Path
import tempfile
import unittest

from delivery import artifact, check_visuals


class DeliveryTests(unittest.TestCase):
    def test_visual_review_is_bound_to_every_video_sha(self):
        manifest = dict(artifacts=[dict(relative=f"{i}.mp4", bytes=1, sha256=str(i) * 64) for i in range(6)])
        visual = dict(status="REVIEWED", media=[dict(relative=f"{i}.mp4", sha256=str(i) * 64,
                      visual_reviewed=True) for i in range(6)])
        check_visuals(manifest, visual)
        visual["media"][0]["sha256"] = "wrong"
        with self.assertRaises(RuntimeError):
            check_visuals(manifest, visual)

    def test_duplicate_and_unreviewed_videos_block(self):
        manifest = dict(artifacts=[dict(relative="a.mp4", sha256="a" * 64)])
        for rows in ([], [dict(relative="a.mp4", sha256="a" * 64, visual_reviewed=False)],
                     [dict(relative="a.mp4", sha256="a" * 64, visual_reviewed=True)] * 2):
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                check_visuals(manifest, dict(status="REVIEWED", media=rows))

    def test_artifacts_reject_symlinks_and_parent_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real.csv").write_text("real")
            (root / "link.csv").symlink_to(root / "real.csv")
            self.assertEqual(artifact(root, "real.csv")["bytes"], 4)
            for relative in ("link.csv", "../outside.csv", str(root / "real.csv")):
                with self.subTest(relative=relative), self.assertRaises(RuntimeError):
                    artifact(root, relative)


if __name__ == "__main__":
    unittest.main()
