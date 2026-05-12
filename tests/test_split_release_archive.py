import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "split_release_archive.py"


class SplitReleaseArchiveTests(unittest.TestCase):
    def test_splits_archive_and_writes_part_checksums(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            archive = tmpdir / "bundle.tar.gz"
            payload = b"feature-economy-release" * 64
            archive.write_bytes(payload)

            subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--archive",
                    str(archive),
                    "--part-size",
                    "100",
                ],
                check=True,
            )

            parts = sorted(tmpdir.glob("bundle.tar.gz.part-*"))
            self.assertGreater(len(parts), 1)
            self.assertTrue((tmpdir / "bundle.tar.gz.parts.sha256").exists())
            reassembled = b"".join(part.read_bytes() for part in parts)
            self.assertEqual(reassembled, payload)
            self.assertIn(
                hashlib.sha256(parts[0].read_bytes()).hexdigest(),
                (tmpdir / "bundle.tar.gz.parts.sha256").read_text(),
            )

    def test_refuses_to_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            archive = tmpdir / "bundle.tar.gz"
            archive.write_bytes(b"abc")
            (tmpdir / "bundle.tar.gz.part-00").write_bytes(b"old")

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--archive",
                    str(archive),
                    "--part-size",
                    "1",
                ],
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--force", result.stderr)


if __name__ == "__main__":
    unittest.main()
