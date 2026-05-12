import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from feature_economy.runtime import check_runtime, require_runtime, write_runtime_report


class RuntimeTests(unittest.TestCase):
    def test_smoke_runtime_available(self):
        statuses = require_runtime("smoke")
        self.assertTrue(all(status.available for status in statuses))

    def test_base_profile_kept_as_alias(self):
        smoke_modules = [status.module for status in check_runtime("smoke")]
        base_modules = [status.module for status in check_runtime("base")]
        self.assertEqual(smoke_modules, base_modules)

    def test_figures_profile_includes_matplotlib(self):
        modules = [status.module for status in check_runtime("figures")]
        self.assertIn("matplotlib", modules)

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):
            check_runtime("unknown")

    def test_write_runtime_report(self):
        statuses = check_runtime("smoke")
        with TemporaryDirectory() as tmp_dir:
            output_path = write_runtime_report(
                statuses,
                Path(tmp_dir) / "runtime.json",
                profile="smoke",
            )
            text = output_path.read_text()
        self.assertIn('"artifact_type": "runtime_dependency_report"', text)
        self.assertIn('"profile": "smoke"', text)


if __name__ == "__main__":
    unittest.main()
