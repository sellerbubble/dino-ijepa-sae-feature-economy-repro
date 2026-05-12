import tempfile
import unittest
from pathlib import Path

from feature_economy.configs import build_reproduction_plan, write_reproduction_plan


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


class RunPlanTests(unittest.TestCase):
    def test_build_reproduction_plan_from_public_configs(self):
        plan = build_reproduction_plan(CONFIG_ROOT)
        self.assertEqual(plan["record_type"], "reproduction_run_plan")
        self.assertEqual(plan["num_rows"], len(plan["rows"]))
        self.assertIn("native_probe", plan["stages"])
        self.assertIn("sae_probe", plan["stages"])
        self.assertIn("feature_ablation", plan["stages"])
        self.assertIn("contribution_scores", plan["stages"])
        self.assertTrue(
            any(
                row["task_id"] == "nyuv2_depth"
                and row["model_id"] == "ijepa_vit_h14"
                and row["stage"] == "feature_ablation"
                for row in plan["rows"]
            )
        )

    def test_write_reproduction_plan_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            plan = write_reproduction_plan(
                config_root=CONFIG_ROOT,
                output_json=tmpdir / "run_plan.json",
                output_csv=tmpdir / "run_plan.csv",
            )
            self.assertTrue((tmpdir / "run_plan.json").exists())
            self.assertTrue((tmpdir / "run_plan.csv").exists())
            self.assertIn(str(plan["num_rows"]), (tmpdir / "run_plan.json").read_text())


if __name__ == "__main__":
    unittest.main()
