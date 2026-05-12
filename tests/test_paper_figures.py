import csv
import tempfile
import unittest
from pathlib import Path

from feature_economy.paper import write_all_figures_from_tables


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_has_matplotlib(), "matplotlib is not installed")
class PaperFigureTests(unittest.TestCase):
    def test_write_all_figures_from_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            table_dir = tmpdir / "tables"
            figure_dir = tmpdir / "figures"
            table_dir.mkdir()
            _write_csv(
                table_dir / "probe_scores.csv",
                [
                    {
                        "record_type": "sae_probe_summary",
                        "task_id": "imagenet_1k",
                        "model_id": "dino_v2_base",
                        "sae_id": "dino_l11_topk32_exp4",
                        "seed": "0",
                        "metric": "top1",
                        "value": "0.5",
                        "smoke": "False",
                        "source_path": "x.json",
                    }
                ],
            )
            _write_csv(
                table_dir / "availability_summary.csv",
                [
                    {
                        "model_id": "dino_v2_base",
                        "sae_id": "dino_l11_topk32_exp4",
                        "dataset_id": "imagenet_1k",
                        "split": "val",
                        "usage_unit": "fired_count",
                        "total_features": "4",
                        "fired_features": "3",
                        "dead_features": "1",
                        "active_ratio": "0.75",
                        "dead_ratio": "0.25",
                        "bucket": "dead",
                        "bucket_count": "1",
                        "smoke": "False",
                        "source_path": "x.json",
                    }
                ],
            )
            _write_csv(
                table_dir / "subset_usage_summary.csv",
                [
                    {
                        "task_id": "imagenet_1k",
                        "model_id": "dino_v2_base",
                        "sae_id": "dino_l11_topk32_exp4",
                        "ranking_method": "probe_weight",
                        "usage_unit": "fired_count",
                        "subset": "top_1",
                        "selection": "task_selected",
                        "num_features": "1",
                        "median_usage": "3",
                        "high_usage_fraction": "1.0",
                        "smoke": "False",
                        "source_path": "x.json",
                    }
                ],
            )
            _write_csv(
                table_dir / "ablation_summary.csv",
                [
                    {
                        "task_id": "imagenet_1k",
                        "model_id": "dino_v2_base",
                        "sae_id": "dino_l11_topk32_exp4",
                        "subset": "top_1",
                        "selection": "ablated",
                        "num_features": "1",
                        "metric": "top1",
                        "baseline_value": "1.0",
                        "value": "0.5",
                        "delta": "0.5",
                        "smoke": "False",
                        "source_path": "x.json",
                    }
                ],
            )
            counts = write_all_figures_from_tables(table_dir, figure_dir, formats=("png",))
            self.assertEqual(counts, {"probe_scores": 1, "availability": 1, "subset_usage": 1, "ablation": 1})
            self.assertTrue((figure_dir / "probe_scores.png").exists())
            self.assertTrue((figure_dir / "availability_buckets.png").exists())
            self.assertTrue((figure_dir / "subset_usage.png").exists())
            self.assertTrue((figure_dir / "ablation_deltas.png").exists())


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
