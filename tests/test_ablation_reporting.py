import tempfile
import unittest
from pathlib import Path

import yaml

from swds.experiments.ablation import AblationConfig, run_ablation_experiment
from swds.experiments.aggregate import run_config_batch
from swds.experiments.pipeline import ExperimentConfig, run_temporal_experiment
from swds.experiments.reporting import build_report_from_results
from swds.experiments.synthetic import SyntheticSpec, make_synthetic_temporal_dataset


class AblationReportingTests(unittest.TestCase):
    def test_ablation_outputs_tables(self):
        dataset, _ = make_synthetic_temporal_dataset(SyntheticSpec(n_samples=1200, seed=21))
        base = ExperimentConfig(
            window_size=120,
            methods=("swds",),
            run_retraining=False,
            seed=21,
        )
        config = AblationConfig(
            projection_counts=(8,),
            window_sizes=(120,),
            threshold_quantiles=(0.9,),
            reference_modes=("train", "recent"),
            representation_modes=("raw", "pca"),
            run_threshold=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = run_ablation_experiment(dataset, base_config=base, config=config, output_dir=tmp)
            self.assertFalse(result.correlations.empty)
            self.assertTrue((Path(tmp) / "ablation_correlations.csv").exists())

    def test_report_builder_from_temporal_result(self):
        dataset, _ = make_synthetic_temporal_dataset(SyntheticSpec(n_samples=1200, seed=22))
        config = ExperimentConfig(window_size=120, methods=("swds", "mean_ks"), seed=22)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            report_dir = Path(tmp) / "report"
            run_temporal_experiment(dataset, config=config, output_dir=run_dir)
            written = build_report_from_results([str(run_dir)], output_dir=report_dir)
            self.assertIn("dataset_summary", written)
            self.assertTrue((report_dir / "tables" / "table_1_dataset_summary.csv").exists())

    def test_batch_runner_records_missing_dataset_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "missing.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {
                            "type": "csv",
                            "path": str(Path(tmp) / "absent.csv"),
                            "target_col": "target",
                            "time_col": "time",
                            "task_type": "classification",
                        }
                    }
                ),
                encoding="utf-8",
            )
            manifest = run_config_batch([str(config_path)], output_dir=Path(tmp) / "batch")
            self.assertEqual(manifest["status"].iloc[0], "excluded")
            self.assertTrue((Path(tmp) / "batch" / "dataset_exclusions.csv").exists())


if __name__ == "__main__":
    unittest.main()
