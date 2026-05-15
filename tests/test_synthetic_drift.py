import tempfile
import unittest
from pathlib import Path

from swds.experiments.synthetic import SyntheticSpec, make_synthetic_temporal_dataset
from swds.experiments.synthetic_drift import SyntheticDriftExperimentConfig, run_synthetic_drift_experiment


class SyntheticDriftTests(unittest.TestCase):
    def test_controlled_drift_experiment_writes_detection_tables(self):
        dataset, _ = make_synthetic_temporal_dataset(SyntheticSpec(n_samples=2000, seed=13))
        config = SyntheticDriftExperimentConfig(
            window_size=100,
            drift_start_window=2,
            scenarios=("mean_shift", "categorical_prior_shift", "concept_like_shift"),
            methods=("swds", "mean_ks"),
            seed=13,
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = run_synthetic_drift_experiment(dataset, config=config, output_dir=tmp)
            self.assertFalse(result.window_scores.empty)
            self.assertFalse(result.summary.empty)
            self.assertTrue((Path(tmp) / "synthetic_drift_windows.csv").exists())
            self.assertTrue((Path(tmp) / "synthetic_drift_summary.csv").exists())
            self.assertIn("auroc", result.summary.columns)
            self.assertEqual(set(result.window_scores["drift_label"]), {0, 1})


if __name__ == "__main__":
    unittest.main()
