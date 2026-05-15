import tempfile
import unittest
from pathlib import Path

from swds.experiments.pipeline import ExperimentConfig, run_temporal_experiment
from swds.experiments.synthetic import SyntheticSpec, make_synthetic_temporal_dataset


class PipelineTests(unittest.TestCase):
    def test_synthetic_pipeline_writes_results(self):
        dataset, _ = make_synthetic_temporal_dataset(SyntheticSpec(n_samples=1500, seed=3))
        config = ExperimentConfig(
            window_size=150,
            methods=("swds", "mean_ks", "max_ks"),
            seed=3,
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = run_temporal_experiment(dataset, config=config, output_dir=tmp)
            self.assertFalse(result.window_scores.empty)
            self.assertFalse(result.correlations.empty)
            self.assertFalse(result.retraining_summary.empty)
            self.assertTrue((Path(tmp) / "window_scores.csv").exists())
            self.assertTrue((Path(tmp) / "retraining_policy_summary.csv").exists())
            self.assertIn("swds", set(result.window_scores["method"]))

    def test_recent_reference_and_pca_representation_run(self):
        dataset, _ = make_synthetic_temporal_dataset(SyntheticSpec(n_samples=1500, seed=4))
        config = ExperimentConfig(
            window_size=150,
            methods=("swds",),
            reference_mode="recent",
            drift_representation="pca",
            pca_components=4,
            run_retraining=False,
            seed=4,
        )
        result = run_temporal_experiment(dataset, config=config)
        self.assertFalse(result.window_scores.empty)
        self.assertEqual(set(result.window_scores["reference_mode"]), {"recent"})
        self.assertEqual(set(result.window_scores["drift_representation"]), {"pca"})


if __name__ == "__main__":
    unittest.main()
