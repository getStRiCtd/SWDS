import tempfile
import unittest
from pathlib import Path

from swds.experiments.config import (
    experiment_config_from_mapping,
    load_dataset_from_experiment_config,
    load_experiment_yaml,
    synthetic_drift_config_from_mapping,
)


class ConfigTests(unittest.TestCase):
    def test_nested_synthetic_config(self):
        raw = {
            "dataset": {"type": "synthetic", "n_samples": 500, "task_type": "classification", "seed": 11},
            "experiment": {
                "window_size": 50,
                "methods": ["swds"],
                "n_jobs": -1,
                "swds_backend": "auto",
                "retraining": {"periods": [2], "history_modes": ["all"]},
            },
        }
        dataset = load_dataset_from_experiment_config(raw)
        config = experiment_config_from_mapping(raw)

        self.assertEqual(len(dataset.frame), 500)
        self.assertEqual(config.methods, ("swds",))
        self.assertEqual(config.n_jobs, -1)
        self.assertEqual(config.swds_backend, "auto")
        self.assertEqual(config.retraining_periods, (2,))
        self.assertEqual(config.retraining_history_modes, ("all",))

    def test_synthetic_drift_config_reuses_experiment_defaults(self):
        raw = {
            "experiment": {"window_size": 123, "methods": ["swds"], "seed": 5},
            "synthetic_drift": {"drift_start_window": 2, "scenarios": ["mean_shift"]},
        }
        config = synthetic_drift_config_from_mapping(raw)
        self.assertEqual(config.window_size, 123)
        self.assertEqual(config.methods, ("swds",))
        self.assertEqual(config.scenarios, ("mean_shift",))
        self.assertEqual(config.seed, 5)

    def test_load_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("dataset:\n  type: synthetic\n", encoding="utf-8")
            self.assertEqual(load_experiment_yaml(path)["dataset"]["type"], "synthetic")


if __name__ == "__main__":
    unittest.main()
