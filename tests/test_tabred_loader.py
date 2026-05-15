import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from swds.data.loaders import load_tabred_dataset
from swds.data.schema import TaskType


class TabReDLoaderTests(unittest.TestCase):
    def test_loads_processed_tabred_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "X_num.npy", np.arange(20, dtype=np.float32).reshape(10, 2))
            np.save(root / "X_bin.npy", (np.arange(10) % 2).astype(np.float32).reshape(10, 1))
            np.save(root / "X_cat.npy", np.array([["a"], ["b"]] * 5))
            np.save(root / "Y.npy", np.array([0, 1] * 5))
            (root / "info.json").write_text(json.dumps({"task_type": "binclass"}), encoding="utf-8")

            split_dir = root / "split-default"
            split_dir.mkdir()
            np.save(split_dir / "train_idx.npy", np.array([0, 1, 2, 3]))
            np.save(split_dir / "val_idx.npy", np.array([4, 5]))
            np.save(split_dir / "test_idx.npy", np.array([6, 7, 8, 9]))

            dataset = load_tabred_dataset(root, name="toy-tabred")

        self.assertEqual(dataset.name, "toy-tabred")
        self.assertEqual(dataset.task_type, TaskType.CLASSIFICATION)
        self.assertEqual(dataset.split_col, "split")
        self.assertEqual(len(dataset.frame), 10)
        self.assertIn("cat_0", dataset.feature_columns)
        self.assertEqual(list(dataset.frame["split"].unique()), ["train", "val", "test"])


if __name__ == "__main__":
    unittest.main()
