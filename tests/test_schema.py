import unittest

import pandas as pd

from swds.data.schema import TaskType, normalize_task_type


class SchemaTests(unittest.TestCase):
    def test_string_target_is_classification(self):
        task = normalize_task_type(None, pd.Series(["yes", "no", "yes"], dtype="string"))
        self.assertEqual(task, TaskType.CLASSIFICATION)


if __name__ == "__main__":
    unittest.main()
