import logging
import tempfile
import unittest
from pathlib import Path

from swds.logging_utils import configure_logging


class LoggingTests(unittest.TestCase):
    def test_configure_logging_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "swds.log"
            root = logging.getLogger()
            old_handlers = list(root.handlers)
            old_level = root.level
            quiet_handler = logging.NullHandler()
            if not old_handlers:
                root.addHandler(quiet_handler)
            try:
                configure_logging(level="INFO", log_file=path)
                logging.getLogger("swds.test").info("hello logging")
                for handler in logging.getLogger().handlers:
                    handler.flush()
            finally:
                for handler in list(root.handlers):
                    if handler not in old_handlers:
                        root.removeHandler(handler)
                        handler.close()
                root.setLevel(old_level)

            text = path.read_text(encoding="utf-8")
            self.assertIn("hello logging", text)
            self.assertIn("swds.test", text)


if __name__ == "__main__":
    unittest.main()
