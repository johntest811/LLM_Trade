import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app_config.paths import DEFAULT_DB_PATH, DEFAULT_LOG_PATH, ENV_PATH, PROJECT_ROOT
from core.engine import TradingEngine
from database.replay_logger import TradeReplayLogger
from database.storage import TradingDatabase


class RuntimePathTests(unittest.TestCase):
    def test_default_runtime_paths_are_anchored_to_project_root(self):
        self.assertTrue(DEFAULT_DB_PATH.is_absolute())
        self.assertTrue(DEFAULT_LOG_PATH.is_absolute())
        self.assertTrue(ENV_PATH.is_absolute())
        self.assertEqual(DEFAULT_DB_PATH.parent, PROJECT_ROOT)
        self.assertEqual(DEFAULT_LOG_PATH.parent, PROJECT_ROOT)
        self.assertEqual(ENV_PATH.parent, PROJECT_ROOT)
        self.assertEqual(
            Path(inspect.signature(TradingDatabase).parameters["db_path"].default),
            DEFAULT_DB_PATH,
        )
        self.assertEqual(
            Path(inspect.signature(TradeReplayLogger).parameters["db_path"].default),
            DEFAULT_DB_PATH,
        )

    def test_engine_replay_logger_uses_injected_database_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "isolated.db")
            engine = TradingEngine(
                object(),
                object(),
                object(),
                object(),
                SimpleNamespace(db_path=db_path),
                object(),
            )

            self.assertEqual(Path(engine.replay_logger.db_path), Path(db_path))


if __name__ == "__main__":
    unittest.main()
