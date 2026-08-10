"""Stable project-relative paths for runtime state.

The terminal is often launched by a shortcut or watchdog whose working
directory is not guaranteed. Runtime state must still resolve to one database,
one environment file, and one log location.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_DB_PATH = PROJECT_ROOT / "trading_system.db"
DEFAULT_LOG_PATH = PROJECT_ROOT / "trading_system.log"

