import io
import os
import sys
import logging
from logging.handlers import RotatingFileHandler

from app_config.paths import DEFAULT_LOG_PATH


def setup_logger(
    name: str = "TradingSystem",
    log_file: str = str(DEFAULT_LOG_PATH),
) -> logging.Logger:
    """
    Sets up a structured logger with console and rotating file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler — force UTF-8 so Unicode symbols render on Windows
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") \
        if hasattr(sys.stdout, "buffer") else sys.stdout
    console_handler = logging.StreamHandler(utf8_stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler (10MB limit per file, keeping 5 backup files)
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        errors="replace",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Global system logger instance
system_logger = setup_logger()
