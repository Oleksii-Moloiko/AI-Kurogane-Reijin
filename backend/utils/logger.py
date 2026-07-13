"""Centralized application logging configuration."""

import logging
from logging.handlers import RotatingFileHandler

from backend.config import settings


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MAX_LOG_FILE_SIZE_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


def configure_logging() -> None:
    """Configure application logging once."""

    root_logger = logging.getLogger()

    if getattr(root_logger, "_kuro_configured", False):
        return

    settings.logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_level = _resolve_log_level(settings.log_level)

    formatter = logging.Formatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )

    file_handler = RotatingFileHandler(
        filename=settings.log_file,
        maxBytes=MAX_LOG_FILE_SIZE_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )

    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)

    setattr(root_logger, "_kuro_configured", True)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""

    configure_logging()

    return logging.getLogger(name)


def _resolve_log_level(level_name: str) -> int:
    """Convert a textual logging level into a logging constant."""

    level = getattr(
        logging,
        level_name.strip().upper(),
        None,
    )

    if not isinstance(level, int):
        raise ValueError(
            f"Unsupported log level: {level_name}"
        )

    return level