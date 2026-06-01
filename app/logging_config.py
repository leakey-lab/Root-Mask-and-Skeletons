"""Centralized logging setup for Root-Mask-and-Skeletons.

Call setup_logging() once at application startup (before any other app imports).
Log file: ~/.root_mask_logs/app.log — rotates at 5 MB, keeps 3 backups.
Set DEBUG=1 env var to enable debug-level output.
"""
import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging(log_dir: Path | None = None) -> None:
    """Configure root logger with stderr + rotating file handlers."""
    level = logging.DEBUG if os.getenv("DEBUG") else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if log_dir is None:
        log_dir = Path.home() / ".root_mask_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(), fh],
    )
