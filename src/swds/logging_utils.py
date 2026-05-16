from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    *,
    level: str | int | None = None,
    log_file: str | Path | None = None,
    force: bool = False,
) -> None:
    """Configure console and optional file logging for CLI runs."""

    log_level = _coerce_log_level(level or os.environ.get("SWDS_LOG_LEVEL", "INFO"))
    resolved_log_file = _resolve_log_file(log_file or os.environ.get("SWDS_LOG_FILE"))
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    root = logging.getLogger()
    if root.handlers and not force:
        root.setLevel(log_level)
        for handler in root.handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
        if resolved_log_file is not None and not _has_file_handler(root, resolved_log_file):
            root.addHandler(_file_handler(resolved_log_file, log_level, formatter))
        logging.captureWarnings(True)
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if resolved_log_file is not None:
        handlers.append(_file_handler(resolved_log_file, log_level, formatter))
    for handler in handlers:
        handler.setLevel(log_level)
        handler.setFormatter(formatter)

    logging.basicConfig(level=log_level, handlers=handlers, force=force)
    logging.captureWarnings(True)


def _coerce_log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    name = str(level).upper()
    numeric = logging.getLevelName(name)
    if isinstance(numeric, int):
        return numeric
    raise ValueError(f"unknown log level: {level!r}")


def _resolve_log_file(log_file: str | Path | None) -> Path | None:
    if log_file is None or str(log_file).strip() == "":
        return None
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _file_handler(path: Path, level: int, formatter: logging.Formatter) -> logging.FileHandler:
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def _has_file_handler(root: logging.Logger, path: Path) -> bool:
    resolved = path.resolve()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == resolved:
            return True
    return False
