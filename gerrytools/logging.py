import logging
import os
import sys
from typing import Optional, TextIO

_GERRYTOOLS_LOGGER_NAME = "gerrytools"

# Library-safe default: don't configure global logging on import.
logging.getLogger(_GERRYTOOLS_LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger namespaced under 'gerrytools'.

    If name is None -> 'gerrytools'
    If name already starts with 'gerrytools' -> returned as-is
    Else -> 'gerrytools.<name>'
    """
    if name is None:
        return logging.getLogger(_GERRYTOOLS_LOGGER_NAME)
    if name.startswith(_GERRYTOOLS_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_GERRYTOOLS_LOGGER_NAME}.{name}")


def configure_logging(
    level: int | str | None = None,
    *,
    stream: TextIO | None = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    force: bool = False,
) -> logging.Logger:
    """
    Configure gerrytools package logging. Call from apps/CLI/notebooks.
    Safe to call multiple times.

    Parameters
    ----------
    level:
        int or string like "INFO". If None, uses env var GERRYTOOLS_LOG_LEVEL
        defaulting to INFO.
    stream:
        Where to write logs. Defaults to sys.stderr.
    fmt/datefmt:
        Standard logging formats.
    force:
        If True, removes existing gerrytools handlers first.
    """
    if level is None:
        level = os.getenv("GERRYTOOLS_LOG_LEVEL", "INFO")

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if stream is None:
        stream = sys.stderr

    logger = logging.getLogger(_GERRYTOOLS_LOGGER_NAME)
    logger.setLevel(level)  # type: ignore[arg-type]

    if force:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)  # type: ignore[arg-type]
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(handler)

    logger.propagate = False
    return logger
