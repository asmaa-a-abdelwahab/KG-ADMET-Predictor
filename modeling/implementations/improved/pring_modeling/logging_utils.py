from __future__ import annotations

import logging
import sys


def get_logger(name: str = "pring_modeling"):
    """Return a Loguru logger when installed, otherwise a stdlib logger.

    This keeps notebooks usable in fresh local environments where users have
    installed only the core scientific stack and have not yet installed loguru.
    """
    try:
        from loguru import logger as loguru_logger  # type: ignore
        return loguru_logger
    except Exception:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        return logger
