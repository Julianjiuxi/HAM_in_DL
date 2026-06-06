"""Logging helpers."""

import logging


def get_logger(name: str = "ham_in_dl") -> logging.Logger:
    """Create a standard logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
