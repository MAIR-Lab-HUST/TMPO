"""Logging utilities."""

import os
import logging
from typing import Optional


_logger = None


def setup_logging(
    name: str = "TMPO",
    log_dir: Optional[str] = None,
    rank: int = 0,
) -> logging.Logger:
    """Set up logger."""
    global _logger
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    _logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    _logger.addHandler(console)

    if log_dir and rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
        fh.setFormatter(formatter)
        _logger.addHandler(fh)

    return _logger


def main_print(*args, **kwargs):
    """Print only on main process (rank 0)."""
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        if _logger is not None:
            _logger.info(" ".join(str(a) for a in args))
        else:
            print(*args, **kwargs)
