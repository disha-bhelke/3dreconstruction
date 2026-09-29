from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "3d_reconstruction",
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
    log_filename: str | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    target_dir = Path(log_dir) if log_dir is not None else Path(__file__).resolve().parents[1] / "outputs" / "logs"
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = target_dir / (log_filename or f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
