"""
Structured logging menggunakan Loguru.
Konfigurasi logger terpusat untuk seluruh bot.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger


def setup_logger(level: str = "INFO", log_file: str | None = None,
                 rotation: str = "10 MB", retention: str = "7 days") -> Any:
    """
    Inisialisasi logger dengan konfigurasi yang diberikan.

    Args:
        level: Level logging (DEBUG, INFO, WARNING, ERROR).
        log_file: Path file log (opsional). Jika None, hanya stdout.
        rotation: Ukuran rotasi file log.
        retention: Durasi penyimpanan file log lama.

    Returns:
        Instance logger yang sudah dikonfigurasi.
    """
    # Hapus handler default
    logger.remove()

    # Format konsol — warna + timestamp + level + module
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stdout, format=console_format, level=level, colorize=True)

    # File log (opsional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        logger.add(
            str(log_path),
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )

    logger.info(f"Logger diinisialisasi — level={level}, file={log_file}")
    return logger


# Re-export logger untuk dipakai langsung:  from src.utils.logger import log
log = logger
