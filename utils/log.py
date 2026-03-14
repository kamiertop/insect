import os
import sys
from loguru import logger


def init_logger() -> None:
    """
    Central loguru configuration.
    Call this once, as early as possible (program entry).
    """
    # Remove default handler to avoid duplicate logs
    logger.remove()

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = os.getenv("LOG_DIR", "./logs")
    log_file = os.path.join(log_dir, "app.log")

    # Console sink
    logger.add(
        sys.stderr,
        level=level,
        enqueue=True,   # safer with threads/processes
        backtrace=False,
        diagnose=False, # set True only for debugging (can leak data)
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    # File sink (optional)
    os.makedirs(log_dir, exist_ok=True)
    logger.add(
        log_file,
        level=level,
        encoding="utf-8",
        enqueue=True,
        rotation=os.getenv("LOG_ROTATION", "20 MB"),  # or "1 day"
        retention=os.getenv("LOG_RETENTION", "14 days"),
        compression=os.getenv("LOG_COMPRESSION", "zip"),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    )
