"""
Logging configuration module for the SGAN project.

Provides centralized logging setup with file and console outputs.
"""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: str = "outputs/logs",
    log_file: str = "training.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure logging for the project.

    Sets up both console and file handlers with consistent formatting.

    Args:
        log_dir: Directory to store log files
        log_file: Name of the log file
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging("outputs/logs")
        >>> logger.info("Training started")
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("melanoma_sgan")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_path = Path(log_dir) / log_file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"melanoma_sgan.{name}")
