# scripts/utils/logging_setup.py

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_file_path, level=logging.DEBUG):
    """
    Sets up logging configuration.

    Args:
        log_file_path (str): The file path for the log file.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    # Ensure the logs directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Define logging format
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create handlers
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,  # Keep up to 5 backup files
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Optionally, prevent logging from propagating to ancestor loggers
    logger.propagate = False
