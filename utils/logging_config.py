import logging
import os
from datetime import datetime


def setup_logger(name="training", log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{run_id}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # IMPORTANT: Avoid duplicate handlers in notebooks
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.propagate = False

    logger.info(f"Logging initialized. Saving to {log_path}")

    return logger
