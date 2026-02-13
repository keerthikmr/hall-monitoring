import logging
import os
from core.config import config


def setup_logger():
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger("QuantumExamMonitor")
    logger.setLevel(config.log_level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(config.log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()
