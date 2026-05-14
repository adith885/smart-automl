import logging
import os
from datetime import datetime

def get_logger(name="SmartAutoML", log_dir="logs"):

    # Create logs folder if not exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Log file name with timestamp
    log_file = os.path.join(
        log_dir,
        f"automl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate logs
    if not logger.handlers:

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info(f"Logging started. File: {log_file}")

    return logger