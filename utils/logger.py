import logging
import os
import sys

class ExperimentLogger:
    @staticmethod
    def setup_logging(output_dir: str):
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        log_file = os.path.join(output_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured. Log file: {log_file}")
        return root_logger
