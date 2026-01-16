import logging
import os
import sys

class ExperimentLogger:
    """
    Handles logging configuration for experiments.
    Ensures that every run has its own log file while also streaming to console.
    """
    @staticmethod
    def setup_logging(output_dir: str):
        """
        Configures the root logger to write to a file in the output directory
        and stream to stdout.
        """
        # Create formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        
        # File Handler
        log_file = os.path.join(output_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Root Logger Configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to prevent duplicates on restarts
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
            
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured. Log file: {log_file}")
        
        return root_logger
