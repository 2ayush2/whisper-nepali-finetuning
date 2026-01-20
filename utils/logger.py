import logging, os, sys

class ExperimentLogger:
    @staticmethod
    def setup(output_dir):
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
        # benign Whisper Turbo weight warnings to avoid Unsloth exceptions
        class UninitializedWeightFilter(logging.Filter):
            def filter(self, record):
                return "proj_out.weight" not in record.getMessage()
        logging.getLogger("transformers.modeling_utils").addFilter(UninitializedWeightFilter())
        return logging.getLogger("WhisperSOTA")
    setup_logging = setup
