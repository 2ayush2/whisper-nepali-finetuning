import os
import datetime
from typing import Tuple

def setup_experiment_dir(config, args) -> Tuple[str, str]:
    """Creates a unique experiment directory based on timestamp and args."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{timestamp}_lr{args.learning_rate}_bs{args.batch_size}"
    output_dir = os.path.join(config.OUTPUT_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, run_name
