import os

# --- Model & Data Configuration ---
MODEL_ID = "openai/whisper-large-v3-turbo"
DATASET_ID = "spktsagar/openslr-nepali-asr-cleaned"
LANGUAGE = "Nepali"
TASK = "transcribe"

# Dataset Limit (set to None for full dataset)
MAX_SAMPLES = None 
TEST_SIZE = 0.1 

# --- Training Hyperparameters ---
NUM_EPOCHS = 10
BATCH_SIZE = 2
GRAD_ACCUMULATION = 4
LEARNING_RATE = 1e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.1

# --- LoRA Configuration ---
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.05
USE_RSLORA = True
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

# --- Checkpointing ---
OUTPUT_DIR = "outputs"
SAVE_STEPS = 500
EVAL_STEPS = 500
SAVE_TOTAL_LIMIT = 2
PATIENCE = 5

# --- Data Processing ---
MAX_LABEL_LENGTH = 448
SAMPLING_RATE = 16000

# --- Training Configuration ---
LOGGING_STEPS = 25
REPORT_TO = ["none"]
PREDICT_WITH_GENERATE = True
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "wer"
GREATER_IS_BETTER = False