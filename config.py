import os
from dotenv import load_dotenv

load_dotenv()

# --- Security ---
HF_TOKEN = os.getenv("HF_TOKEN")
# Optional: Only needed if uploading to HuggingFace Hub

# --- Model & Data Configuration ---
MODEL_ID = "Dragneel/whisper-medium-nepali-openslr"
DATASET_ID = "spktsagar/openslr-nepali-asr-cleaned"
LANGUAGE = "Nepali"
TASK = "transcribe"

# Dataset Limit (set to None for full dataset)
MAX_SAMPLES = 500  # Total samples to use (None = use all). Split by TEST_SIZE ratio.
TEST_SIZE = 0.1 

# --- Training Hyperparameters ---
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUMULATION = 4
LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"

# --- LoRA Configuration ---
LORA_R = 64
LORA_ALPHA = 64

# --- Checkpointing ---
OUTPUT_DIR = "outputs"
SAVE_STEPS = 250
EVAL_STEPS = 250
SAVE_TOTAL_LIMIT = 2
PATIENCE = 3
