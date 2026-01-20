# Whisper Fine-Tuning for Nepali ASR

Fine-tuning OpenAI Whisper models for Nepali speech recognition using Unsloth and LoRA.

## Requirements

- Python 3.10+
- CUDA-compatible GPU (16GB+ VRAM recommended)
- Linux (Ubuntu 20.04+ recommended)

## Installation

```bash
git clone https://github.com/2ayush2/whisper-nepali-finetuning.git
cd whisper-nepali-finetuning
pip install -r requirements.txt
```

## Usage

Basic training:
```bash
python train.py
```

With custom arguments:
```bash
python train.py --learning_rate 1e-4 --batch_size 2 --epochs 10 --lora_rank 128
```

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --learning_rate | float | 1e-4 | Learning rate |
| --batch_size | int | 2 | Per-device batch size |
| --epochs | int | 10 | Number of training epochs |
| --lora_rank | int | 128 | LoRA rank |

## Configuration

Edit `config.py` to modify default settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| MODEL_ID | openai/whisper-large-v3-turbo | Base model |
| DATASET_ID | spktsagar/openslr-nepali-asr-cleaned | Dataset |
| MAX_SAMPLES | 500 | Limit samples (None for full) |
| GRAD_ACCUMULATION | 4 | Gradient accumulation steps |
| LORA_R | 128 | LoRA rank |
| LORA_ALPHA | 256 | LoRA alpha |
| EVAL_STEPS | 50 | Evaluation frequency |
| SAVE_STEPS | 50 | Checkpoint frequency |

## Output Structure

```
outputs/
  run_2026-01-20_10-30-00_lr0.0001_bs2/
    training.log
    checkpoint-50/
    checkpoint-100/
    final_model_merged/
```
