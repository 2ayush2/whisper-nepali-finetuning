# Whisper Fine-Tuning Setup

A research-grade, production-ready setup for fine-tuning Whisper models.

## 🚀 Quick Start
1.  **Dependencies**: `pip install -r requirements.txt`
2.  **Environment**: Create `.env` and set your `HF_TOKEN`.
3.  **Run**: `run_training.bat` (Windows) or `python train.py` (Linux).

## 🔬 Experimentation (Research Mode)
This setup supports experiment tracking. Each run creates a unique, timestamped folder in `outputs/` so you never overwrite good models.

### CLI Arguments
You can override default settings from the command line to quick-test new parameters:

```bash
# Experiment 1: Lower learning rate
python train.py --learning_rate 5e-5

# Experiment 2: Larger batch size, more epochs
python train.py --batch_size 4 --epochs 3
```

### Output Structure
```
outputs/
└── run_2024-01-16_10-50_lr0.0002_bs2/
    ├── training.log            # Full training logs
    ├── hyperparameters.json    # Exact settings used for this run
    ├── checkpoint-250/         # Saved model checkpoints
    └── ...
```

## ⚙️ Configuration (`config.py`)
Edit `config.py` to change default "production" settings:
- `MODEL_ID`: Base model (e.g., `Dragneel/whisper-medium-nepali-openslr`)
- `LORA_R`: LoRA Rank (Efficiency vs Capacity)
- `TARGET_MODULES`: Layers to fine-tune
- `SAVE_STEPS`: How often to save.

## 🛠️ Utils (`utils.py`)
Contains data preprocessing logic. Modify this if you need to change how audio is cleaned or how metrics (WER) are calculated.
