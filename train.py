import os
import sys
import logging
import torch
import gc
import subprocess
import argparse
from functools import partial
from typing import Optional, Tuple

from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, Audio
from unsloth import FastModel, is_bf16_supported

import config
from utils.logger import ExperimentLogger
from utils.data_utils import formatting_prompts_func, remove_bad_examples, DataCollatorSpeechSeq2SeqWithPadding, compute_metrics
from utils.file_utils import setup_experiment_dir

logger = logging.getLogger(__name__)

class WhisperFinetuner:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.output_dir, self.run_name = setup_experiment_dir(config, self.args)
        self.logger = ExperimentLogger.setup_logging(self.output_dir)
        self.logger.info(f"Initialized WhisperFinetuner for {self.run_name}")

    def load_model(self):
        self.logger.info(f"Loading Base Model: {config.MODEL_ID}")
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=config.MODEL_ID,
            dtype=None,
            load_in_4bit=False,
            auto_model=WhisperForConditionalGeneration,
            whisper_language=config.LANGUAGE,
            whisper_task=config.TASK,
        )
        self.logger.info(f"Applying LoRA Adapters (Rank: {self.args.lora_rank})")
        self.model = FastModel.get_peft_model(
            self.model,
            r=self.args.lora_rank,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=self.args.lora_rank * 2,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        # Generation Config
        self.model.generation_config.language = "ne"
        self.model.generation_config.task = config.TASK
        self.model.config.suppress_tokens = []
        
    def prepare_data(self):
        """Loads and preprocesses the dataset."""
        self.logger.info(f"Loading Dataset: {config.DATASET_ID}")
        dataset = load_dataset(config.DATASET_ID, name="cleaned", split="train", trust_remote_code=True)
        dataset = dataset.cast_column("utterance", Audio(sampling_rate=16000))
    
        # Optional: Limit total samples first
        if config.MAX_SAMPLES:
            dataset = dataset.select(range(min(config.MAX_SAMPLES, len(dataset))))

        # Split into Train/Test using ratio
        dataset = dataset.train_test_split(test_size=config.TEST_SIZE)
        raw_train = dataset["train"]
        raw_test = dataset["test"]

        self.logger.info(f"Train Size: {len(raw_train)}, Test Size: {len(raw_test)}")

        # Processing
        self.logger.info("Mapping and Filtering Dataset...")
        process_func = partial(formatting_prompts_func, tokenizer=self.tokenizer)

        self.train_dataset = raw_train.map(process_func, remove_columns=raw_train.column_names, batched=False)
        self.test_dataset = raw_test.map(process_func, remove_columns=raw_test.column_names)
        
        self.train_dataset = self.train_dataset.filter(remove_bad_examples)
        self.test_dataset = self.test_dataset.filter(remove_bad_examples)

    def train(self):
        self.logger.info("Initializing Seq2SeqTrainer...")
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=config.GRAD_ACCUMULATION,
            learning_rate=self.args.learning_rate,
            warmup_ratio=0.1,
            num_train_epochs=self.args.epochs,
            logging_steps=25,
            save_steps=config.SAVE_STEPS,
            eval_steps=config.EVAL_STEPS,
            save_total_limit=config.SAVE_TOTAL_LIMIT,
            optim="adamw_8bit",
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            weight_decay=0.001,
            lr_scheduler_type=config.LR_SCHEDULER,
            predict_with_generate=True,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            eval_strategy="steps",
            report_to="none",
            seed=3407,
        )
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=self.tokenizer),
            tokenizer=self.tokenizer,
            compute_metrics=partial(compute_metrics, tokenizer=self.tokenizer),
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.PATIENCE)],
        )
        self.logger.info("Starting training...")
        self.trainer.train()

    def export_model(self):
        self.logger.info("Saving Final Model Locally...")
        local_model_dir = os.path.join(self.output_dir, "final_model")
        self.model.save_pretrained_merged(local_model_dir, self.tokenizer, save_method="merged_16bit")
        self.logger.info(f"Model saved to: {local_model_dir}")
        self.logger.info("Export Completed Successfully!")

def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Fine-Tuning Manager")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lora_rank", type=int, default=config.LORA_R)
    return parser.parse_args()

def main():
    # Clear Cache
    gc.collect()
    torch.cuda.empty_cache()

    args = parse_args()
    
    # Initialize Finetuner
    finetuner = WhisperFinetuner(args)
    
    # Run Pipeline
    finetuner.load_model()
    finetuner.prepare_data()
    finetuner.train()
    finetuner.export_model()

if __name__ == "__main__":
    main()
