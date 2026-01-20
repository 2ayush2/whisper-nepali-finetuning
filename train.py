import os
import logging
import gc
import argparse
import torch
from datasets import load_dataset, Audio
import datasets
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
from transformers import (WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback)
from transformers.trainer_utils import get_last_checkpoint
from unsloth import FastModel, is_bf16_supported
import config
from utils.logger import ExperimentLogger
from utils.data_utils import (create_formatting_function, remove_bad_examples, DataCollatorSpeechSeq2SeqWithPadding, create_compute_metrics)
from utils.file_utils import setup_experiment_dir

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
        try:
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=config.MODEL_ID,
                dtype=None,
                load_in_4bit=False,
                auto_model=WhisperForConditionalGeneration,
                whisper_language=config.LANGUAGE,
                whisper_task=config.TASK)
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
        self.logger.info(f"Applying LoRA (Rank={self.args.lora_rank}, Alpha={config.LORA_ALPHA})")
        self.model = FastModel.get_peft_model(
            self.model,
            r=self.args.lora_rank,
            target_modules=config.LORA_TARGET_MODULES,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=config.USE_RSLORA,
            loftq_config=None,
            task_type=None)
        self.model.generation_config.language = "<|ne|>"
        self.model.generation_config.task = config.TASK
        self.model.config.suppress_tokens = []
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.max_length = config.MAX_LABEL_LENGTH
        self.model.generation_config.num_beams = 5
        # NOTE: Spec augment disabled due to torch._dynamo errors
        self.model.config.apply_spec_augment = False
        self.model.config.mask_time_prob = 0.05
        self.model.config.mask_feature_prob = 0.05
        self.model.config.mask_time_length = 10
        self.model.config.mask_feature_length = 10
        self.model.config.use_cache = False

    def prepare_data(self):
        self.logger.info(f"Loading Dataset: {config.DATASET_ID}")
        dataset = load_dataset(config.DATASET_ID, name="cleaned", split="train", trust_remote_code=True)
        dataset = dataset.cast_column("utterance", Audio(sampling_rate=config.SAMPLING_RATE))
        if config.MAX_SAMPLES:
            dataset = dataset.select(range(min(config.MAX_SAMPLES, len(dataset))))
        dataset = dataset.train_test_split(test_size=config.TEST_SIZE, seed=3407)
        raw_train, raw_test = dataset["train"], dataset["test"]
        self.logger.info(f"Train Size: {len(raw_train)}, Test Size: {len(raw_test)}")
        self.logger.info("Preprocessing dataset...")
        formatting_func = create_formatting_function(self.tokenizer)
        self.train_dataset = raw_train.map(formatting_func, remove_columns=raw_train.column_names, batched=False, desc="Cleaning Train Data")
        self.test_dataset = raw_test.map(formatting_func, remove_columns=raw_test.column_names, batched=False, desc="Cleaning Test Data")
        self.train_dataset = self.train_dataset.filter(remove_bad_examples)
        self.test_dataset = self.test_dataset.filter(remove_bad_examples)
        self.logger.info(f"Final Filtered Sizes -> Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}")

    def train(self):
        self.logger.info("Initializing Seq2SeqTrainer...")
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=config.GRAD_ACCUMULATION,
            learning_rate=self.args.learning_rate,
            warmup_steps=config.WARMUP_STEPS,
            num_train_epochs=self.args.epochs,
            logging_steps=config.LOGGING_STEPS,
            save_steps=config.SAVE_STEPS,
            eval_steps=config.EVAL_STEPS,
            save_total_limit=config.SAVE_TOTAL_LIMIT,
            optim="adamw_torch",
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            weight_decay=config.WEIGHT_DECAY,
            lr_scheduler_type=config.LR_SCHEDULER,
            warmup_ratio=config.WARMUP_RATIO,
            predict_with_generate=config.PREDICT_WITH_GENERATE,
            generation_max_length=config.MAX_LABEL_LENGTH,
            generation_num_beams=5,
            load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
            metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
            greater_is_better=config.GREATER_IS_BETTER,
            eval_strategy="steps",
            remove_unused_columns=False,
            label_names=["labels"],
            report_to=config.REPORT_TO,
            seed=3407,
            dataloader_num_workers=2,
        )
        compute_metrics_func = create_compute_metrics(self.tokenizer)
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=self.tokenizer),
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_func,
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.PATIENCE)])
        last_checkpoint = get_last_checkpoint(self.output_dir)
        if last_checkpoint:
            self.logger.info(f"Resuming from: {last_checkpoint}")
            self.trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            self.logger.info("Starting training...")
            self.trainer.train()

    def export_model(self):
        self.logger.info("Exporting model (16-bit)...")
        local_model_dir = os.path.join(self.output_dir, "final_model_merged")
        self.model.save_pretrained_merged(local_model_dir, self.tokenizer, save_method="merged_16bit")
        self.logger.info(f"Saved to: {local_model_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Fine-Tuning")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lora_rank", type=int, default=config.LORA_R)
    return parser.parse_args()

def main():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    args = parse_args()
    finetuner = WhisperFinetuner(args)
    finetuner.load_model()
    finetuner.prepare_data()
    finetuner.train()
    finetuner.export_model()
if __name__ == "__main__":
    main()
